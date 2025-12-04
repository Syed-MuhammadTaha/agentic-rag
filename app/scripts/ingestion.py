"""Data ingestion pipeline for loading processed data into Qdrant collections using LangChain."""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Set

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import Config
from app.scripts.preprocessor import DataType, PreProcessor
from app.utils.embeddings import Embedder


@dataclass
class ProgressUpdate:
    """Progress update for ingestion process."""

    progress: int
    processed: int = 0
    total: int = 0
    message: str = ""
    error: Optional[str] = None

    def dict(self) -> Dict[str, Any]:
        """Convert progress update to dictionary."""
        return {
            "progress": self.progress,
            "processed": self.processed,
            "total": self.total,
            "message": self.message,
            "error": self.error,
        }


class QdrantIngestion:
    """Pipeline for ingesting documents into Qdrant collections using LangChain."""

    def __init__(self):
        """Initialize the ingestion pipeline with Qdrant client."""
        self.client = QdrantClient(url=Config.QDRANT_URL)
        self._embedder = None  # Lazy load embedder
        self._vector_stores = {}  # Lazy load vector stores

        # Define collection names and their vector dimensions
        self.collections = {
            DataType.CHUNKS: {"name": "book_chunks", "size": Config.VECTOR_SIZE},
            DataType.QUOTES: {"name": "book_quotes", "size": Config.VECTOR_SIZE},
        }

    @property
    def embedder(self):
        """Lazy load the embedder when needed."""
        if self._embedder is None:
            self._embedder = Embedder()  # Returns JinaEmbeddings instance
        return self._embedder

    def get_vector_store(self, collection_name: str) -> QdrantVectorStore:
        """Get or create a vector store for a collection."""
        if collection_name not in self._vector_stores:
            self._vector_stores[collection_name] = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embedder,
            )
        return self._vector_stores[collection_name]

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute MD5 hash of content for duplicate detection."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def get_existing_hashes(self, collection_name: str) -> Set[str]:
        """Retrieve all existing content hashes from a collection."""
        try:
            existing_hashes = set()

            # Check if collection exists and has points
            try:
                collection_info = self.client.get_collection(collection_name)
                if collection_info.points_count == 0:
                    return existing_hashes
            except Exception:
                # Collection doesn't exist yet
                return existing_hashes

            # Scroll through all points and collect hashes
            offset = None
            while True:
                records, offset = self.client.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,  # We don't need vectors, just metadata
                )

                if not records:
                    break

                for record in records:
                    if record.payload and "content_hash" in record.payload:
                        existing_hashes.add(record.payload["content_hash"])

                if offset is None:
                    break

            return existing_hashes

        except Exception as e:
            print(f"Error fetching existing hashes: {e}")
            return set()

    def cleanup_collections(self) -> None:
        """Safely delete all collections."""
        for collection_info in self.collections.values():
            collection_name = collection_info["name"]
            try:
                self.client.delete_collection(collection_name=collection_name)
                print(f"✓ Deleted collection: {collection_name}")
            except Exception as e:
                print(f"Error deleting collection {collection_name}: {e}")

    def setup_collections(self) -> None:
        """Set up Qdrant collections for chunks and quotes with payload indexing."""
        for collection_info in self.collections.values():
            collection_name = collection_info["name"]
            try:
                # Create collection
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=collection_info["size"], distance=Distance.COSINE
                    ),
                )

                # Create text index on content for duplicate detection
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="content",
                    field_schema="text",  # Enable full-text search on content
                )

                print(f"✓ Created collection and indices: {collection_name}")
            except Exception as e:
                if "already exists" in str(e):
                    print(f"Collection {collection_name} already exists")
                else:
                    print(f"Error creating collection {collection_name}: {e}")

    def process_batch(
        self, documents: List[Dict[str, Any]], collection_name: str
    ) -> int:
        """Process a batch of documents using LangChain's vector store."""
        batch_start_time = time.time()

        try:
            # Convert to LangChain documents and add content hash to metadata
            langchain_docs = []
            for doc in documents:
                # Compute content hash and add to metadata
                content_hash = self.compute_content_hash(doc["content"])
                metadata = doc["metadata"].copy()
                metadata["content_hash"] = content_hash

                langchain_docs.append(
                    Document(page_content=doc["content"], metadata=metadata)
                )

            # Generate UUIDs for the batch
            doc_ids = [str(uuid.uuid4()) for _ in documents]

            # Add documents to vector store
            vector_store = self.get_vector_store(collection_name)
            vector_store.add_documents(documents=langchain_docs, ids=doc_ids)

            batch_time = time.time() - batch_start_time
            print(
                f"Inserted {len(documents)} documents in {batch_time:.2f} seconds. "
                f"Speed: {len(documents) / batch_time:.2f} docs/sec"
            )
            return len(documents)

        except Exception as e:
            print(f"Error processing batch: {e}")
            return 0

    def ingest_documents(
        self, data_type: DataType, batch_size: int = Config.BATCH_SIZE
    ) -> Generator[str, None, None]:
        """Ingest documents into specified collection with progress updates."""
        collection_info = self.collections.get(data_type)
        if not collection_info:
            update = ProgressUpdate(progress=0, error=f"Unknown data type: {data_type}")
            yield json.dumps(update.dict()) + "\n"
            return

        # Initialize preprocessor and get documents
        preprocessor = PreProcessor()
        if data_type == DataType.CHUNKS:
            processed_data = preprocessor.get_chunks()
        else:  # DataType.QUOTES
            processed_data = preprocessor.get_quotes()

        documents = processed_data.documents
        total_documents = len(documents)

        if total_documents == 0:
            update = ProgressUpdate(progress=0, error="No documents to process")
            yield json.dumps(update.dict()) + "\n"
            return

        collection_name = collection_info["name"]

        # Get existing hashes once upfront (much faster than per-document checks)
        print(f"Checking for existing documents in {collection_name}...")
        existing_hashes = self.get_existing_hashes(collection_name)
        print(f"Found {len(existing_hashes)} existing documents")

        # Filter out duplicates and prepare documents for ingestion
        new_documents = []
        skipped_count = 0

        for doc in documents:
            content_hash = self.compute_content_hash(doc.page_content)

            if content_hash in existing_hashes:
                skipped_count += 1
                continue

            # Convert Document to dict format
            doc_dict = {"content": doc.page_content, "metadata": doc.metadata}
            new_documents.append(doc_dict)

        if skipped_count > 0:
            print(f"Skipped {skipped_count} duplicate documents")

        if len(new_documents) == 0:
            update = ProgressUpdate(
                progress=100,
                processed=0,
                total=total_documents,
                message=f"All {total_documents} documents already exist. Nothing to ingest.",
            )
            yield json.dumps(update.dict()) + "\n"
            return

        print(f"Ingesting {len(new_documents)} new documents...")
        processed_documents = 0

        # Process documents in batches
        current_batch = []
        for doc_dict in new_documents:
            current_batch.append(doc_dict)

            # Process batch when it reaches the size limit
            if len(current_batch) >= Config.BATCH_SIZE:
                processed = self.process_batch(current_batch, collection_name)
                processed_documents += processed
                current_batch = []

                # Send progress update
                progress = int((processed_documents / len(new_documents)) * 100)
                update = ProgressUpdate(
                    progress=progress,
                    processed=processed_documents,
                    total=len(new_documents),
                    message=f"Processed batch: {processed} documents",
                )
                yield json.dumps(update.dict()) + "\n"

        # Process remaining documents
        if current_batch:
            processed = self.process_batch(current_batch, collection_name)
            processed_documents += processed

        # Final update
        update = ProgressUpdate(
            progress=100,
            processed=processed_documents,
            total=len(new_documents),
            message=f"Completed ingestion for {data_type.value} ({skipped_count} duplicates skipped)",
        )
        yield json.dumps(update.dict()) + "\n"

    def ingest_all(
        self, batch_size: int = Config.BATCH_SIZE
    ) -> Generator[str, None, None]:
        """Ingest both chunks and quotes with progress updates."""
        total_progress = 0

        for data_type in [DataType.CHUNKS, DataType.QUOTES]:
            print(f"\nProcessing {data_type.value}...")

            for update_str in self.ingest_documents(data_type, batch_size):
                update = json.loads(update_str)

                # Adjust progress to account for multiple collections
                adjusted_progress = total_progress + (
                    update["progress"] / len(self.collections)
                )
                update["progress"] = int(adjusted_progress)

                yield json.dumps(update) + "\n"

            total_progress += 100 / len(self.collections)

        # Final completion message
        update = ProgressUpdate(
            progress=100, message="All collections processed successfully!"
        )
        yield json.dumps(update.dict()) + "\n"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument(
        "--setup", action="store_true", help="Setup collections before ingestion"
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="Delete all collections before setup"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Config.BATCH_SIZE,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--data-type",
        choices=["chunks", "quotes", "all"],
        default="all",
        help="Type of data to ingest",
    )

    args = parser.parse_args()

    # Initialize ingestion pipeline
    pipeline = QdrantIngestion()

    # Cleanup if requested
    if args.cleanup:
        pipeline.cleanup_collections()

    # Setup collections if requested
    if args.setup or args.cleanup:  # Always setup after cleanup
        pipeline.setup_collections()

    # Process based on data type
    if args.data_type == "all":
        generator = pipeline.ingest_all(args.batch_size)
    else:
        data_type = DataType.CHUNKS if args.data_type == "chunks" else DataType.QUOTES
        generator = pipeline.ingest_documents(data_type, args.batch_size)

    # Process and print updates
    try:
        for update_str in generator:
            update = json.loads(update_str)
            if update.get("error"):
                print(f"Error: {update['error']}")
            else:
                progress = update["progress"]
                message = update.get("message", "")
                print(f"Progress: {progress}% - {message}")
    except KeyboardInterrupt:
        print("\nIngestion interrupted by user")
    except Exception as e:
        print(f"Error during ingestion: {e}")
