"""Data ingestion pipeline for loading processed data into Qdrant collections."""

import json
import time
import uuid
from typing import List, Dict, Any, Generator, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from config import Config
from embeddings import Embedder
from preprocessor import PreProcessor, DataType

@dataclass
class ProgressUpdate:
    """Progress update for ingestion process."""
    progress: int
    processed: int = 0
    total: int = 0
    message: str = ""
    error: Optional[str] = None

    def dict(self) -> Dict[str, Any]:
        return {
            "progress": self.progress,
            "processed": self.processed,
            "total": self.total,
            "message": self.message,
            "error": self.error
        }


class QdrantIngestion:
    """Pipeline for ingesting documents into Qdrant collections."""

    def __init__(self):
        """Initialize the ingestion pipeline with Qdrant client."""
        self.client = QdrantClient(url=Config.QDRANT_URL)
        self._embedder = None  # Lazy load embedder
        
        # Define collection names and their vector dimensions
        self.collections = {
            DataType.CHUNKS: {
                "name": "book_chunks",
                "size": Config.VECTOR_SIZE
            },
            DataType.QUOTES: {
                "name": "book_quotes",
                "size": Config.VECTOR_SIZE
            }
        }

    @property
    def embedder(self) -> Embedder:
        """Lazy load the embedder when needed."""
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

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
                        size=collection_info["size"],
                        distance=Distance.COSINE
                    )
                )
                
                # Create text index on content for duplicate detection
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="content",
                    field_schema="text"  # Enable full-text search on content
                )
                
                print(f"✓ Created collection and indices: {collection_name}")
            except Exception as e:
                if "already exists" in str(e):
                    print(f"Collection {collection_name} already exists")
                else:
                    print(f"Error creating collection {collection_name}: {e}")

    def document_exists(self, content: str, collection_name: str) -> bool:
        """Check if a document with similar content exists in the collection using text search."""
        try:
            # First try exact text match
            results = self.client.scroll(
                collection_name=collection_name,
                limit=1,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="content",
                            match=MatchValue(value=content)
                        )
                    ]
                )
            )[0]
            
            return len(results) > 0
        except Exception as e:
            print(f"Error checking document existence: {e}")
            return False

    def process_batch(
        self,
        documents: List[Dict[str, Any]],
        collection_name: str
    ) -> int:
        """Process a batch of documents and insert into Qdrant."""
        batch_start_time = time.time()

        try:
            # Prepare texts and metadata
            texts = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            # Generate embeddings
            embeddings = self.embedder.embed_documents(texts)
            if not embeddings:
                print("Failed to generate embeddings for batch")
                return 0

            # Prepare points for Qdrant
            points = []
            for text, metadata, embedding in zip(texts, metadatas, embeddings):
                # Keep payload structure simple
                payload = {
                    "content": text,
                    "metadata": metadata
                }
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload
                )
                points.append(point)

            # Insert into Qdrant
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )

            batch_time = time.time() - batch_start_time
            print(
                f"Inserted {len(texts)} documents in {batch_time:.2f} seconds. "
                f"Speed: {len(texts) / batch_time:.2f} docs/sec"
            )
            return len(texts)

        except Exception as e:
            print(f"Error processing batch: {e}")
            return 0

    def ingest_documents(
        self,
        data_type: DataType,
        batch_size: int = Config.BATCH_SIZE
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

        processed_documents = 0
        collection_name = collection_info["name"]

        # Process documents in batches
        current_batch = []
        for doc in documents:
            # Convert Document to dict format
            doc_dict = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }

            # Check for duplicates
            if self.document_exists(doc.page_content, collection_name):
                update = ProgressUpdate(
                    progress=int((processed_documents / total_documents) * 100),
                    processed=processed_documents,
                    total=total_documents,
                    message="Skipped duplicate document"
                )
                yield json.dumps(update.dict()) + "\n"
                continue

            current_batch.append(doc_dict)

            # Process batch when it reaches the size limit
            if len(current_batch) >= batch_size:
                processed = self.process_batch(current_batch, collection_name)
                processed_documents += processed
                current_batch = []

                # Send progress update
                progress = int((processed_documents / total_documents) * 100)
                update = ProgressUpdate(
                    progress=progress,
                    processed=processed_documents,
                    total=total_documents,
                    message=f"Processed batch: {processed} documents"
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
            total=total_documents,
            message=f"Completed ingestion for {data_type.value}"
        )
        yield json.dumps(update.dict()) + "\n"

    def ingest_all(self, batch_size: int = Config.BATCH_SIZE) -> Generator[str, None, None]:
        """Ingest both chunks and quotes with progress updates."""
        total_progress = 0
        
        for data_type in [DataType.CHUNKS, DataType.QUOTES]:
            print(f"\nProcessing {data_type.value}...")
            
            for update_str in self.ingest_documents(data_type, batch_size):
                update = json.loads(update_str)
                
                # Adjust progress to account for multiple collections
                adjusted_progress = total_progress + (update["progress"] / len(self.collections))
                update["progress"] = int(adjusted_progress)
                
                yield json.dumps(update) + "\n"
            
            total_progress += 100 / len(self.collections)

        # Final completion message
        update = ProgressUpdate(
            progress=100,
            message="All collections processed successfully!"
        )
        yield json.dumps(update.dict()) + "\n"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument("--setup", action="store_true", help="Setup collections before ingestion")
    parser.add_argument("--cleanup", action="store_true", help="Delete all collections before setup")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE, help="Batch size for processing")
    parser.add_argument("--data-type", choices=["chunks", "quotes", "all"], 
                       default="all", help="Type of data to ingest")
    
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