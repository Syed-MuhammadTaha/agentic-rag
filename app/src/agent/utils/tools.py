from typing import List

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import Config
from utils.embeddings import Embedder

# Initialize Qdrant stores
client = QdrantClient(url=Config.QDRANT_URL)
embedder = Embedder()

chunks_store = QdrantVectorStore(
    client=client,
    collection_name="book_chunks",
    embedding=embedder
)

quotes_store = QdrantVectorStore(
    client=client,
    collection_name="book_quotes",
    embedding=embedder
)

@tool
def search_chunks(query: str, k: int = 5) -> List[Document]:
    """Search for quotes in the book.

    Args:
        query: The search query
    Returns:
        List of relevant quotes
    """
    return chunks_store.similarity_search(query, k=k)

@tool
def search_quotes(query: str) -> List[Document]:
    """Search for quotes in the book.

    Args:
        query: The search query
    Returns:
        List of relevant quotes
    """
    return quotes_store.similarity_search(query)

@tool
def search_all(query: str, k: int = 3) -> List[Document]:
    """Search both chunks and quotes, returning a mix of results.
    
    Args:
    query: The search query
        k: Number of results to return from each source (default: 2)

    Returns:
        Combined list of chunks and quotes
    """
    return chunks_store.similarity_search(query, k=k) + quotes_store.similarity_search(query, k=k)