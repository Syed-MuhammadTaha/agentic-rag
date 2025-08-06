"""Tools for the agent graph."""

from typing import Any, Dict, List


class VectorSearchTool:
    """Tool for searching vector databases."""
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search the vector database for relevant documents."""
        # TODO: Implement vector search using Qdrant
        pass


class DocumentRetrievalTool:
    """Tool for retrieving documents from multiple sources."""
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents."""
        # TODO: Implement document retrieval logic
        pass