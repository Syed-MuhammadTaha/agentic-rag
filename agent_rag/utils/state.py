"""State definition for the agent graph."""

from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State structure for the agent graph."""
    
    query: str
    retrieved_docs: List[Dict[str, Any]]
    response: str
    target_collection: Optional[str]
    conversation_history: List[Dict[str, str]]
    metadata: Dict[str, Any]