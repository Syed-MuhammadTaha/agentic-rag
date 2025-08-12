from typing import Annotated, List, Optional

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Input(BaseModel):
    """Input schema for the LangGraph agent."""
    question: str

class State(BaseModel):
    """State schema for the LangGraph agent."""
    messages: Annotated[list, add_messages] = Field(default_factory=list)

class PlanExecute(BaseModel):
    """State schema for the PlanExecute node."""
    current_state: Optional[str] = None
    question: Optional[str] = None
    query_to_retrieve_or_answer: Optional[str] = None
    past_steps: Optional[List[str]] = None
    mapping: Optional[dict] = None
    curr_context: Optional[str] = None
    aggregated_context: Optional[str] = None
    plan: List[str] = Field(default_factory=list)
    tool: Optional[str] = None
    response: Optional[str] = None

class Plan(BaseModel):
    """Plan to follow in future."""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )