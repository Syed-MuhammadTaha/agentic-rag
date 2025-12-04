"""State schemas for the LangGraph agent."""

from typing import Annotated, List, TypedDict

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

    curr_state: str = Field(default="", description="current state of the agent")
    question: str = Field(default="", description="original user question")
    query_to_retrieve_or_answer: str = Field(
        default="", description="query to retrieve or answer"
    )
    plan: List[str] = Field(
        default_factory=list, description="plan to follow in future"
    )
    context: str = Field(default="", description="current context")
    past_steps: List[str] = Field(default_factory=list, description="past steps taken")
    mapping: dict = Field(default_factory=dict, description="mapping of steps to tools")
    curr_context: str = Field(default="", description="current context")
    aggregated_context: str = Field(default="", description="aggregated context")
    tool: str = Field(default="", description="tool to use")
    response: str = Field(default="", description="response from the tool")


class Plan(BaseModel):
    """Plan to follow in future."""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class ActPossibleResults(BaseModel):
    """Possible results of the replanning action."""

    plan: Plan = Field(description="Plan to follow in future.")
    explanation: str = Field(description="Explanation of the action.")


class TaskHandlerOutput(BaseModel):
    """Output schema for the task handler."""

    query: str = Field(
        description="The query to be either retrieved from the vector store, "
        "or the question that should be answered from context."
    )
    curr_context: str = Field(
        description="The context to be based on in order to answer the query."
    )
    tool: str = Field(
        description=(
            "The tool to be used should be either "
            "retrieve_chunks, retrieve_quotes, or answer_from_context."
        )
    )


class KeepRelevantContent(BaseModel):
    """Schema for keeping only relevant content from retrieved documents."""

    relevant_content: str = Field(
        description="The relevant content from the retrieved documents that is relevant to the query."
    )


class QualitativeRetrievalGraphState(TypedDict):
    """State for qualitative retrieval workflow."""

    question: str
    context: str
    relevant_context: str


class QuestionAnswerFromContext(BaseModel):
    """Schema for answering questions based on provided context."""

    answer_based_on_content: str = Field(
        description="Generates an answer to a query based on a given context."
    )


class IsDistilledContentGroundedOnContent(BaseModel):
    """Schema for checking if distilled content is grounded in the original context."""

    grounded: bool = Field(
        description="True if the distilled content is grounded on the original context, False otherwise."
    )
    explanation: str = Field(
        description="An explanation of why the distilled content is or is not grounded on the original context."
    )
