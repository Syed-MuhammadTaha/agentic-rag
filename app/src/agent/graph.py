"""Plan-Execute LangGraph for RAG agent."""

from langgraph.graph import END, START, StateGraph

from agent.utils.nodes import (
    answer_question_from_context_node,
    break_down_plan_node,
    get_final_answer_node,
    planner_node,
    replanner_node,
    task_handler_node,
)
from agent.utils.retrieval_nodes import (
    can_question_be_answered,
    is_answer_grounded_on_context,
    retrieve_book_quotes_context_per_question,
    retrieve_chunks_context_per_question,
)
from agent.utils.state import Input, PlanExecute
from agent.utils.workflow import build_retrieval_workflow

graph_builder = StateGraph(PlanExecute, input=Input)

# Add nodes
graph_builder.add_node("planner", planner_node)
graph_builder.add_node("break_down_plan", break_down_plan_node)
graph_builder.add_node("task_handler", task_handler_node)
graph_builder.add_node("answer", answer_question_from_context_node)
graph_builder.add_node("get_final_answer", get_final_answer_node)

# Add retrieval workflow subgraphs
chunks_retrieval_workflow = build_retrieval_workflow(
    "retrieve_chunks_context_per_question", retrieve_chunks_context_per_question
)
quotes_retrieval_workflow = build_retrieval_workflow(
    "retrieve_book_quotes_context_per_question",
    retrieve_book_quotes_context_per_question,
)

graph_builder.add_node("retrieve_chunks", chunks_retrieval_workflow)
graph_builder.add_node("retrieve_quotes", quotes_retrieval_workflow)
graph_builder.add_node("replanner", replanner_node)


def route_based_on_tool(state: PlanExecute) -> str:
    """Route to the appropriate retrieval workflow based on the tool selected."""
    tool = state.tool
    if tool == "retrieve_chunks":
        return "retrieve_chunks"
    elif tool == "retrieve_quotes":
        return "retrieve_quotes"
    elif tool == "answer_from_context":
        return "answer"


# Add edges
graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "break_down_plan")
graph_builder.add_edge("break_down_plan", "task_handler")

# Add conditional edges from task_handler to retrieval workflows
graph_builder.add_conditional_edges(
    "task_handler",
    route_based_on_tool,
    {
        "retrieve_chunks": "retrieve_chunks",
        "retrieve_quotes": "retrieve_quotes",
        "answer": "answer",
    },
)

# Add edges from retrieval workflows back to replanner
graph_builder.add_edge("retrieve_chunks", "replanner")
graph_builder.add_edge("retrieve_quotes", "replanner")

# Conditional edge from replanner: check if question can be answered
graph_builder.add_conditional_edges(
    "replanner",
    can_question_be_answered,
    {"useful": "get_final_answer", "not_useful": "break_down_plan"},
)

# get_final_answer goes directly to END
graph_builder.add_edge("get_final_answer", END)

# Conditional edge from answer: check if answer is grounded
graph_builder.add_conditional_edges(
    "answer",
    is_answer_grounded_on_context,
    {"hallucination": "answer", "grounded on context": "replanner"},
)

graph = graph_builder.compile()
