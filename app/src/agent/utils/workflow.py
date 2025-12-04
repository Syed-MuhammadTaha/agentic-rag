"""Workflow utilities for building retrieval graphs."""

from langgraph.graph import END, StateGraph

from agent.utils.retrieval_nodes import (
    is_distilled_content_grounded_on_content,
    keep_only_relevant_content,
)
from agent.utils.state import QualitativeRetrievalGraphState


def build_retrieval_workflow(node_name, retrieve_fn):
    """Build a retrieval workflow graph.

    Args:
        node_name: Name of the retrieval node.
        retrieve_fn: Function to retrieve content.

    Returns:
        Compiled StateGraph for retrieval workflow.
    """
    graph = StateGraph(QualitativeRetrievalGraphState)
    graph.add_node(node_name, retrieve_fn)
    graph.add_node("keep_only_relevant_content", keep_only_relevant_content)
    graph.set_entry_point(node_name)
    graph.add_edge(node_name, "keep_only_relevant_content")
    graph.add_conditional_edges(
        "keep_only_relevant_content",
        is_distilled_content_grounded_on_content,
        {
            "grounded on the original context": END,
            "not grounded on the original context": "keep_only_relevant_content",
        },
    )
    app = graph.compile()
    # Optionally display the graph (commented out for subgraph usage)
    # try:
    #     display(Image(app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)))
    # except Exception:
    #     pass  # IPython not available or error in visualization
    return app
