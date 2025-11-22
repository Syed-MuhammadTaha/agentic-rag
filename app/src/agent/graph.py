"""Plan-Execute LangGraph for RAG agent."""

from langgraph.graph import END, START, StateGraph

from src.agent.utils.nodes import break_down_plan_node, planner_node
from src.agent.utils.state import Input, PlanExecute

graph_builder = StateGraph(PlanExecute, input=Input)

# Add nodes
graph_builder.add_node("planner", planner_node)
graph_builder.add_node("break_down_plan", break_down_plan_node)

# Add edges
graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "break_down_plan")
graph_builder.add_edge("break_down_plan", END)

graph = graph_builder.compile()
