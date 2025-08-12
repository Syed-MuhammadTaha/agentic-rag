"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""
from dataclasses import dataclass
from typing import Annotated

from langgraph.graph import END, START, StateGraph

from src.agent.utils.nodes import planner_node
from src.agent.utils.state import PlanExecute

graph_builder = StateGraph(PlanExecute)


graph_builder.add_node("planner", planner_node)
graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", END)
graph = graph_builder.compile()

