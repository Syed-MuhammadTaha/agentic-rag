"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""
from dataclasses import dataclass
from typing import Annotated, Any, Dict

from dotenv import load_dotenv
from langchain.chat_models.base import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()



@dataclass
class State(TypedDict):
    """State schema for the LangGraph agent."""
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = init_chat_model("groq:llama-3.1-8b-instant")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

