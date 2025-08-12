from typing import Dict

from dotenv import load_dotenv
from langchain.chat_models.base import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tools import tool

from src.agent.utils.prompts import planner_prompt
from src.agent.utils.state import Input, Plan, PlanExecute
from src.agent.utils.tools import search_all, search_chunks, search_quotes

load_dotenv()

llm = init_chat_model("groq:llama-3.1-8b-instant")

def planner_node(state: Input) -> PlanExecute:
    """Planner node for the LangGraph agent."""
    planner_prompt_template = ChatPromptTemplate.from_template(planner_prompt)
    planner_chain = planner_prompt_template | llm.with_structured_output(Plan)
    return PlanExecute(
        question=state.question,
        plan=planner_chain.invoke({"question": state.question})
    )