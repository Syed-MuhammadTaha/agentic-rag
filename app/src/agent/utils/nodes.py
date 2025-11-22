
from dotenv import load_dotenv
from langchain.chat_models.base import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from src.agent.utils.prompts import planner_prompt
from src.agent.utils.state import Input, Plan, PlanExecute

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