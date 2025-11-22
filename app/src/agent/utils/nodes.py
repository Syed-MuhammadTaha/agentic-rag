"""Node implementations for the LangGraph agent."""

from dotenv import load_dotenv
from langchain.chat_models.base import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from src.agent.utils.prompts import (
    break_down_plan_prompt_template,
    planner_prompt,
    replanner_prompt,
    tasks_handler_prompt_template,
)
from src.agent.utils.state import (
    ActPossibleResults,
    Input,
    Plan,
    PlanExecute,
    TaskHandlerOutput,
)

load_dotenv()

llm = init_chat_model("groq:llama-3.1-8b-instant")


def planner_node(state: Input) -> dict:
    """Generate initial plan from user question."""
    planner_prompt_template = ChatPromptTemplate.from_template(planner_prompt)
    planner_chain = planner_prompt_template | llm.with_structured_output(Plan)
    
    plan_result = planner_chain.invoke({"question": state.question})
    
    return {
        "question": state.question,
        "plan": plan_result.steps,
    }


def break_down_plan_node(state: PlanExecute) -> dict:
    """Refine plan to make each step executable by retrieval or QA."""
    breakdown_prompt_template = ChatPromptTemplate.from_template(
        break_down_plan_prompt_template
    )
    breakdown_chain = breakdown_prompt_template | llm.with_structured_output(Plan)
    
    refined_plan_result = breakdown_chain.invoke({"plan": state.plan})
    
    return {"plan": refined_plan_result.steps}


def replanner_node(state: PlanExecute) -> dict:
    """Update plan based on past steps and aggregated context."""
    replanner_prompt_template = ChatPromptTemplate.from_template(replanner_prompt)
    replanner_chain = replanner_prompt_template | llm.with_structured_output(
        ActPossibleResults
    )
    
    result = replanner_chain.invoke({
        "question": state.question,
        "plan": state.plan,
        "past_steps": state.past_steps,
        "aggregated_context": state.aggregated_context,
    })
    
    return {"plan": result.plan.steps}


def task_handler_node(state: PlanExecute) -> dict:
    """Decide which tool to use for the current task."""
    task_handler_prompt_template = ChatPromptTemplate.from_template(
        tasks_handler_prompt_template
    )
    task_handler_chain = task_handler_prompt_template | llm.with_structured_output(
        TaskHandlerOutput
    )
    
    # Get current task (first item in plan)
    curr_task = state.plan[0] if state.plan else ""
    
    result = task_handler_chain.invoke({
        "curr_task": curr_task,
        "aggregated_context": state.aggregated_context,
        "last_tool": state.tool,
        "past_steps": state.past_steps,
        "question": state.question,
    })
    
    return {
        "query_to_retrieve_or_answer": result.query,
        "curr_context": result.curr_context,
        "tool": result.tool,
    }
