"""Node implementations for the LangGraph agent."""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama

from agent.utils.prompts import (
    break_down_plan_prompt_template,
    final_answer_prompt_template,
    planner_prompt,
    question_answer_cot_prompt_template,
    replanner_prompt,
    tasks_handler_prompt_template,
)
from agent.utils.state import (
    ActPossibleResults,
    FinalAnswer,
    Input,
    Plan,
    PlanExecute,
    QuestionAnswerFromContext,
    TaskHandlerOutput,
)

load_dotenv()

llm = ChatOllama(model="llama3.1:8b", temperature=0.0)


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
    """Refine plan to make steps executable by retrieval or QA."""
    breakdown_prompt_template = ChatPromptTemplate.from_template(
        break_down_plan_prompt_template
    )
    breakdown_chain = breakdown_prompt_template | llm.with_structured_output(Plan)

    refined_plan_result = breakdown_chain.invoke({"plan": state.plan})

    return {"plan": refined_plan_result.steps}


def replanner_node(state: PlanExecute) -> dict:
    """Update plan based on past steps and context."""
    replanner_prompt_template = ChatPromptTemplate.from_template(replanner_prompt)
    replanner_chain = replanner_prompt_template | llm.with_structured_output(
        ActPossibleResults
    )

    result = replanner_chain.invoke(
        {
            "question": state.question,
            "plan": state.plan,
            "past_steps": state.past_steps,
            "aggregated_context": state.aggregated_context,
        }
    )

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

    result = task_handler_chain.invoke(
        {
            "curr_task": curr_task,
            "aggregated_context": state.aggregated_context,
            "last_tool": state.tool,
            "past_steps": state.past_steps,
            "question": state.question,
        }
    )

    return {
        "query_to_retrieve_or_answer": result.query,
        "curr_context": result.curr_context,
        "tool": result.tool,
    }


def answer_question_from_context_node(state: PlanExecute):
    """Answers a question from a given context using a chain-of-thought LLM chain.

    Args:
        state (dict): A dictionary containing:
            - "question": The query question.
            - "context": The context to answer the question from.
            - Optionally, "aggregated_context": an aggregated context to use instead.

    Returns:
        dict: A dictionary with:
            - "answer": The generated answer.
            - "context": The context used.
            - "question": The original question.
    """
    # Extract the question from the state
    question = state["question"]
    # Use "aggregated_context" if present, otherwise use "context"
    context = (
        state["aggregated_context"]
        if "aggregated_context" in state
        else state["context"]
    )

    input_data = {"question": question, "context": context}

    question_answer_from_context_cot_prompt = PromptTemplate(
        template=question_answer_cot_prompt_template,  # Uses examples and instructions for step-by-step reasoning
        input_variables=[
            "context",
            "question",
        ],  # Expects 'context' and 'question' as inputs
    )
    question_answer_from_context_cot_chain = (
        question_answer_from_context_cot_prompt
        | llm.with_structured_output(QuestionAnswerFromContext)
    )
    # Invoke the chain-of-thought LLM chain to generate an answer
    output = question_answer_from_context_cot_chain.invoke(input_data)
    answer = output.answer_based_on_content
    # Return the answer, context, and question in a dictionary
    return {"answer": answer, "context": context, "question": question}


def get_final_answer_node(state: PlanExecute) -> dict:
    """Synthesize all gathered evidence into a comprehensive final answer.

    Args:
        state: A PlanExecute state containing:
            - "question": The original user question.
            - "aggregated_context": All gathered context and evidence.
            - "past_steps": List of steps taken to gather information.

    Returns:
        dict: A dictionary with:
            - "response": The final synthesized answer.
    """
    question = state.question
    aggregated_context = state.aggregated_context
    past_steps = state.past_steps

    # Prepare input for the final answer chain
    input_data = {
        "question": question,
        "aggregated_context": aggregated_context,
        "past_steps": past_steps,
    }

    # Create the prompt template for final answer synthesis
    final_answer_prompt = ChatPromptTemplate.from_template(final_answer_prompt_template)

    # Build the chain: prompt -> LLM -> structured output
    final_answer_chain = final_answer_prompt | llm.with_structured_output(FinalAnswer)

    # Invoke the chain to synthesize the final answer
    output = final_answer_chain.invoke(input_data)
    final_response = output.final_answer

    return {"response": final_response}
