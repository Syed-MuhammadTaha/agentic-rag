"""Prompt templates for the LangGraph agent."""

planner_prompt = """For the given query {question}, come up with a simple step by step plan of how to figure out the answer. 

This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
"""


break_down_plan_prompt_template = """You receive a plan {plan} which contains a series of steps to follow in order to answer a query. 
you need to go through the plan refine it according to this:
1. every step has to be able to be executed by either:
    i. retrieving relevant information from a vector store of book chunks
    ii. retrieving relevant information from a vector store of book quotes
    iii. answering a question from a given context.
2. every step should contain all the information needed to execute it.

output the refined plan
"""


replanner_prompt = """For the given objective, come up with a simple step by step plan of how to figure out the answer. 

This plan should involve individual tasks, that if executed correctly will yield the correct answer. 

Do not add any superfluous steps. The result of the final step should be the final answer. 

Make sure that each step has all the information needed - do not skip steps.

Assume that the answer was not found yet and you need to update the plan accordingly, so the plan should never be empty.

Your objective was this:
{question}

Your original plan was this:
{plan}

You have currently done the following steps:
{past_steps}

You already have the following context:
{aggregated_context}

Update your plan accordingly. If further steps are needed, fill out the plan with only those steps.

Do not return previously done steps as part of the plan.
"""

tasks_handler_prompt_template = """
You are a task handler that receives a task: {curr_task} and must decide which tool to use to execute the task.

You have the following tools at your disposal:

Tool A: Retrieves relevant information from a vector store of book chunks based on a given query.
    - Use Tool A when the current task should search for information in the book chunks.

Tool B: Retrieves relevant information from a vector store of quotes from the book based on a given query.
    - Use Tool B when the current task should search for information in the book quotes.

Tool C: Answers a question from a given context.
    - Use Tool C ONLY when the current task can be answered by the aggregated context: {aggregated_context}

Additional context for decision making:
- You also receive the last tool used: {last_tool}
    - If {last_tool} was retrieve_chunks, avoid using Tool A again; prefer other tools.
- You also have the past steps: {past_steps} to help understand the context of the task.
- You also have the initial user's question: {question} for additional context.

Instructions for output:
- If you decide to use Tools A or B, output the query to be used for the tool and specify the relevant tool.
- If you decide to use Tool C, output the question to be used for the tool, the context, and specify that the tool to be used is Tool C.
"""