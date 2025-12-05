"""Prompt templates for the LangGraph agent."""

planner_prompt = """For the given query {question}, come up with a
simple step by step plan of how to figure out the answer.

This plan should involve individual tasks, that if executed correctly
will yield the correct answer. Do not add any superfluous steps.
The result of the final step should be the final answer. Make sure
that each step has all the information needed - do not skip steps.
"""


break_down_plan_prompt_template = """You receive a plan {plan} which
contains a series of steps to follow in order to answer a query.
you need to go through the plan refine it according to this:
1. every step has to be able to be executed by either:
    i. retrieving relevant information from a vector store of book chunks
    ii. retrieving relevant information from a vector store of book quotes
    iii. answering a question from a given context.
2. every step should contain all the information needed to execute it.

output the refined plan
"""


replanner_prompt = """For the given objective, come up with a simple
step by step plan of how to figure out the answer.

This plan should involve individual tasks, that if executed correctly
will yield the correct answer.

Do not add any superfluous steps. The result of the final step should
be the final answer.

Make sure that each step has all the information needed - do not skip
steps.

Assume that the answer was not found yet and you need to update the
plan accordingly, so the plan should never be empty.

Your objective was this:
{question}

Your original plan was this:
{plan}

You have currently done the following steps:
{past_steps}

You already have the following context:
{aggregated_context}

Update your plan accordingly. If further steps are needed, fill out
the plan with only those steps.

Do not return previously done steps as part of the plan.
"""

tasks_handler_prompt_template = """
You are a task handler that receives a task: {curr_task} and must
decide which tool to use to execute the task.

You have the following tools at your disposal:

Tool A: Retrieves relevant information from a vector store of book
chunks based on a given query.
    - Use Tool A when the current task should search for information
      in the book chunks.

Tool B: Retrieves relevant information from a vector store of quotes
from the book based on a given query.
    - Use Tool B when the current task should search for information
      in the book quotes.

Tool C: Answers a question from a given context.
    - Use Tool C ONLY when the current task can be answered by the
      aggregated context: {aggregated_context}

Additional context for decision making:
- You also receive the last tool used: {last_tool}
    - If {last_tool} was retrieve_chunks, avoid using Tool A again;
      prefer other tools.
- You also have the past steps: {past_steps} to help understand the
  context of the task.
- You also have the initial user's question: {question} for
  additional context.

Instructions for output:
- If you decide to use Tools A or B, output the query to be used for
  the tool and specify the relevant tool.
- If you decide to use Tool C, output the question to be used for the
  tool, the context, and specify that the tool to be used is Tool C.
"""

keep_only_relevant_content_prompt_template = """
You receive a query: {query} and retrieved documents: {retrieved_documents} from a vector store.
You need to filter out all the non-relevant information that does not supply important information regarding the {query}.
Your goal is to filter out the non-relevant information only.
You can remove parts of sentences that are not relevant to the query or remove whole sentences that are not relevant to the query.
DO NOT ADD ANY NEW INFORMATION THAT IS NOT IN THE RETRIEVED DOCUMENTS.
Output the filtered relevant content.
"""
# Prompt template for checking if distilled content is grounded in the original context
is_distilled_content_grounded_on_content_prompt_template = """
You receive some distilled content: {distilled_content} and the original context: {original_context}.
You need to determine if the distilled content is grounded on the original context.
If the distilled content is grounded on the original context, set the grounded field to true.
If the distilled content is not grounded on the original context, set the grounded field to false.
{format_instructions}
"""

question_answer_cot_prompt_template = """ 
Chain-of-Thought Reasoning Examples

Example 1  
Context: Mary is taller than Jane. Jane is shorter than Tom. Tom is the same height as David.  
Question: Who is the tallest person?  
Reasoning:  
Mary > Jane  
Jane < Tom → Tom > Jane  
Tom = David  
So: Mary > Tom = David > Jane  
Final Answer: Mary  

Example 2  
Context: Harry read about three spells—one turns people into animals, one levitates objects, and one creates light.  
Question: If Harry cast these spells, what could he do?  
Reasoning:  
Spell 1: transform people into animals  
Spell 2: levitate things  
Spell 3: make light  
Final Answer: He could transform people, levitate objects, and create light  

Example 3  
Context: Harry Potter got a Nimbus 2000 broomstick for his birthday.  
Question: Why did Harry receive a broomstick?  
Reasoning:  
The context says he received a broomstick  
It doesn’t explain why or who gave it  
No info on hobbies or purpose  
Final Answer: Not enough context to know why he received it  

Now, follow the same pattern below.

Context:  
{context}  
Question:  
{question}  
"""
is_grounded_on_facts_prompt_template = """You are a fact-checker that determines if the given answer {answer} is grounded in the given context {context}
you don't mind if it doesn't make sense, as long as it is grounded in the context.
output a json containing the answer to the question, and appart from the json format don't output any additional text.
 """

can_be_answered_prompt_template = """You are an evaluator that determines if a question can be fully answered from the given context.

Question: {question}
Context: {context}

Your task is to determine if the context provides enough information to fully and comprehensively answer the question.

Return True if:
- The context contains all the necessary information to answer the question completely
- No important aspects of the question are left unanswered

Return False if:
- The context is missing key information needed to answer the question
- Only partial information is available
- Additional context or information would be needed for a complete answer

Provide a clear explanation for your decision.
"""

final_answer_prompt_template = """You are a comprehensive answer synthesizer that creates a complete, well-structured response based on all the gathered evidence.

Original Question: {question}

Aggregated Context and Evidence:
{aggregated_context}

Past Steps Taken:
{past_steps}

Your task is to:
1. Synthesize all the information from the aggregated context
2. Create a complete, coherent answer to the original question
3. Ensure the answer is well-structured and addresses all aspects of the question
4. Base your answer ONLY on the provided context - do not add external information
5. If multiple pieces of evidence support the answer, weave them together naturally

Provide a comprehensive final answer that fully addresses the user's question.
"""
