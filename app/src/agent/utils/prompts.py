"""Prompt templates for the LangGraph agent."""

planner_prompt = """Create a concise step-by-step plan to answer: {question}

Requirements:
- List only essential steps
- Each step must be self-contained
- Final step should produce the answer
- Maximum 5 steps

Respond ONLY with valid JSON in this EXACT format:
{{
  "steps": [
    "First step description",
    "Second step description",
    "Third step description"
  ]
}}

IMPORTANT: Each step must be a STRING, not an object. Do NOT use {{"step": "...", "description": "..."}}.
"""


break_down_plan_prompt_template = """Refine this plan: {plan}

Each step must use ONE of:
1. Retrieve from book chunks database
2. Retrieve from quotes database  
3. Answer from existing context

Ensure steps are executable and self-contained.

Respond ONLY with JSON in this EXACT format:
{{
  "steps": [
    "Retrieve information about X from book chunks database",
    "Retrieve quotes about Y from quotes database",
    "Answer question Z from existing context"
  ]
}}

IMPORTANT: Each step must be a STRING, not an object. Do NOT use {{"step": "...", "description": "..."}}.
"""


replanner_prompt = """Update the plan to answer: {question}

Original plan: {plan}
Completed steps: {past_steps}
Current context: {aggregated_context}

Provide ONLY remaining steps needed (never empty). Exclude completed steps.

Respond ONLY with JSON in this EXACT format:
{{
  "plan": {{
    "steps": [
      "Remaining step 1 as a string",
      "Remaining step 2 as a string"
    ]
  }},
  "explanation": "Brief explanation here"
}}

IMPORTANT: Each step must be a STRING, not an object. Do NOT use {{"step": "...", "description": "..."}}.
"""

tasks_handler_prompt_template = """
Task: {curr_task}

Select tool:
1. "retrieve_chunks" - Search book chapters/sections
2. "retrieve_quotes" - Search book quotes
3. "answer_from_context" - Use existing context

Last tool: {last_tool}
(Avoid using same tool twice in a row)

JSON format:
{{"query": "search text", "curr_context": "", "tool": "retrieve_chunks"}}
"""

keep_only_relevant_content_prompt_template = """
Query: {query}
Documents: {retrieved_documents}

Filter out irrelevant information. Keep only text relevant to the query.
Do NOT add new information.

JSON: {{"relevant_content": "filtered text"}}
"""
# Prompt template for checking if distilled content is grounded in the original context
is_distilled_content_grounded_on_content_prompt_template = """
Determine if the distilled content is grounded in the original context.

Distilled Content: {distilled_content}

Original Context: {original_context}

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{{
  "grounded": true,
  "explanation": "your explanation here"
}}

Do NOT write code, do NOT add any text outside the JSON. Just the JSON object.
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

Respond ONLY with JSON: {{"answer_based_on_content": "your answer here"}}
"""
is_grounded_on_facts_prompt_template = """Is this answer grounded in the context?

Context: {context}
Answer: {answer}

Respond ONLY with valid JSON:
{{"grounded_on_facts": true}} or {{"grounded_on_facts": false}}

No code, no explanation, just JSON.
"""

can_be_answered_prompt_template = """Determine if the question can be satisfactorily answered using the provided context.

Question: {question}

Available Context:
{context}

Evaluation Criteria:
- Does the context contain relevant information to answer the question?
- Can a reasonable answer be formed from this context?
- If the context has SOME useful information (even if not complete), answer 'true'
- Only answer 'false' if the context is empty or completely unrelated

IMPORTANT: Be generous in evaluation. If you have gathered ANY relevant context that helps answer the question, return true.

Respond ONLY with valid JSON in this format:
{{
  "can_be_answered": true,
  "explanation": "brief explanation"
}}

No code, just JSON.
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

Respond ONLY with JSON: {{"final_answer": "your comprehensive answer here"}}

No code, just JSON.
"""
