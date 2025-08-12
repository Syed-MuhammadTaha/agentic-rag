planner_prompt =""" For the given query {question}, create a clear, step-by-step plan to arrive at the final answer.  
Each step must be self-contained, with all the information needed to execute it, and no unnecessary steps.  
Every step should be executable by exactly one of the following methods:  
    1. Retrieving relevant information from a vector store of book chunks  
    2. Retrieving relevant information from a vector store of book quotes  
    3. Answering a question from a given context  

Ensure that completing the final step yields the final answer.  
Output only the refined plan.

"""
