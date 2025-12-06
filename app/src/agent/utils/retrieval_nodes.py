"""Retrieval workflow node implementations."""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama

from agent.utils.prompts import (
    can_be_answered_prompt_template,
    is_distilled_content_grounded_on_content_prompt_template,
    is_grounded_on_facts_prompt_template,
    keep_only_relevant_content_prompt_template,
)
from agent.utils.state import (
    CanBeAnswered,
    GroundedOnFacts,
    IsDistilledContentGroundedOnContent,
    KeepRelevantContent,
    QualitativeRetrievalGraphState,
)
from agent.utils.tools import search_chunks, search_quotes

load_dotenv()

# Optimized LLM configuration for faster inference
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.1,  # Slightly higher for faster sampling
    num_predict=512,  # Limit output tokens
    num_ctx=2048,  # Reduce context window
    format="json",  # Force JSON output mode
)


def retrieve_book_quotes_context_per_question(state):
    """Retrieve book quotes context for the given question."""
    # Handle both Pydantic models and dict
    question = state.question if hasattr(state, "question") else state["question"]

    docs_book_quotes = search_quotes(question)
    book_qoutes = " ".join(doc.page_content for doc in docs_book_quotes)
    book_qoutes_context = book_qoutes.replace('"', '\\"').replace(
        "'", "\\'"
    )  # Escape quotes for downstream processing

    return {"context": book_qoutes_context, "question": question}


def retrieve_chunks_context_per_question(state):
    """Retrieve relevant context for a given question. The context is retrieved from the book chunks and chapter summaries."""
    # Handle both Pydantic models and dict
    question = state.question if hasattr(state, "question") else state["question"]
    docs = search_chunks(question)

    # Concatenate document content
    context = " ".join(doc.page_content for doc in docs)
    context = context.replace('"', '\\"').replace(
        "'", "\\'"
    )  # Escape quotes for downstream processing
    return {"context": context, "question": question}


def keep_only_relevant_content(state):
    """Filter and retain only the content from the retrieved documents that is relevant to the query."""
    # Handle both Pydantic models and dict
    question = getattr(
        state, "question", state.get("question", "") if isinstance(state, dict) else ""
    )
    context = getattr(
        state, "context", state.get("context", "") if isinstance(state, dict) else ""
    )

    # Prepare input for the LLM chain
    input_data = {"query": question, "retrieved_documents": context}

    # Create prompt template from the prompt string
    keep_only_relevant_content_prompt = ChatPromptTemplate.from_template(
        keep_only_relevant_content_prompt_template
    )

    keep_only_relevant_content_chain = (
        keep_only_relevant_content_prompt
        | llm.with_structured_output(KeepRelevantContent, method="json_mode")
    )
    # Invoke the LLM chain to filter out non-relevant content
    output = keep_only_relevant_content_chain.invoke(input_data)
    relevant_content = output.relevant_content

    # Ensure the result is a string (in case it's not)
    relevant_content = "".join(relevant_content)

    # Escape quotes for downstream processing
    relevant_content = relevant_content.replace('"', '\\"').replace("'", "\\'")

    return {
        "relevant_context": relevant_content,
        "context": context,
        "question": question,
    }


def is_distilled_content_grounded_on_content(
    state: QualitativeRetrievalGraphState,
) -> str:
    """Determine if the distilled content is grounded in the original context."""
    # Extract distilled content and original context from state
    # Handle both Pydantic models and dict
    distilled_content = getattr(
        state,
        "relevant_context",
        state.get("relevant_context", "") if isinstance(state, dict) else "",
    )
    original_context = getattr(
        state, "context", state.get("context", "") if isinstance(state, dict) else ""
    )

    # Prepare input for the LLM chain
    input_data = {
        "distilled_content": distilled_content,
        "original_context": original_context,
    }

    # Create prompt template and chain with structured output
    is_distilled_content_grounded_on_content_prompt = ChatPromptTemplate.from_template(
        is_distilled_content_grounded_on_content_prompt_template
    )

    # Force JSON mode with format parameter
    is_distilled_content_grounded_on_content_chain = (
        is_distilled_content_grounded_on_content_prompt
        | llm.with_structured_output(
            IsDistilledContentGroundedOnContent, method="json_mode"
        )
    )

    # Invoke the LLM chain to check grounding
    output = is_distilled_content_grounded_on_content_chain.invoke(input_data)
    grounded = output.grounded

    # Return result based on grounding
    if grounded:
        return "grounded on the original context"
    else:
        return "not grounded on the original context"


def is_answer_grounded_on_context(state):
    """Determine if the answer to the question is grounded in the facts.

    Args:
        state: A dictionary containing the context and answer.

    Returns:
        "hallucination" if the answer is not grounded in the context,
        "grounded on context" if the answer is grounded in the context.
    """
    # Handle both Pydantic models and dict
    context = getattr(
        state, "context", state.get("context", "") if isinstance(state, dict) else ""
    )
    answer = getattr(
        state, "answer", state.get("answer", "") if isinstance(state, dict) else ""
    )
    # Create the prompt object
    is_grounded_on_facts_prompt = PromptTemplate(
        template=is_grounded_on_facts_prompt_template,
        input_variables=["context", "answer"],
    )

    # Build the chain: prompt -> LLM -> structured output
    is_grounded_on_facts_chain = (
        is_grounded_on_facts_prompt
        | llm.with_structured_output(GroundedOnFacts, method="json_mode")
    )

    # Use the is_grounded_on_facts_chain to check if the answer is grounded in the context
    result = is_grounded_on_facts_chain.invoke({"context": context, "answer": answer})
    grounded_on_facts = result.grounded_on_facts

    if not grounded_on_facts:
        return "hallucination"
    else:
        return "grounded on context"


def can_question_be_answered(state):
    """Check if the question can be fully answered from the aggregated context.

    Returns: "useful" if it can be answered, "not_useful" if more context is needed.
    """
    # Handle both Pydantic models and dict
    question = state.question if hasattr(state, "question") else state["question"]
    aggregated_context = (
        state.aggregated_context
        if hasattr(state, "aggregated_context")
        else state.get("aggregated_context", "")
    )

    # Create the prompt for checking if the question can be answered
    can_be_answered_prompt = ChatPromptTemplate.from_template(
        can_be_answered_prompt_template
    )

    # Build the chain: prompt -> LLM -> structured output
    can_be_answered_chain = can_be_answered_prompt | llm.with_structured_output(
        CanBeAnswered, method="json_mode"
    )

    # Check if the question can be fully answered from the aggregated context
    result = can_be_answered_chain.invoke(
        {"question": question, "context": aggregated_context}
    )

    return "useful" if result.can_be_answered else "not_useful"
