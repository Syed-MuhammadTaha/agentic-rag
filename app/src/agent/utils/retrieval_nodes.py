"""Retrieval workflow node implementations."""

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama

from agent.utils.prompts import (
    is_distilled_content_grounded_on_content_prompt_template,
    keep_only_relevant_content_prompt_template,
)
from agent.utils.state import (
    Input,
    IsDistilledContentGroundedOnContent,
    KeepRelevantContent,
    QualitativeRetrievalGraphState,
)
from agent.utils.tools import search_chunks, search_quotes

load_dotenv()

llm = ChatOllama(model="llama3.1:8b", temperature=0.0)


def retrieve_book_quotes_context_per_question(state: Input):
    """Retrieve book quotes context for the given question."""
    question = state["question"]

    docs_book_quotes = search_quotes(question)
    book_qoutes = " ".join(doc.page_content for doc in docs_book_quotes)
    book_qoutes_context = book_qoutes.replace('"', '\\"').replace(
        "'", "\\'"
    )  # Escape quotes for downstream processing

    return {"context": book_qoutes_context, "question": question}


def retrieve_chunks_context_per_question(state: Input):
    """Retrieve relevant context for a given question. The context is retrieved from the book chunks and chapter summaries."""
    # Retrieve relevant documents
    question = state["question"]
    docs = search_chunks(question)

    # Concatenate document content
    context = " ".join(doc.page_content for doc in docs)
    context = context.replace('"', '\\"').replace(
        "'", "\\'"
    )  # Escape quotes for downstream processing
    return {"context": context, "question": question}


def keep_only_relevant_content(state):
    """Filter and retain only the content from the retrieved documents that is relevant to the query."""
    question = state["question"]
    context = state["context"]

    # Prepare input for the LLM chain
    input_data = {"query": question, "retrieved_documents": context}

    # Create prompt template from the prompt string
    keep_only_relevant_content_prompt = ChatPromptTemplate.from_template(
        keep_only_relevant_content_prompt_template
    )

    keep_only_relevant_content_chain = (
        keep_only_relevant_content_prompt
        | llm.with_structured_output(KeepRelevantContent)
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

    distilled_content = state["relevant_context"]
    original_context = state["context"]

    # Prepare input for the LLM chain
    input_data = {
        "distilled_content": distilled_content,
        "original_context": original_context,
    }
    # Output parser for the LLM response
    is_distilled_content_grounded_on_content_json_parser = JsonOutputParser(
        pydantic_object=IsDistilledContentGroundedOnContent
    )

    # PromptTemplate for the LLM
    is_distilled_content_grounded_on_content_prompt = PromptTemplate(
        template=is_distilled_content_grounded_on_content_prompt_template,
        input_variables=["distilled_content", "original_context"],
        partial_variables={
            "format_instructions": is_distilled_content_grounded_on_content_json_parser.get_format_instructions()
        },
    )
    is_distilled_content_grounded_on_content_chain = (
        is_distilled_content_grounded_on_content_prompt
        | llm
        | is_distilled_content_grounded_on_content_json_parser
    )
    # Invoke the LLM chain to check grounding
    output = is_distilled_content_grounded_on_content_chain.invoke(input_data)
    grounded = output["grounded"]

    # Return result based on grounding
    if grounded:
        return "grounded on the original context"
    else:
        return "not grounded on the original context"
