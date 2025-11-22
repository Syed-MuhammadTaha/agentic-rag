"""Embeddings using Jina Cloud API via LangChain integration."""

from langchain_community.embeddings import JinaEmbeddings

from app.config import Config


def Embedder():
    """Create and return a Jina embeddings instance."""
    if not Config.JINA_API_KEY:
        raise ValueError("JINA_API_KEY environment variable is not set")
    
    return JinaEmbeddings(
        jina_api_key=Config.JINA_API_KEY,
        model_name=Config.EMBEDDING_MODEL,
    )


if __name__ == "__main__":
    embedder = Embedder()
    print(f"Query embedding dimension: {len(embedder.embed_query('Hello, world!'))}")
    print(
        f"Document embeddings count: {len(embedder.embed_documents(['Hello, world!', 'Hello, world!']))}"
    )