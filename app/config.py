"""Configuration settings for the application."""

import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for the application."""

    # Vector Database Configuration
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

    # Embedding Service Configuration - Jina Cloud API
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    EMBEDDING_MODEL = "jina-embeddings-v3"
    VECTOR_SIZE = 1024  # Jina v3 default dimension

    # Data Processing Configuration
    MIN_QUOTE_LENGTH = 50
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Ingestion Configuration
    BATCH_SIZE = 10

    DATA_PATH = "app/data/Harry Potter - Book 1 - The Sorcerers Stone.pdf"
    OUTPUT_PATH = "app/data/processed"
