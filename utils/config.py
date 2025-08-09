"""Configuration settings for the application."""

class Config:
    """Configuration class for the application."""
    
    # Vector Database Configuration
    QDRANT_URL = "http://localhost:6333"
    
    # Embedding Service Configuration
    TORCHSERVE_URL = "http://localhost:8080/predictions/my_model"
    VECTOR_SIZE = 384  # Update based on your model's output size
    
    # Data Processing Configuration
    MIN_QUOTE_LENGTH = 50
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Ingestion Configuration
    BATCH_SIZE = 10
    SCORE_THRESHOLD = 0.98

    DATA_PATH = "data/Harry Potter - Book 1 - The Sorcerers Stone.pdf"
    OUTPUT_PATH = "data/processed"