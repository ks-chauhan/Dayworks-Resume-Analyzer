import os
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseSettings

os.environ["ANONYMIZED_TELEMETRY"] = "False"

class Settings(BaseSettings):
    # Embedding Model Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIMENSION: int = 768
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "resume_embeddings"
    
    # Similarity Thresholds
    MIN_SIMILARITY_SCORE: float = 0.3
    MAX_RESULTS_SINGLE_MODE: int = 1
    
    # File Processing
    MAX_FILE_SIZE_MB: int = 10
    SUPPORTED_FILE_TYPES: list = [".pdf", ".txt", ".docx"]
    
    # Batch Processing
    MAX_BATCH_SIZE: int = 100
    
    class Config:
        env_file = ".env"

settings = Settings()