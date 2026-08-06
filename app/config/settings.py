from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    chroma_db_path: str = "data/chroma_db"
    bm25_index_path: str = "data/bm25_index.pkl"
    log_level: str = "INFO"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval parameters
    top_k: int = 5
    vector_k: int = 10
    bm25_k: int = 10
    rerank_top_k: int = 3
    rrf_k: int = 60
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    # OpenAI & LLM configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.0
    openai_max_tokens: int = 1000

    # Validation thresholds
    min_confidence_threshold: float = 0.35
    min_chunks_required: int = 1

    # Uploads
    max_upload_size: int = 10485760  # 10MB default
    uploaded_contracts_dir: str = "data/contracts"

    # Evaluation thresholds
    faithfulness_threshold: float = 0.8
    answer_relevancy_threshold: float = 0.8
    latency_threshold: float = 2.0  # seconds

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
