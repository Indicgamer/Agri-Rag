"""
Settings module for Agri-RAG project.
Centralizes all configuration management.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Neo4jSettings(BaseSettings):
    uri: str = Field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    username: str = Field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"))
    database: str = Field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "agri_rag_kg"))

    class Config:
        env_prefix = "AGRI_RAG_NEO4J_"
        extra = "ignore"

class ChromaDBSettings(BaseSettings):
    persistence_dir: Path = Field(default_factory=lambda: Path(os.getenv("CHROMADB_PERSISTENCE_DIR", "data/chroma_db")))
    collection_name: str = Field(default_factory=lambda: os.getenv("CHROMADB_COLLECTION_NAME", "agriculture_corpus"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("CHROMADB_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

    class Config:
        env_prefix = "AGRI_RAG_CHROMA_"
        extra = "ignore"

class TripleExtractorSettings(BaseSettings):
    provider: str = Field(default_factory=lambda: os.getenv("TRIPLET_PROVIDER", os.getenv("LLM_PROVIDER", "ollama")))
    model_name: str = Field(default_factory=lambda: os.getenv("TRIPLET_OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2:1b")))
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    groq_model: str = Field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    openai_base_url: str = Field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    groq_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    allow_rule_fallback: bool = Field(default_factory=lambda: os.getenv("ALLOW_RULE_FALLBACK", "false").lower() == "true")
    temperature: float = Field(default_factory=lambda: float(os.getenv("TRIPLET_TEMPERATURE", "0.3")))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("TRIPLET_MAX_TOKENS", "500")))
    ollama_timeout_seconds: int = Field(default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")))
    groq_max_retries: int = Field(default_factory=lambda: int(os.getenv("GROQ_MAX_RETRIES", "2")))
    groq_max_rate_limit_wait_seconds: int = Field(default_factory=lambda: int(os.getenv("GROQ_MAX_RATE_LIMIT_WAIT_SECONDS", "30")))

    class Config:
        env_prefix = "AGRI_RAG_TRIPLET_"
        extra = "ignore"

class NLISettings(BaseSettings):
    provider: str = Field(default_factory=lambda: os.getenv("NLI_PROVIDER", "transformers"))
    model_name: str = Field(default_factory=lambda: os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-base"))
    device: str = Field(default_factory=lambda: os.getenv("NLI_DEVICE", "cpu"))
    hf_token: Optional[str] = Field(default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN"))
    entailment_threshold: float = Field(default=0.7)
    neutral_threshold: float = Field(default=0.4)

    class Config:
        env_prefix = "AGRI_RAG_NLI_"
        extra = "ignore"

class LLMSettings(BaseSettings):
    provider: str = Field(default_factory=lambda: os.getenv("RESPONSE_PROVIDER", os.getenv("LLM_PROVIDER", "ollama")))
    model_name: str = Field(default_factory=lambda: os.getenv("RESPONSE_OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2:1b")))
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    groq_model: str = Field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    openai_base_url: str = Field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    groq_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    allow_response_fallback: bool = Field(default_factory=lambda: os.getenv("ALLOW_RESPONSE_FALLBACK", "false").lower() == "true")
    temperature: float = Field(default_factory=lambda: float(os.getenv("RESPONSE_TEMPERATURE", "0.2")))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("RESPONSE_MAX_TOKENS", "350")))
    context_window: int = Field(default_factory=lambda: int(os.getenv("RESPONSE_CONTEXT_WINDOW", "2500")))

    class Config:
        env_prefix = "AGRI_RAG_LLM_"
        extra = "ignore"

class RetrievalSettings(BaseSettings):
    vector_weight: float = Field(default_factory=lambda: float(os.getenv("RETRIEVAL_VECTOR_WEIGHT", "0.35")))
    graph_weight: float = Field(default_factory=lambda: float(os.getenv("RETRIEVAL_GRAPH_WEIGHT", "0.65")))
    top_k_vector: int = Field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K_VECTOR", "4")))
    top_k_graph: int = Field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K_GRAPH", "8")))
    max_hops: int = Field(default_factory=lambda: int(os.getenv("RETRIEVAL_MAX_HOPS", "2")))

    class Config:
        env_prefix = "AGRI_RAG_RETRIEVAL_"
        extra = "ignore"

class Settings(BaseSettings):
    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = Field(default=Path("data"))
    log_dir: Path = Field(default=Path("logs"))
    
    # Subsettings
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    chromadb: ChromaDBSettings = Field(default_factory=ChromaDBSettings)
    triplet_extractor: TripleExtractorSettings = Field(default_factory=TripleExtractorSettings)
    nli: NLISettings = Field(default_factory=NLISettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    
    # General settings
    crops: list = Field(default=["Rice", "Wheat", "Cotton", "Maize", "Sugarcane"])
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    
    class Config:
        env_file = ".env"
        env_prefix = "AGRI_RAG_"
        case_sensitive = False
        extra = "ignore"

# Global settings instance
settings = Settings()
