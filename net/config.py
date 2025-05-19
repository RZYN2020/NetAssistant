from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    """
    Application Configuration.
    Values are loaded from environment variables.
    Prefix for env vars: APP_ (e.g., APP_LLM_PROVIDER)
    """
    model_config = SettingsConfigDict(env_prefix='APP_', extra='ignore', case_sensitive=False)
    
    DEBUG: bool = True

    # Ollama settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_HELP_MODEL: str = "deepseek-r1:1.5b"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_CHAT_MODELS: List[str] = ["deepseek-r1:1.5b"]
    OLLAMA_ENABLE: bool = True

    # Knowledge base and vector store paths
    KNOWLEDGE_BASE_DIR: str = "knowledge_base_data"
    RECRATE_KNOWLEDGE: bool = False
    VECTOR_DB_PATH: str = "vector_db_faiss_modular"
    
    # RAG settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    SEARCH_K_DOCS: int = 7 # Number of documents to retrieve
    TOP_K: int = 3 # Number of top results to return
    PROMPT_PATH: str = "prompt_template.txt"

    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

# Initialize settings
settings = AppSettings()