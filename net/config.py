from enum import Enum
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMProviderEnum(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

class AppSettings(BaseSettings):
    """
    Application Configuration.
    Values are loaded from environment variables.
    Prefix for env vars: APP_ (e.g., APP_LLM_PROVIDER)
    """
    model_config = SettingsConfigDict(env_prefix='APP_', extra='ignore', case_sensitive=False)

    LLM_PROVIDER: LLMProviderEnum = LLMProviderEnum.OLLAMA

    # Ollama settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_CHAT_MODEL: str = "deepseek-r1:1.5b"

    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = None # Loaded from env: APP_OPENAI_API_KEY or OPENAI_API_KEY
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str = "gpt-3.5-turbo"

    # Knowledge base and vector store paths
    KNOWLEDGE_BASE_DIR: str = "knowledge_base_data"
    ADD_KNOWLEDGE: bool = False
    VECTOR_DB_PATH: str = "vector_db_faiss_modular"
    
    # RAG settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    SEARCH_K_DOCS: int = 5 # Number of documents to retrieve
    PROMPT_PATH: str = "prompt_template.txt"

    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

# Initialize settings
settings = AppSettings()