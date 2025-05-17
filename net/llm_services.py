import os
from abc import ABC, abstractmethod

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from fastapi import HTTPException
from .config import AppSettings
from .config import LLMProviderEnum

class BaseLLMService(ABC):
    """Abstract base class for LLM and Embedding services."""
    def __init__(self, app_settings: AppSettings):
        self.settings = app_settings

    @abstractmethod
    def get_chat_model(self) -> BaseChatModel:
        pass

    @abstractmethod
    def get_embedding_model(self) -> Embeddings:
        pass

    def get_effective_chat_model_name(self) -> str:
        """Returns the name of the chat model that will be used."""
        # This method should be overridden by subclasses if they have specific logic
        # for determining the model name (e.g., from settings).
        # For now, a generic placeholder.
        if isinstance(self, OllamaService):
            return self.settings.OLLAMA_CHAT_MODEL
        elif isinstance(self, OpenAIService):
            return self.settings.OPENAI_CHAT_MODEL
        return "unknown_model"


class OllamaService(BaseLLMService):
    """Ollama LLM and Embedding Service Implementation."""
    def get_chat_model(self) -> BaseChatModel:
        print(f"Initializing Ollama chat model: {self.settings.OLLAMA_CHAT_MODEL} at {self.settings.OLLAMA_BASE_URL}")
        try:
            return ChatOllama(
                model=self.settings.OLLAMA_CHAT_MODEL,
                base_url=self.settings.OLLAMA_BASE_URL,
                temperature=0.3 # Default, can be overridden by request if implemented
            )
        except Exception as e:
            print(f"Failed to initialize Ollama chat model: {e}")
            raise HTTPException(status_code=500, detail=f"Ollama chat model initialization failed: {e}")

    def get_embedding_model(self) -> Embeddings:
        print(f"Initializing Ollama embedding model: {self.settings.OLLAMA_EMBEDDING_MODEL} at {self.settings.OLLAMA_BASE_URL}")
        try:
            return OllamaEmbeddings(
                model=self.settings.OLLAMA_EMBEDDING_MODEL,
                base_url=self.settings.OLLAMA_BASE_URL
            )
        except Exception as e:
            print(f"Failed to initialize Ollama embedding model: {e}")
            raise HTTPException(status_code=500, detail=f"Ollama embedding model initialization failed: {e}")
    
    def get_effective_chat_model_name(self) -> str:
        return self.settings.OLLAMA_CHAT_MODEL

class OpenAIService(BaseLLMService):
    """OpenAI LLM and Embedding Service Implementation."""
    def __init__(self, app_settings: AppSettings):
        super().__init__(app_settings)
        if not self.settings.OPENAI_API_KEY:
            # Try to load from standard OPENAI_API_KEY if APP_OPENAI_API_KEY is not set
            self.settings.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
            if not self.settings.OPENAI_API_KEY:
                 raise ValueError("OpenAI API key is not set. Please set APP_OPENAI_API_KEY or OPENAI_API_KEY environment variable.")
        # Langchain's OpenAI clients will pick up the API key from env or explicit pass
        os.environ["OPENAI_API_KEY"] = self.settings.OPENAI_API_KEY


    def get_chat_model(self) -> BaseChatModel:
        print(f"Initializing OpenAI chat model: {self.settings.OPENAI_CHAT_MODEL}")
        try:
            return ChatOpenAI(
                model_name=self.settings.OPENAI_CHAT_MODEL,
                openai_api_key=self.settings.OPENAI_API_KEY,
                temperature=0.3 # Default
            )
        except Exception as e:
            print(f"Failed to initialize OpenAI chat model: {e}")
            raise HTTPException(status_code=500, detail=f"OpenAI chat model initialization failed: {e}")

    def get_embedding_model(self) -> Embeddings:
        print(f"Initializing OpenAI embedding model: {self.settings.OPENAI_EMBEDDING_MODEL}")
        try:
            return OpenAIEmbeddings(
                model=self.settings.OPENAI_EMBEDDING_MODEL,
                openai_api_key=self.settings.OPENAI_API_KEY
            )
        except Exception as e:
            print(f"Failed to initialize OpenAI embedding model: {e}")
            raise HTTPException(status_code=500, detail=f"OpenAI embedding model initialization failed: {e}")

    def get_effective_chat_model_name(self) -> str:
        return self.settings.OPENAI_CHAT_MODEL

def get_llm_service(app_settings: AppSettings) -> BaseLLMService:
    """Factory function to get the appropriate LLM service."""
    if app_settings.LLM_PROVIDER == LLMProviderEnum.OLLAMA:
        print("Using Ollama LLM Service.")
        return OllamaService(app_settings)
    elif app_settings.LLM_PROVIDER == LLMProviderEnum.OPENAI:
        print("Using OpenAI LLM Service.")
        return OpenAIService(app_settings)
    else:
        raise ValueError(f"Unsupported LLM provider: {app_settings.LLM_PROVIDER}")
