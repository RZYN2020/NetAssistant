from typing import Dict, List
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from fastapi import HTTPException
from .config import AppSettings


class LLMServiceManager:
    def __init__(self, app_settings: AppSettings):
        self.app_settings = app_settings
        self.llm_chat_models = get_chat_models_from_settings(app_settings)
    
    def get_chat_model(self, model_name: str) -> BaseChatModel:
        """Get the chat model instance for the specified model name."""
        if model_name not in self.llm_chat_models:
            raise HTTPException(status_code=400, detail=f"Model {model_name} is not available.")
        return self.llm_chat_models[model_name]
    
    def get_chat_models(self) -> Dict[str, BaseChatModel]:
        """Get all available chat models."""
        return self.llm_chat_models
    
    def get_chat_model_names(self) -> List[str]:
        """Get the list of available chat models."""
        return list(self.llm_chat_models.keys())
    
    def get_embedding_model(self) -> Embeddings:
        """Get the embedding model instance."""
        if self.app_settings.OPENAI_ENABLE:
            raise NotImplementedError("OpenAI embedding service is not implemented yet.")
        elif self.app_settings.OLLAMA_ENABLE:
            return OllamaEmbeddings(
                model=self.app_settings.OLLAMA_EMBEDDING_MODEL,
                base_url=self.app_settings.OLLAMA_BASE_URL
            )
        else:
            raise HTTPException(status_code=500, detail="No LLM service is enabled.")

    
    
def get_chat_models_from_settings(app_settings: AppSettings) -> Dict[str, BaseChatModel]:
    """Factory function to get the appropriate LLM service."""
    def get_chat_model(name) -> BaseChatModel:
        try:
            return ChatOllama(
                model=name,
                base_url=app_settings.OLLAMA_BASE_URL,
                temperature=0.3 # Default, can be overridden by request if implemented
            )
        except Exception as e:
            print(f"Failed to initialize Ollama chat model: {e}")
            raise HTTPException(status_code=500, detail=f"Ollama chat model initialization failed: {e}")
    if app_settings.OPENAI_ENABLE:
        raise NotImplementedError("OpenAI service is not implemented yet.")
    if app_settings.OLLAMA_ENABLE:
        return {model: get_chat_model(model) for model in app_settings.OLLAMA_CHAT_MODELS}
    return {}
