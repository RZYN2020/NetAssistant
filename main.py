import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request

app = FastAPI(title="Modular RAG TA API", version="1.0.0")

# Global state for initialized components
# In a class-based structure, these would be attributes of an AppManager or similar
class AppState:
    llm_service_instance: Optional[BaseLLMService] = None
    rag_chain_instance: Optional[RetrievalQA] = None
    is_initialized: bool = False
    effective_model_name: str = "not_initialized"

app_state = AppState()


@app.on_event("startup")
async def startup_event():
    """Initializes all components on application startup."""
    print("Application startup: Initializing RAG system...")
    print(f"Using LLM Provider: {settings.LLM_PROVIDER.value}")

    try:
        # 1. Initialize LLM Service (Ollama or OpenAI)
        app_state.llm_service_instance = get_llm_service(settings)
        chat_model = app_state.llm_service_instance.get_chat_model()
        embedding_model = app_state.llm_service_instance.get_embedding_model()
        app_state.effective_model_name = app_state.llm_service_instance.get_effective_chat_model_name()

        # 2. Load and process documents
        split_docs = load_and_split_documents(
            settings.KNOWLEDGE_BASE_DIR,
            settings.CHUNK_SIZE,
            settings.CHUNK_OVERLAP
        )

        # 3. Create or load vector store
        # Consider adding a config option for force_recreate_db for development
        vector_store = get_vector_store(
            split_docs,
            embedding_model,
            settings.VECTOR_DB_PATH,
            force_recreate=False # Set to True to rebuild DB on startup
        )

        if not vector_store:
            print("Error: Vector store could not be initialized. RAG chain will not be built.")
            app_state.is_initialized = False
            # Optionally raise an exception here to prevent app from starting if DB is critical
            # raise RuntimeError("Failed to initialize vector store.")
            return


        # 4. Build RAG chain
        app_state.rag_chain_instance = build_rag_chain(
            chat_model,
            vector_store,
            settings.SEARCH_K_DOCS
        )
        
        app_state.is_initialized = True
        print(f"RAG system initialized successfully. Effective chat model: {app_state.effective_model_name}")

    except Exception as e:
        app_state.is_initialized = False
        print(f"FATAL: RAG system initialization failed: {e}")
        # Depending on severity, you might want the app to not start
        # For now, it will start but /chat/completions will fail
        # raise # Re-raise to stop FastAPI startup on critical failure


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions_endpoint(request: ChatCompletionRequest, http_req: Request):
    """Handles chat completion requests, compatible with OpenAI API format."""
    if not app_state.is_initialized or not app_state.rag_chain_instance:
        raise HTTPException(
            status_code=503,
            detail="RAG service is not available or failed to initialize. Please check server logs."
        )

    user_query = None
    if request.messages:
        # Get the last user message as the query
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_query = msg.content
                break
    
    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found in the request.")

    client_ip = http_req.client.host if http_req.client else "unknown"
    print(f"Received query from {client_ip}: '{user_query[:150]}...'")

    try:
        # Asynchronously invoke the RAG chain
        # Note: RetrievalQA.ainvoke might need specific setup or Langchain version.
        # If ainvoke is problematic, use a thread pool executor for sync invoke.
        # For now, assuming ainvoke works as expected for async FastAPI.
        print("Invoking RAG chain...")
        rag_result = await app_state.rag_chain_instance.ainvoke({"query": user_query})
        
        answer = rag_result.get("result", "Sorry, I could not find an answer to your question.")
        # source_documents = rag_result.get("source_documents", []) # For logging or advanced responses
        
        print(f"Generated answer: {answer[:150]}...")

    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during RAG processing: {str(e)}")

    response_message = ChatMessage(role="assistant", content=answer)
    choice = ChatCompletionChoice(index=0, message=response_message)
    
    # Determine the model name to return in the response
    # If client specified a model and it matches the provider's configured model, use it.
    # Otherwise, use the server's effective model name.
    # This is mostly for API compatibility display.
    response_model_name = app_state.effective_model_name
    if request.model:
        # Simple check: if user requests 'ollama' and provider is ollama, or 'openai' and provider is openai
        # This could be made more sophisticated if you allow multiple models per provider.
        if (settings.LLM_PROVIDER == LLMProviderEnum.OLLAMA and "ollama" in request.model.lower()) or \
           (settings.LLM_PROVIDER == LLMProviderEnum.OPENAI and ("gpt" in request.model.lower() or "openai" in request.model.lower())):
            # Potentially use request.model if it's a valid model for the current provider,
            # but for now, stick to the configured one for simplicity.
            pass # print(f"Client requested model {request.model}, server using {response_model_name}")
        else:
            print(f"Warning: Client requested model '{request.model}', but server is configured for '{settings.LLM_PROVIDER.value}' with model '{response_model_name}'.")


    return ChatCompletionResponse(
        model=response_model_name,
        choices=[choice]
        # Usage field is default, actual token counts are hard with local models / not always provided
    )

@app.get("/health")
async def health_check_endpoint():
    """Provides a health check for the service."""
    if app_state.is_initialized:
        return {
            "status": "ok",
            "message": "RAG service is initialized and running.",
            "llm_provider": settings.LLM_PROVIDER.value,
            "chat_model": app_state.effective_model_name
        }
    else:
        return {
            "status": "error",
            "message": "RAG service is not initialized or failed during startup. Check server logs.",
            "llm_provider": settings.LLM_PROVIDER.value
        }

# Main entry point for uvicorn if running script directly
if __name__ == "__main__":
    print("Starting Modular RAG TA API server...")
    print(f"Configuration - LLM Provider: {settings.LLM_PROVIDER.value}")
    if settings.LLM_PROVIDER == LLMProviderEnum.OLLAMA:
        print(f"  Ollama URL: {settings.OLLAMA_BASE_URL}")
        print(f"  Ollama Embedding Model: {settings.OLLAMA_EMBEDDING_MODEL}")
        print(f"  Ollama Chat Model: {settings.OLLAMA_CHAT_MODEL}")
    elif settings.LLM_PROVIDER == LLMProviderEnum.OPENAI:
        print(f"  OpenAI Embedding Model: {settings.OPENAI_EMBEDDING_MODEL}")
        print(f"  OpenAI Chat Model: {settings.OPENAI_CHAT_MODEL}")
        if not settings.OPENAI_API_KEY and not os.environ.get("OPENAI_API_KEY"):
            print("  WARNING: OpenAI API Key not found in APP_OPENAI_API_KEY or OPENAI_API_KEY env vars!")


    print(f"Knowledge Base Directory: {settings.KNOWLEDGE_BASE_DIR}")
    print(f"Vector DB Path: {settings.VECTOR_DB_PATH}")
    print(f"API will run on: http://{settings.API_HOST}:{settings.API_PORT}")
    
    # Ensure knowledge base directory exists (also done in load_and_split_documents)
    if not os.path.exists(settings.KNOWLEDGE_BASE_DIR):
        os.makedirs(settings.KNOWLEDGE_BASE_DIR, exist_ok=True)
        print(f"Created knowledge base directory: {settings.KNOWLEDGE_BASE_DIR}")

    uvicorn.run("__main__:app", host=settings.API_HOST, port=settings.API_PORT, reload=False)
    # For development, use: uvicorn your_filename:app --reload
