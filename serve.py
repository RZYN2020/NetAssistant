import os
from typing import Dict
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from contextlib import asynccontextmanager
from net.api_models import (
    ChatCompletionResponse, ChatCompletionRequest, ChatMessage, 
    ChatCompletionChoice
)
from net.llm_services import LLMServiceManager
from net.rag_builder import build_rag_chain, RetrievalQA
from net.config import settings, LLMProviderEnum
from net.vector_store import get_vector_store
from net.document_loader import load_and_split_documents
from sse_starlette.sse import EventSourceResponse
import json

class AppState:
    llm_service_manager = None
    rag_chain_manager: Dict[str, RetrievalQA] = {}
    is_initialized: bool = False
    vector_store = None

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes all components on application startup."""
    print("Application startup: Initializing RAG system...")

    try:
        # 1. Initialize LLM Service (Ollama or OpenAI)
        app_state.llm_service_manager = LLMServiceManager(settings)
        embedding_model = app_state.llm_service_manager.get_embedding_model()

        # 2. Load and process documents
        if settings.RECRATE_KNOWLEDGE:
            split_docs = load_and_split_documents(
                settings.KNOWLEDGE_BASE_DIR,
                settings.CHUNK_SIZE,
                settings.CHUNK_OVERLAP
            )
        else:
            print("No knowledge base documents to load. Skipping document loading.")
            split_docs = []

        # 3. Create or load vector store
        app_state.vector_store = get_vector_store(
            split_docs,
            embedding_model,
            settings.VECTOR_DB_PATH,
            force_recreate=settings.RECRATE_KNOWLEDGE
        )

        if not app_state.vector_store:
            print("Error: Vector store could not be initialized.")
            app_state.is_initialized = False
            return

        # 4. Build RAG chain
        app_state.rag_chain_manager = {name: build_rag_chain(llm, app_state.vector_store, settings.SEARCH_K_DOCS)
                        for name, llm in app_state.llm_service_manager.get_chat_models().items()}
        app_state.is_initialized = True
        print(f"RAG system initialized successfully.")

    except Exception as e:
        app_state.is_initialized = False
        print(f"FATAL: RAG system initialization failed: {e}")

    yield

    print("Application shutdown: Cleaning up resources...")
    app_state.rag_chain_manager = None
    app_state.llm_service_manager = None
    app_state.is_initialized = False
    print("RAG system resources cleaned up.")

app = FastAPI(title="RAG TA API", version="1.0.0", lifespan=lifespan)

# Document Management APIs
@app.post("/v1/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a new document to the knowledge base."""
    if not app_state.is_initialized:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        file_path = os.path.join(settings.KNOWLEDGE_BASE_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process and add to vector store
        docs = load_and_split_documents([file_path], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        embedding_model = app_state.llm_service_manager.get_embedding_model()
        app_state.vector_store.add_documents(docs, embedding_model)
        
        return {"status": "success", "message": f"Document {file.filename} uploaded and processed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/documents/list")
async def list_documents():
    """List all documents in the knowledge base."""
    try:
        files = os.listdir(settings.KNOWLEDGE_BASE_DIR)
        return {
            "documents": [
                {
                    "name": f,
                    "size": os.path.getsize(os.path.join(settings.KNOWLEDGE_BASE_DIR, f))
                }
                for f in files if os.path.isfile(os.path.join(settings.KNOWLEDGE_BASE_DIR, f))
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/v1/documents/{document_name}")
async def delete_document(document_name: str):
    """Delete a document from the knowledge base."""
    try:
        file_path = os.path.join(settings.KNOWLEDGE_BASE_DIR, document_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            # Note: This is a simple implementation. In practice, you'd want to also
            # remove the document's embeddings from the vector store
            return {"status": "success", "message": f"Document {document_name} deleted"}
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/rag/recreate")
async def recreate_rag():
    """Recreate the RAG system."""
    if not app_state.is_initialized:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Recreate vector store and RAG chain
        split_docs = load_and_split_documents(
            settings.KNOWLEDGE_BASE_DIR,
            settings.CHUNK_SIZE,
            settings.CHUNK_OVERLAP
        )
        app_state.vector_store = get_vector_store(
            split_docs,
            app_state.llm_service_manager.get_embedding_model(),
            settings.VECTOR_DB_PATH,
            force_recreate=True
        )
        app_state.rag_chain_manager = build_rag_chain(
            app_state.llm_service_manager.get_chat_model(),
            app_state.vector_store,
            settings.SEARCH_K_DOCS
        )
        return {"status": "success", "message": "RAG system recreated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/v1/models")
async def list_models():
    """List available models, mimicking OpenAI's API format."""
    if not app_state.is_initialized:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    model_ids = app_state.llm_service_manager.get_chat_model_names()
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "user",
                "permission": [
                    {
                        "id": f"perm-{model_id}",
                        "created": 1677858242,
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": False,
                        "allow_search_indices": False,
                        "allow_view_logprobs": False,
                        "is_blocking": False
                    }
                ]
            }
            for model_id in model_ids
        ]
    }
    
@app.post("/v1/chat/completions")
async def chat_completions_endpoint(request: ChatCompletionRequest, http_req: Request):
    """Handles chat completion requests, compatible with OpenAI API format."""
    if not app_state.is_initialized or not app_state.rag_chain_manager:
        raise HTTPException(
            status_code=503,
            detail="RAG service is not available or failed to initialize. Please check server logs."
        )

    # Validate and select model
    requested_model = request.model
    if requested_model not in app_state.rag_chain_manager:
        available_models = list(app_state.rag_chain_manager.keys())
        if not available_models:
            raise HTTPException(status_code=503, detail="No models available")
        # Fall back to first available model
        requested_model = available_models[0]
        print(f"Warning: Requested model not found, falling back to {requested_model}")

    user_query = None
    if request.messages:
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_query = msg.content
                break
    
    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found in the request.")

    client_ip = http_req.client.host if http_req.client else "unknown"
    print(f"Received query from {client_ip}: '{user_query[:150]}...'")

    # Check if streaming is requested
    if request.stream:
        return EventSourceResponse(stream_response(user_query, requested_model))

    try:
        print(f"Invoking RAG chain with model {requested_model}...")
        rag_chain = app_state.rag_chain_manager[requested_model]
        rag_result = await rag_chain.ainvoke({"query": user_query})
        answer = rag_result.get("result", "Sorry, I could not find an answer to your question.")
        print(f"Generated answer: {answer[:150]}...")

    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during RAG processing.")

    response_message = ChatMessage(role="assistant", content=answer)
    choice = ChatCompletionChoice(index=0, message=response_message)

    return ChatCompletionResponse(
        model=requested_model,
        choices=[choice]
    )

async def stream_response(query: str, model_name: str):
    """Generator for streaming responses."""
    try:
        rag_chain = app_state.rag_chain_manager[model_name]
        async for chunk in rag_chain.astream({"query": query}):
            if isinstance(chunk, dict):
                content = chunk.get("result", "")
            else:
                content = str(chunk)
                
            if content:
                response_data = {
                    "choices": [{
                        "delta": {"content": content},
                        "index": 0,
                        "finish_reason": None
                    }],
                    "model": model_name
                }
                yield json.dumps(response_data)
                
        # Send final chunk with finish_reason
        yield json.dumps({
            "choices": [{
                "delta": {"content": ""},
                "index": 0,
                "finish_reason": "stop"
            }],
            "model": model_name
        })
                
    except Exception as e:
        print(f"Error during streaming: {e}")
        yield json.dumps({"error": str(e)})
@app.get("/health")
async def health_check_endpoint():
    """Provides a health check for the service."""
    if app_state.is_initialized:
        return { 
            "status": "ok",
            "message": "RAG service is running and initialized successfully."
        }
    else:
        return {
            "status": "error",
            "message": "RAG service is not initialized or failed during startup. Check server logs.",
        }

# Main entry point for uvicorn if running script directly
if __name__ == "__main__":
    print("Starting RAG TA API server...")
    if settings.OLLAMA_ENABLE:
        print(f"  Ollama URL: {settings.OLLAMA_BASE_URL}")
        print(f"  Ollama Embedding Model: {settings.OLLAMA_EMBEDDING_MODEL}")
        print(f"  Ollama Chat Models: {settings.OLLAMA_CHAT_MODELS}")
    if settings.OPENAI_ENABLE:
        print(f"  OpenAI Embedding Model: {settings.OPENAI_EMBEDDING_MODEL}")
        print(f"  OpenAI Chat Models: {settings.OPENAI_CHAT_MODELS}")
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
