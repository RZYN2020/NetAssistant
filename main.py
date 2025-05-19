import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from net.config import settings
from sse_starlette.sse import EventSourceResponse
import json
from net.rag_builder import Rag
import uuid
import time
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None # Client can specify, but server decides based on config
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    # Add other OpenAI compatible fields if needed, though they might not all be used by all models

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str # Actual model used by the backend
    choices: List[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)

    
    
rag = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = Rag()
    yield
    print("Application shutdown: Cleaning up resources...")

app = FastAPI(title="RAG TA API", version="1.0.0", lifespan=lifespan)

@app.get("/v1/models")
async def list_models():
    """List available models, mimicking OpenAI's API format."""
    model_ids = rag.get_models()
    model_ids = ["rag-" + model_id for model_id in model_ids]
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
    
    
# https://platform.openai.com/docs/api-reference/introduction
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_req: Request):
    """Handles chat completion requests, compatible with OpenAI API format."""
    
    request_model = request.model[4:] if request.model else None
    models = rag.get_models()
    if request_model not in models:
        raise HTTPException(status_code=400, detail=f"Model '{request.model}' not found.")

    user_query = next((msg.content for msg in reversed(request.messages) 
                      if msg.role == "user"), None)
    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found in the request.")

    client_ip = http_req.client.host if http_req.client else "unknown"
    print(f"Received query from {client_ip}: '{user_query[:150]}...'")

    if request.stream:
        return EventSourceResponse(stream_response(user_query, request_model))
    else:
        try:
            rag_result = await rag.invoke(request_model, user_query)
            answer = rag_result.get("result", "Sorry, I could not find an answer to your question.")
        except Exception as e:
            print(f"Error during RAG chain invocation: {e}")
            raise HTTPException(status_code=500, detail="Internal server error during RAG processing.")

        response_message = ChatMessage(role="assistant", content=answer)
        choice = ChatCompletionChoice(index=0, message=response_message)

        return ChatCompletionResponse(
            model=request.model,
            choices=[choice]
        )

async def stream_response(query: str, model_name: str):
    """Generator for streaming responses."""
    try:
        rag_chain = rag.get_rag_chain(model_name)
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
    return { 
            "status": "ok",
            "message": "RAG service is running and initialized successfully."
        }

# Main entry point for uvicorn if running script directly
if __name__ == "__main__":
    print("Starting RAG TA API server...")
    print(f"  Ollama URL: {settings.OLLAMA_BASE_URL}")
    print(f"  Ollama Embedding Model: {settings.OLLAMA_EMBEDDING_MODEL}")
    print(f"  Ollama Chat Models: {settings.OLLAMA_CHAT_MODELS}")

    print(f"Knowledge Base Directory: {settings.KNOWLEDGE_BASE_DIR}")
    print(f"Vector DB Path: {settings.VECTOR_DB_PATH}")
    print(f"API will run on: http://{settings.API_HOST}:{settings.API_PORT}")
    
    # Ensure knowledge base directory exists (also done in load_and_split_documents)
    if not os.path.exists(settings.KNOWLEDGE_BASE_DIR):
        os.makedirs(settings.KNOWLEDGE_BASE_DIR, exist_ok=True)
        print(f"Created knowledge base directory: {settings.KNOWLEDGE_BASE_DIR}")

    uvicorn.run("__main__:app", host=settings.API_HOST, port=settings.API_PORT, reload=False)
    # For development, use: uvicorn your_filename:app --reload
