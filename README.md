# NetAssistant
1. `pip install "fastapi[all]" uvicorn langchain langchain_community langchain_openai faiss-cpu tiktoken pypdf sentence-transformers ollama pydantic-settings`
2. Set up environment variables (see config.py section)
3. Create your knowledge base directory (e.g., knowledge_base_data/) and add documents.
4. Run with uvicorn: uvicorn main:app --host 0.0.0.0 --port 8000 --reload