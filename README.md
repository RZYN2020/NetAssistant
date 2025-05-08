# NetAssistant
1. Set up environment variables (see config.py section)
2. Create your knowledge base directory (e.g., knowledge_base_data/) and add documents.
3. Run with uvicorn: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# todo
1. 流式响应 （以支持各种客户端）
2. better rag（处理retirval效果不好的问题）