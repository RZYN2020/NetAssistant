# NetAssistant
1. Set up environment variables (see config.py section)
2. Create your knowledge base directory (e.g., knowledge_base_data/) and add documents.
3. Run with uvicorn: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# todo
1. 流式响应 （以支持各种客户端）
2. better rag（处理retirval效果不好的问题）
   1. 更好的数据源（好的教材以及开源文档）
   2. 更好的算法（graphrag? better embeding?...）
4. 性能优化（更好的推理框架etc）
