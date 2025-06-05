#!/usr/bin/env python3
import os
import json
import uuid
import time
import asyncio
import subprocess
import platform
from typing import List, Optional, AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse


# 配置项
class Settings:
    API_HOST = "0.0.0.0"
    API_PORT = 8002
    GRAPHRAG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphrag")
    GRAPHRAG_METHOD = "local"
    GRAPHRAG_MODELS = ["graphrag-local"]  # 支持的模型列表

settings = Settings()


# OpenAI 兼容的数据模型
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionChoiceStream(BaseModel):
    index: int
    delta: dict
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoiceStream]


class GraphRAGExecutor:
    """GraphRAG命令执行器"""
    
    def __init__(self):
        self.is_windows = platform.system().lower() == "windows"

    def _build_command(self, query: str, streaming: bool = True) -> List[str]:
        """构建GraphRAG命令"""
        cmd = [
            "graphrag", "query",
            "--root", settings.GRAPHRAG_ROOT,
            "--method", settings.GRAPHRAG_METHOD,
            "--query", query
        ]

        if streaming:
            cmd.append("--streaming")
        else:
            cmd.append("--no-streaming")
            
        return cmd
    
    async def execute_streaming(self, query: str) -> AsyncGenerator[str, None]:
        """流式执行GraphRAG命令"""
        cmd = self._build_command(query, streaming=True)
        
        try:
            # 创建子进程
            if self.is_windows:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                # 标记是否已经开始输出有效内容
                content_started = False
                buffer = ""

                # 逐行读取输出
                while True:
                    line_bytes = await process.stdout.readline()
                    if not line_bytes:
                        break

                    # 解码原始行以进行换行符检查
                    original_decoded_line = line_bytes.decode('utf-8', errors='ignore')
                    current_line_content = original_decoded_line

                    # 检查是否遇到SUCCESS行
                    if not content_started:
                        success_marker = "SUCCESS:"
                        idx = current_line_content.find(success_marker)
                        if idx != -1:
                            content_started = True
                            # 提取 SUCCESS: 标记之后的内容
                            current_line_content = current_line_content[idx + len(success_marker):]
                            # 如果 SUCCESS: 标记之后行为空或仅包含空白，则继续读取下一行
                            if not current_line_content.strip():
                                continue
                        else:
                            # 非 SUCCESS: 标记行，且内容尚未开始，跳过此行
                            continue
                    
                    # 如果 current_line_content 不为空 (例如 SUCCESS: 后有数据，或这是 SUCCESS: 后的数据行)
                    if current_line_content:
                        buffer += current_line_content
                    
                    # 当原始读取的行以换行符结束时，处理缓冲区
                    if original_decoded_line.endswith('\\n'):
                        if buffer.strip():
                            yield buffer.strip()
                            buffer = ""

                # 输出剩余内容
                if buffer.strip():
                    yield buffer.strip()

                # 等待进程结束
                await process.wait()

                if process.returncode != 0:
                    stderr = await process.stderr.read()
                    error_msg = stderr.decode('utf-8', errors='ignore')
                    raise RuntimeError(f"GraphRAG command failed: {error_msg}")

        except Exception as e:
            raise RuntimeError(f"Failed to execute GraphRAG command: {str(e)}")
    
    async def execute_non_streaming(self, query: str) -> str:
        """非流式执行GraphRAG命令"""
        cmd = self._build_command(query, streaming=False)

        try:
            if self.is_windows:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await process.communicate()

                if process.returncode != 0:
                    error_msg = stderr.decode('utf-8', errors='ignore')
                    raise RuntimeError(f"GraphRAG command failed: {error_msg}")

                output = stdout.decode('utf-8', errors='ignore')

                # 提取有效内容
                lines = output.split('\n')
                content_started = False
                result_lines = []

                for line in lines:
                    # 检查是否遇到SUCCESS行
                    if not content_started:
                        if line.startswith("SUCCESS:"):
                            content_started = True
                        continue

                    # 收集SUCCESS行之后的所有内容
                    if content_started:
                        result_lines.append(line)

                return '\n'.join(result_lines).strip()

        except Exception as e:
            raise RuntimeError(f"Failed to execute GraphRAG command: {str(e)}")


# 全局执行器实例
graphrag_executor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global graphrag_executor
    print("Initializing GraphRAG CLI API...")
    graphrag_executor = GraphRAGExecutor()
    
    # 检查GraphRAG是否可用
    try:
        result = subprocess.run(
            ["graphrag", "--help"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode != 0:
            print("Warning: GraphRAG command not found or not working properly")
    except Exception as e:
        print(f"Warning: Could not verify GraphRAG installation: {e}")
    
    yield
    print("Shutting down GraphRAG CLI API...")


app = FastAPI(
    title="GraphRAG CLI API",
    version="1.0.0",
    description="OpenAI-compatible API wrapper for GraphRAG CLI",
    lifespan=lifespan
)


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
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
                        "created": int(time.time()),
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": False,
                        "allow_search_indices": False,
                        "allow_view_logprobs": False,
                        "is_blocking": False
                    }
                ]
            }
            for model_id in settings.GRAPHRAG_MODELS
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_req: Request):
    """处理聊天完成请求"""
    
    # 验证模型
    if request.model and request.model not in settings.GRAPHRAG_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{request.model}' not found. Available models: {settings.GRAPHRAG_MODELS}"
        )
    
    # 获取用户查询
    user_query = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_query = msg.content
            break
    
    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found in the request.")
    
    # 获取客户端信息
    client_ip = http_req.client.host if http_req.client else "unknown"
    print(f"Received query from {client_ip}: '{user_query[:100]}...'")
    
    model_name = request.model or settings.GRAPHRAG_MODELS[0]
    
    if request.stream:
        return EventSourceResponse(stream_response(user_query, model_name))
    else:
        try:
            result = await graphrag_executor.execute_non_streaming(user_query)
            
            if not result:
                result = "Sorry, I could not find an answer to your question."
            
            response_message = ChatMessage(role="assistant", content=result)
            choice = ChatCompletionChoice(index=0, message=response_message)
            
            return ChatCompletionResponse(
                model=model_name,
                choices=[choice]
            )
            
        except Exception as e:
            print(f"Error during GraphRAG execution: {e}")
            raise HTTPException(status_code=500, detail="Internal server error during GraphRAG processing.")


async def stream_response(query: str, model_name: str):
    """流式响应生成器"""
    try:
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        
        async for chunk in graphrag_executor.execute_streaming(query):
            if chunk:
                response_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(response_data)}\n\n"
        
        # 发送结束标记
        final_response = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_response)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        print(f"Error during streaming: {e}")
        error_response = {
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": "graphrag_error"
            }
        }
        yield f"data: {json.dumps(error_response)}\n\n"


@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        # 简单测试GraphRAG是否可用
        result = subprocess.run(
            ["graphrag", "--help"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        
        if result.returncode == 0:
            status = "healthy"
            message = "GraphRAG CLI API is running and GraphRAG command is available"
        else:
            status = "degraded"
            message = "GraphRAG CLI API is running but GraphRAG command may not be working properly"
            
    except Exception as e:
        status = "degraded"
        message = f"GraphRAG CLI API is running but GraphRAG command check failed: {str(e)}"
    
    return {
        "status": status,
        "message": message,
        "timestamp": int(time.time()),
        "version": "1.0.0"
    }


def main():
    """主入口函数"""
    print("Starting GraphRAG CLI API server...")
    print(f"  GraphRAG Root: {settings.GRAPHRAG_ROOT}")
    print(f"  GraphRAG Method: {settings.GRAPHRAG_METHOD}")
    print(f"  Available Models: {settings.GRAPHRAG_MODELS}")
    print(f"  API will run on: http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"  Platform: {platform.system()}")
    
    # 检查GraphRAG根目录
    if not os.path.exists(settings.GRAPHRAG_ROOT):
        print(f"Warning: GraphRAG root directory '{settings.GRAPHRAG_ROOT}' does not exist")
    
    uvicorn.run(
        "graphrag_cli_api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
