"""
Agent Template - 入口文件

提供 FastAPI 服务端点，也可作为 CLI 入口。
启动: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from config.loader import get_settings
from llm import create_llm
from memory import LocalMemory
from agents import ChatAgent
from utils.logger import get_logger

logger = get_logger("main")

# ---------- 初始化 ----------
settings = get_settings(config_dir="config")

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="现代化 Agent 项目模板 API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 全局组件 ----------
llm = create_llm({
    "provider": settings.llm.provider,
    "model": settings.llm.model,
    "api_key": settings.llm.api_key,
    "base_url": settings.llm.base_url,
    "temperature": settings.llm.temperature,
    "max_tokens": settings.llm.max_tokens,
})

memory = LocalMemory(max_messages=settings.memory.max_history)

agent = ChatAgent(
    llm=llm,
    memory=memory,
    system_prompt="你是一个智能 AI 助手，用中文回答问题。",
)

logger.info(f"Agent Template 启动: provider={settings.llm.provider}, model={settings.llm.model}")


# ---------- 请求/响应模型 ----------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    reply: str
    session_id: str


# ---------- 路由 ----------
@app.get("/health")
async def health():
    return {"status": "ok", "version": settings.version}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """对话接口"""
    try:
        reply = await agent.chat(req.message, session_id=req.session_id)
        return ChatResponse(reply=reply, session_id=req.session_id)
    except Exception as e:
        logger.error(f"对话异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/switch_model")
async def switch_model(provider: str, model: str, api_key: str, base_url: Optional[str] = None):
    """运行时切换模型"""
    global llm
    try:
        new_llm = create_llm({
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
        })
        agent.llm = new_llm
        llm = new_llm
        logger.info(f"模型已切换: {provider}/{model}")
        return {"status": "ok", "provider": provider, "model": model}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
