"""
Agent 基类

组合 LLM + Memory + RAG + Tools，提供统一的 Agent 生命周期管理。
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from llm.base import LLMBase, ChatMessage, LLMResponse
from memory.base import MemoryBase
from rag.retriever import Retriever
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Agent 响应结构"""
    content: str = ""
    tool_calls: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    tokens_used: int = 0
    model: str = ""
    session_id: str = ""
    elapsed_seconds: float = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "sources": self.sources,
            "tokens_used": self.tokens_used,
            "model": self.model,
            "session_id": self.session_id,
            "elapsed_seconds": self.elapsed_seconds,
        }


class AgentBase:
    """
    Agent 基类

    组合以下组件：
    - LLM: 大语言模型调用
    - Memory: 对话历史管理
    - Retriever: RAG 检索增强（可选）
    - ToolRegistry: 工具注册中心（可选）

    使用示例：
        llm = create_llm(config)
        memory = LocalMemory()
        agent = AgentBase(
            llm=llm,
            memory=memory,
            system_prompt="你是一个助手",
        )
        response = await agent.chat("你好", session_id="s1")
    """

    def __init__(
        self,
        llm: LLMBase,
        memory: Optional[MemoryBase] = None,
        retriever: Optional[Retriever] = None,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: str = "你是一个智能助手。",
        memory_window_size: int = 10,
    ):
        self.llm = llm
        self.memory = memory
        self.retriever = retriever
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.memory_window_size = memory_window_size

    async def chat(
        self,
        user_input: str,
        session_id: str = "default",
        use_rag: bool = True,
    ) -> AgentResponse:
        """
        Agent 主对话入口

        流程：
        1. 构建系统提示词
        2. RAG 检索增强（可选）
        3. 加载对话历史
        4. 调用 LLM
        5. 保存对话历史
        """
        start_time = time.time()

        # 1. 系统提示词
        messages = [ChatMessage(role="system", content=self.system_prompt)]

        # 2. RAG 检索
        sources = []
        if use_rag and self.retriever:
            docs = await self.retriever.retrieve(user_input)
            if docs:
                context = Retriever.format_context(docs)
                messages.append(ChatMessage(
                    role="system",
                    content=f"【参考资料】\n{context}\n\n请结合上述参考资料回答用户问题。",
                ))
                sources = [
                    {"content": d["content"][:200], "score": d.get("score", 0)}
                    for d in docs
                ]

        # 3. 加载对话历史
        if self.memory:
            history = await self.memory.get_history(session_id, limit=self.memory_window_size)
            for msg in history:
                messages.append(ChatMessage(role=msg.role, content=msg.content))

        # 4. 添加用户输入
        messages.append(ChatMessage(role="user", content=user_input))

        # 5. 调用 LLM
        llm_response = await self.llm.generate(messages=messages)

        # 6. 保存对话历史
        if self.memory:
            await self.memory.add(session_id, "user", user_input)
            await self.memory.add(session_id, "assistant", llm_response.content)

        elapsed = time.time() - start_time

        return AgentResponse(
            content=llm_response.content,
            sources=sources,
            tokens_used=llm_response.tokens_used,
            model=llm_response.model,
            session_id=session_id,
            elapsed_seconds=round(elapsed, 2),
        )
