"""
任务型 Agent

支持 Function Calling 的 Agent，实现 ReAct 循环：
1. 用户输入 -> LLM 思考
2. LLM 决定调用工具 -> 执行工具 -> 将结果返回给 LLM
3. 重复步骤 2 直到 LLM 给出最终回答
"""
import json
import time
import logging
from typing import Optional

from llm.base import LLMBase, ChatMessage
from memory.base import MemoryBase
from rag.retriever import Retriever
from tools.registry import ToolRegistry
from agents.base import AgentBase, AgentResponse

logger = logging.getLogger(__name__)


class TaskAgent(AgentBase):
    """
    任务型 Agent（支持 Tool Calling）

    实现 ReAct 循环：思考 -> 调用工具 -> 观察 -> 再思考 -> 最终回答

    使用示例：
        registry = ToolRegistry()
        registry.add(WebSearchTool())
        registry.add(CodeExecutorTool())

        agent = TaskAgent(
            llm=llm,
            memory=LocalMemory(),
            tool_registry=registry,
            system_prompt="你是一个能使用工具的智能助手。",
            max_tool_rounds=5,
        )
        response = await agent.chat("帮我搜索今天的新闻")
    """

    def __init__(
        self,
        llm: LLMBase,
        memory: Optional[MemoryBase] = None,
        retriever: Optional[Retriever] = None,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: str = "你是一个智能助手，可以使用工具来完成任务。",
        memory_window_size: int = 10,
        max_tool_rounds: int = 5,
    ):
        super().__init__(
            llm=llm,
            memory=memory,
            retriever=retriever,
            tool_registry=tool_registry,
            system_prompt=system_prompt,
            memory_window_size=memory_window_size,
        )
        self.max_tool_rounds = max_tool_rounds

    async def chat(
        self,
        user_input: str,
        session_id: str = "default",
        use_rag: bool = True,
    ) -> AgentResponse:
        """
        带工具调用的对话入口

        实现 ReAct 循环：
        LLM 回复 -> 检查是否有 tool_calls -> 执行工具 -> 将结果传回 LLM -> 循环
        """
        start_time = time.time()

        # 1. 构建消息列表
        messages = [ChatMessage(role="system", content=self.system_prompt)]

        # 2. RAG 检索
        sources = []
        if use_rag and self.retriever:
            docs = await self.retriever.retrieve(user_input)
            if docs:
                context = Retriever.format_context(docs)
                messages.append(ChatMessage(
                    role="system",
                    content=f"【参考资料】\n{context}\n\n请结合参考资料回答。",
                ))
                sources = [{"content": d["content"][:200], "score": d.get("score", 0)} for d in docs]

        # 3. 加载对话历史
        if self.memory:
            history = await self.memory.get_history(session_id, limit=self.memory_window_size)
            for msg in history:
                messages.append(ChatMessage(role=msg.role, content=msg.content))

        # 4. 添加用户输入
        messages.append(ChatMessage(role="user", content=user_input))

        # 5. 获取工具 Schema
        tools = self.tool_registry.get_function_schemas() if self.tool_registry else None

        # 6. ReAct 循环
        all_tool_calls = []
        for round_idx in range(self.max_tool_rounds):
            llm_response = await self.llm.generate(messages=messages, tools=tools)

            if not llm_response.is_tool_call:
                # LLM 给出了最终回答
                break

            # 处理工具调用
            logger.info(f"[ReAct 第{round_idx + 1}轮] 工具调用: {len(llm_response.tool_calls)} 个")

            # 将 assistant 的 tool_calls 消息加入历史
            messages.append(ChatMessage(
                role="assistant",
                content=llm_response.content or "",
                tool_calls=llm_response.tool_calls,
            ))

            # 逐个执行工具
            for tc in llm_response.tool_calls:
                func_name = tc.get("function", {}).get("name", "")
                logger.info(f"  执行工具: {func_name}")

                result = await self.tool_registry.call_from_llm_response(tc)
                all_tool_calls.append({
                    "tool": func_name,
                    "result": result.to_str()[:500],
                    "success": result.success,
                })

                # 将工具结果传回 LLM
                messages.append(ChatMessage(
                    role="tool",
                    content=result.to_str(),
                    tool_call_id=tc.get("id", ""),
                    name=func_name,
                ))
        else:
            # 达到最大轮次仍未结束
            logger.warning(f"达到最大工具调用轮次 ({self.max_tool_rounds})")

        # 7. 保存对话历史
        if self.memory:
            await self.memory.add(session_id, "user", user_input)
            await self.memory.add(session_id, "assistant", llm_response.content)

        elapsed = time.time() - start_time

        return AgentResponse(
            content=llm_response.content,
            tool_calls=all_tool_calls,
            sources=sources,
            tokens_used=llm_response.tokens_used,
            model=llm_response.model,
            session_id=session_id,
            elapsed_seconds=round(elapsed, 2),
        )
