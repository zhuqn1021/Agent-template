"""
对话型 Agent

纯对话场景，支持多轮对话 + RAG 检索增强。
不涉及工具调用，适用于问答、咨询、客服等场景。
"""
from agents.base import AgentBase


class ChatAgent(AgentBase):
    """
    对话型 Agent

    继承 AgentBase，无额外逻辑，语义清晰地表示「纯对话场景」。

    使用示例：
        agent = ChatAgent(
            llm=llm,
            memory=LocalMemory(),
            system_prompt="你是一个友善的客服助手。",
        )
        response = await agent.chat("你好")
    """
    pass
