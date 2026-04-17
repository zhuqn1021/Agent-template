"""
Agent 核心层

提供 ChatAgent（对话型）和 TaskAgent（任务型，带工具调用）。
"""
from agents.base import AgentBase, AgentResponse
from agents.chat_agent import ChatAgent
from agents.task_agent import TaskAgent

__all__ = ["AgentBase", "AgentResponse", "ChatAgent", "TaskAgent"]
