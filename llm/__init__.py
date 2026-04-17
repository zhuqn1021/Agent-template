"""
LLM 多模型抽象层

支持一键切换不同的 LLM Provider：
- DashScope（阿里云通义千问系列）
- OpenAI（GPT 系列，兼容 DeepSeek/Moonshot/GLM 等）

使用方式：
    from llm import create_llm
    llm = create_llm(config)
    response = await llm.generate(messages)
"""

from llm.base import LLMBase, LLMResponse, ChatMessage
from llm.factory import create_llm

__all__ = ["LLMBase", "LLMResponse", "ChatMessage", "create_llm"]
