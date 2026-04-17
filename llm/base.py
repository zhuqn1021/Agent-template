"""
LLM 基类定义

定义统一的 LLM 调用接口，所有 Provider 实现必须继承此基类。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Literal, AsyncIterator


@dataclass
class ChatMessage:
    """聊天消息结构"""
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list] = None  # Function Calling 返回的工具调用列表

    def to_dict(self) -> dict:
        """转为 API 请求格式"""
        msg = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


@dataclass
class LLMResponse:
    """LLM 响应结构"""
    content: str = ""
    tool_calls: list = field(default_factory=list)
    model: str = ""
    finish_reason: str = ""           # stop / tool_calls / length / error
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def is_tool_call(self) -> bool:
        """是否为工具调用响应"""
        return bool(self.tool_calls)

    @property
    def is_error(self) -> bool:
        """是否为错误响应"""
        return self.finish_reason == "error"


class LLMBase(ABC):
    """
    LLM 基类

    所有 LLM Provider 必须实现以下方法：
    - generate: 非流式生成
    - stream: 流式生成

    使用示例：
        llm = DashScopeLLM(config)
        response = await llm.generate([
            ChatMessage(role="system", content="你是一个助手"),
            ChatMessage(role="user", content="你好"),
        ])
        print(response.content)
    """

    def __init__(self, model: str, api_key: str, **kwargs):
        """
        初始化 LLM

        :param model: 模型名称
        :param api_key: API 密钥
        :param kwargs: 其他参数（temperature, max_tokens, top_p 等）
        """
        self.model = model
        self.api_key = api_key
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 2048)
        self.top_p = kwargs.get("top_p", 0.95)
        self.timeout = kwargs.get("timeout", 60)

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        非流式生成

        :param messages: 消息列表
        :param tools: 可用工具列表（Function Calling Schema）
        :param temperature: 温度参数（覆盖默认值）
        :param max_tokens: 最大 token 数（覆盖默认值）
        :return: LLMResponse
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        流式生成

        :param messages: 消息列表
        :param tools: 可用工具列表
        :param temperature: 温度参数
        :param max_tokens: 最大 token 数
        :return: 异步字符串迭代器
        """
        ...

    def _get_params(
        self,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> tuple[float, int]:
        """获取实际调用参数"""
        return (
            temperature if temperature is not None else self.temperature,
            max_tokens if max_tokens is not None else self.max_tokens,
        )
