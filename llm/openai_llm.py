"""
OpenAI 兼容 LLM 实现

支持所有 OpenAI API 兼容的模型服务：
- OpenAI (GPT-4o / GPT-4 / GPT-3.5)
- DeepSeek (deepseek-chat / deepseek-coder)
- Moonshot (moonshot-v1-8k / moonshot-v1-32k)
- 智谱 GLM (glm-4 / glm-3-turbo)
- 零一万物 (yi-large / yi-medium)
- 本地部署 (Ollama / vLLM / LocalAI)

只需修改 base_url 和 api_key 即可切换。
"""
import json
import logging
from typing import Optional, AsyncIterator

from llm.base import LLMBase, LLMResponse, ChatMessage

logger = logging.getLogger(__name__)


class OpenAILLM(LLMBase):
    """
    OpenAI 兼容 LLM 实现

    配置示例 (settings.yaml):
        # OpenAI 原生
        llm:
          provider: "openai"
          model: "gpt-4o"
          api_key: "sk-xxxx"

        # DeepSeek
        llm:
          provider: "openai"
          model: "deepseek-chat"
          api_key: "sk-xxxx"
          base_url: "https://api.deepseek.com/v1"

        # Moonshot
        llm:
          provider: "openai"
          model: "moonshot-v1-32k"
          api_key: "sk-xxxx"
          base_url: "https://api.moonshot.cn/v1"

        # 本地 Ollama
        llm:
          provider: "openai"
          model: "qwen2:7b"
          api_key: "ollama"
          base_url: "http://localhost:11434/v1"
    """

    def __init__(self, model: str, api_key: str, **kwargs):
        super().__init__(model, api_key, **kwargs)
        self.base_url = kwargs.get("base_url")

    def _get_client(self):
        """获取 OpenAI 异步客户端"""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai 未安装，请运行: pip install openai")

        client_kwargs = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        return AsyncOpenAI(**client_kwargs)

    async def generate(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """非流式调用 OpenAI 兼容 API"""
        temp, tokens = self._get_params(temperature, max_tokens)

        try:
            client = self._get_client()
            formatted_messages = [m.to_dict() for m in messages]

            call_kwargs = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": temp,
                "max_tokens": tokens,
                "top_p": self.top_p,
            }
            if tools:
                call_kwargs["tools"] = [
                    {"type": "function", "function": t} for t in tools
                ]
                call_kwargs["tool_choice"] = "auto"

            response = await client.chat.completions.create(**call_kwargs)

            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = choice.finish_reason or "stop"

            # 提取 tool_calls
            tool_calls = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })
                finish_reason = "tool_calls"

            # Token 使用量
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                model=self.model,
                finish_reason=finish_reason,
                tokens_used=prompt_tokens + completion_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        except ImportError:
            return LLMResponse(content="openai SDK 未安装", finish_reason="error")
        except Exception as e:
            logger.error(f"OpenAI 调用异常: {e}", exc_info=True)
            return LLMResponse(
                content=f"调用异常: {str(e)}", model=self.model, finish_reason="error"
            )

    async def stream(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式调用 OpenAI 兼容 API"""
        temp, tokens = self._get_params(temperature, max_tokens)

        try:
            client = self._get_client()
            formatted_messages = [m.to_dict() for m in messages]

            call_kwargs = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": temp,
                "max_tokens": tokens,
                "top_p": self.top_p,
                "stream": True,
            }
            if tools:
                call_kwargs["tools"] = [
                    {"type": "function", "function": t} for t in tools
                ]

            response = await client.chat.completions.create(**call_kwargs)

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except ImportError:
            yield "[错误] openai SDK 未安装"
        except Exception as e:
            logger.error(f"OpenAI 流式调用异常: {e}", exc_info=True)
            yield f"[错误] {str(e)}"
