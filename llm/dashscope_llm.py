"""
DashScope LLM 实现

支持阿里云通义千问系列模型：
- qwen-turbo / qwen-plus / qwen-max / qwen3-max
- 以及所有 DashScope 兼容模型
"""
import json
import logging
from typing import Optional, AsyncIterator

from llm.base import LLMBase, LLMResponse, ChatMessage

logger = logging.getLogger(__name__)


class DashScopeLLM(LLMBase):
    """
    DashScope LLM 实现

    使用阿里云 DashScope SDK 调用通义千问系列模型。

    配置示例 (settings.yaml):
        llm:
          provider: "dashscope"
          model: "qwen-max"
          api_key: "sk-xxxx"
    """

    async def generate(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """非流式调用 DashScope API"""
        temp, tokens = self._get_params(temperature, max_tokens)

        try:
            import dashscope
            from dashscope import Generation

            dashscope.api_key = self.api_key

            # 构建请求参数
            formatted_messages = [m.to_dict() for m in messages]
            call_kwargs = {
                "model": self.model,
                "messages": formatted_messages,
                "result_format": "message",
                "temperature": temp,
                "max_tokens": tokens,
                "top_p": self.top_p,
            }

            # 添加工具定义（Function Calling）
            if tools:
                call_kwargs["tools"] = tools

            response = Generation.call(**call_kwargs)

            # 兼容不同版本的状态码获取方式
            status_code = getattr(response, "status_code", None) or getattr(
                response, "status", None
            )

            if status_code == 200:
                choice = response.output.choices[0]
                message = choice.message
                content = message.content or ""
                finish_reason = getattr(choice, "finish_reason", "stop")

                # 提取 tool_calls
                tool_calls = []
                raw_tool_calls = getattr(message, "tool_calls", None)
                if raw_tool_calls:
                    for tc in raw_tool_calls:
                        tool_calls.append({
                            "id": getattr(tc, "id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        })
                    finish_reason = "tool_calls"

                # 提取 token 使用量
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0

                return LLMResponse(
                    content=content,
                    tool_calls=tool_calls,
                    model=self.model,
                    finish_reason=finish_reason,
                    tokens_used=prompt_tokens + completion_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            else:
                error_msg = getattr(response, "message", str(response))
                logger.error(f"DashScope API 调用失败: {error_msg}")
                return LLMResponse(
                    content=f"API 调用失败: {error_msg}",
                    model=self.model,
                    finish_reason="error",
                )

        except ImportError:
            logger.error("dashscope 未安装，请运行: pip install dashscope")
            return LLMResponse(content="dashscope SDK 未安装", finish_reason="error")
        except Exception as e:
            logger.error(f"DashScope 调用异常: {e}", exc_info=True)
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
        """流式调用 DashScope API"""
        temp, tokens = self._get_params(temperature, max_tokens)

        try:
            import dashscope
            from dashscope import Generation

            dashscope.api_key = self.api_key

            formatted_messages = [m.to_dict() for m in messages]
            call_kwargs = {
                "model": self.model,
                "messages": formatted_messages,
                "result_format": "message",
                "temperature": temp,
                "max_tokens": tokens,
                "top_p": self.top_p,
                "stream": True,
            }
            if tools:
                call_kwargs["tools"] = tools

            response = Generation.call(**call_kwargs)

            for chunk in response:
                chunk_status = getattr(chunk, "status_code", None) or getattr(
                    chunk, "status", None
                )
                if chunk_status == 200 and chunk.output and chunk.output.choices:
                    delta = chunk.output.choices[0].message.content
                    if delta:
                        yield delta

        except ImportError:
            yield "[错误] dashscope SDK 未安装"
        except Exception as e:
            logger.error(f"DashScope 流式调用异常: {e}", exc_info=True)
            yield f"[错误] {str(e)}"
