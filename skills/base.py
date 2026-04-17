"""
Skill 基类

Skill 是一个可组合的能力单元，封装了：
- LLM 调用
- 提示词模板
- 输出解析
- 错误处理

与 Tool 的区别：
- Tool: 执行确定性操作（API调用、文件读写、代码执行）
- Skill: 利用 LLM 完成智能任务（文本分析、内容生成、信息提取）
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from llm.base import LLMBase, ChatMessage, LLMResponse

logger = logging.getLogger(__name__)


class Skill(ABC):
    """
    Skill 基类

    使用示例：
        class SummarizeSkill(Skill):
            name = "summarize"
            system_prompt = "你是一个文本摘要专家。"

            async def run(self, text: str) -> str:
                response = await self.call_llm(f"请总结以下内容：\\n{text}")
                return response.content
    """

    name: str = ""
    description: str = ""
    system_prompt: str = ""

    def __init__(self, llm: Optional[LLMBase] = None):
        """
        初始化 Skill

        :param llm: LLM 实例（不传则需要在调用时注入）
        """
        self._llm = llm

    def set_llm(self, llm: LLMBase):
        """设置 LLM 实例"""
        self._llm = llm

    @property
    def llm(self) -> LLMBase:
        """获取 LLM 实例"""
        if self._llm is None:
            raise RuntimeError(f"Skill '{self.name}' 未设置 LLM 实例，请先调用 set_llm()")
        return self._llm

    async def call_llm(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        调用 LLM 的便捷方法

        :param user_message: 用户消息
        :param system_message: 系统消息（不传则使用 self.system_prompt）
        :param temperature: 温度参数
        :param max_tokens: 最大 token 数
        :return: LLMResponse
        """
        messages = []
        sys_msg = system_message or self.system_prompt
        if sys_msg:
            messages.append(ChatMessage(role="system", content=sys_msg))
        messages.append(ChatMessage(role="user", content=user_message))

        return await self.llm.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """
        执行技能

        子类必须实现此方法。
        """
        ...

    async def safe_run(self, *args, **kwargs) -> Any:
        """安全执行（自动捕获异常）"""
        try:
            return await self.run(*args, **kwargs)
        except Exception as e:
            logger.error(f"Skill '{self.name}' 执行失败: {e}", exc_info=True)
            return None
