"""
Memory 基类

定义统一的记忆管理接口。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class MemoryMessage:
    """记忆消息"""
    role: Literal["user", "assistant", "system"]
    content: str


class MemoryBase(ABC):
    """
    记忆管理基类

    所有记忆后端实现必须继承此类。
    """

    @abstractmethod
    async def add(self, session_id: str, role: str, content: str) -> bool:
        """
        添加消息到记忆

        :param session_id: 会话 ID
        :param role: 角色（user/assistant/system）
        :param content: 消息内容
        :return: 是否成功
        """
        ...

    @abstractmethod
    async def get_history(
        self, session_id: str, limit: int = 10
    ) -> list[MemoryMessage]:
        """
        获取对话历史

        :param session_id: 会话 ID
        :param limit: 最大消息数（滑动窗口）
        :return: 消息列表
        """
        ...

    @abstractmethod
    async def clear(self, session_id: str) -> bool:
        """
        清除会话历史

        :param session_id: 会话 ID
        :return: 是否成功
        """
        ...
