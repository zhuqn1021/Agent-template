"""
检索器

封装向量检索 + 上下文构建逻辑。
"""
import logging
from typing import Optional

from rag.vector_store import VectorStoreBase

logger = logging.getLogger(__name__)


class Retriever:
    """
    检索器

    封装向量检索，提供格式化的上下文文本供 LLM 使用。

    使用示例：
        retriever = Retriever(vector_store, top_k=5)
        docs = await retriever.retrieve("Python 异步编程")
        context = retriever.format_context(docs)
    """

    def __init__(self, vector_store: VectorStoreBase, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        执行检索

        :param query: 查询文本
        :param top_k: 返回数量
        :param filter: 元数据过滤条件
        :return: 文档列表 [{"content": ..., "metadata": ..., "score": ...}]
        """
        k = top_k or self.top_k
        return await self.vector_store.similarity_search(query=query, k=k, filter=filter)

    @staticmethod
    def format_context(docs: list[dict], max_length: int = 8000) -> str:
        """
        将检索结果格式化为上下文文本

        :param docs: 检索结果列表
        :param max_length: 上下文最大长度
        :return: 格式化的上下文文本
        """
        if not docs:
            return ""

        parts = []
        total_len = 0
        for i, doc in enumerate(docs):
            content = doc.get("content", "")
            score = doc.get("score", 0)
            source = doc.get("metadata", {}).get("source", "unknown")

            part = f"[参考{i + 1}] (相关度: {score:.2f}, 来源: {source})\n{content}"
            if total_len + len(part) > max_length:
                break
            parts.append(part)
            total_len += len(part)

        return "\n\n---\n\n".join(parts)
