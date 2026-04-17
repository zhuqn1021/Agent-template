"""
Embedding 抽象层

支持多种 Embedding 模型：
- DashScope text-embedding-v2
- OpenAI text-embedding-ada-002 / text-embedding-3-small
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class EmbeddingBase(ABC):
    """Embedding 基类"""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """单条文本向量化"""
        ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量文本向量化"""
        ...


class DashScopeEmbedding(EmbeddingBase):
    """
    DashScope Embedding 实现

    使用阿里云 text-embedding-v2 模型，向量维度 1536。
    """

    def __init__(self, model: str = "text-embedding-v2", api_key: str = ""):
        self.model = model
        self.api_key = api_key

    def embed_query(self, text: str) -> list[float]:
        """单条文本向量化"""
        text = (text or "").strip()
        if not text:
            return []

        # 限制长度
        if len(text) > 2048:
            text = text[:2048]

        try:
            import dashscope
            from dashscope import TextEmbedding

            dashscope.api_key = self.api_key
            response = TextEmbedding.call(model=self.model, input=text)

            if response.status_code == 200:
                return response.output["embeddings"][0]["embedding"]
            else:
                logger.error(f"Embedding 失败: {getattr(response, 'message', response)}")
                return []
        except ImportError:
            logger.error("dashscope 未安装")
            return []
        except Exception as e:
            logger.error(f"Embedding 异常: {e}")
            return []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量文本向量化"""
        valid_texts = [t.strip()[:4096] for t in texts if t and t.strip()]
        if not valid_texts:
            return []

        try:
            import dashscope
            from dashscope import TextEmbedding

            dashscope.api_key = self.api_key
            response = TextEmbedding.call(model=self.model, input=valid_texts)

            if response.status_code == 200:
                return [item["embedding"] for item in response.output["embeddings"]]
            else:
                logger.error(f"批量 Embedding 失败: {getattr(response, 'message', response)}")
                return []
        except Exception as e:
            logger.error(f"批量 Embedding 异常: {e}")
            return []


class OpenAIEmbedding(EmbeddingBase):
    """
    OpenAI Embedding 实现

    支持 text-embedding-3-small / text-embedding-ada-002 等模型。
    也兼容其他支持 OpenAI Embedding API 的服务。
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = "", base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def _get_client(self):
        from openai import OpenAI
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    def embed_query(self, text: str) -> list[float]:
        """单条文本向量化"""
        try:
            client = self._get_client()
            response = client.embeddings.create(model=self.model, input=text)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI Embedding 异常: {e}")
            return []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量文本向量化"""
        try:
            client = self._get_client()
            response = client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI 批量 Embedding 异常: {e}")
            return []
