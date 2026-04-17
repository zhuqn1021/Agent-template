"""
向量存储抽象层

支持 Qdrant 向量数据库，预留 Chroma/FAISS 接口。
"""
import uuid
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any

from rag.embeddings import EmbeddingBase

logger = logging.getLogger(__name__)


class VectorStoreBase(ABC):
    """向量存储基类"""

    @abstractmethod
    async def add_texts(self, texts: list[str], metadatas: Optional[list[dict]] = None) -> list[str]:
        ...

    @abstractmethod
    async def similarity_search(self, query: str, k: int = 5, filter: Optional[dict] = None) -> list[dict]:
        ...


class QdrantVectorStore(VectorStoreBase):
    """
    Qdrant 向量存储

    使用示例：
        from rag.embeddings import DashScopeEmbedding
        embedding = DashScopeEmbedding(api_key="sk-xxx")
        store = QdrantVectorStore(
            collection_name="my_knowledge",
            embedding=embedding,
            host="localhost",
            port=6333,
        )
        await store.add_texts(["文档内容1", "文档内容2"])
        results = await store.similarity_search("查询内容")
    """

    def __init__(
        self,
        collection_name: str,
        embedding: EmbeddingBase,
        host: str = "localhost",
        port: int = 6333,
        api_key: str = "",
        prefer_grpc: bool = True,
        embedding_dim: int = 1536,
        score_threshold: float = 0.3,
    ):
        self.collection_name = collection_name
        self.embedding = embedding
        self.host = host
        self.port = port
        self.api_key = api_key
        self.prefer_grpc = prefer_grpc
        self.embedding_dim = embedding_dim
        self.score_threshold = score_threshold

    def _get_client(self):
        """获取 Qdrant 客户端"""
        from qdrant_client import QdrantClient

        kwargs = {
            "url": f"http://{self.host}:{self.port}",
            "timeout": 120,
            "prefer_grpc": self.prefer_grpc,
        }
        if self.prefer_grpc:
            kwargs["grpc_port"] = 6334
        if self.api_key:
            kwargs["api_key"] = self.api_key

        return QdrantClient(**kwargs)

    def _ensure_collection(self, client):
        """确保 Collection 存在"""
        from qdrant_client.http import models

        collections = client.get_collections().collections
        names = [c.name for c in collections]

        if self.collection_name not in names:
            logger.info(f"创建 Collection: {self.collection_name}")
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dim,
                    distance=models.Distance.COSINE,
                ),
            )

    async def add_texts(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> list[str]:
        """添加文本到向量存储"""
        try:
            from qdrant_client.http import models

            client = self._get_client()
            self._ensure_collection(client)

            vectors = self.embedding.embed_documents(texts)
            if not vectors:
                return []

            points = []
            ids = []
            for idx, (text, vector) in enumerate(zip(texts, vectors)):
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
                metadata = metadatas[idx] if metadatas else {}
                points.append(
                    models.PointStruct(
                        id=doc_id,
                        vector=vector,
                        payload={"content": text, **metadata},
                    )
                )

            client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            logger.info(f"添加 {len(points)} 条文档到 {self.collection_name}")
            return ids
        except Exception as e:
            logger.error(f"向量存储添加失败: {e}")
            return []

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        """相似度检索"""
        try:
            client = self._get_client()
            query_vector = self.embedding.embed_query(query)
            if not query_vector:
                return []

            query_params: dict[str, Any] = {
                "collection_name": self.collection_name,
                "query": query_vector,
                "limit": k,
                "with_payload": True,
                "with_vectors": False,
            }

            # 构建过滤条件
            if filter:
                from qdrant_client.http import models
                conditions = [
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                    for key, value in filter.items()
                ]
                query_params["query_filter"] = models.Filter(must=conditions)

            results = client.query_points(**query_params)

            docs = []
            for hit in results.points:
                score = hit.score or 0
                if score >= self.score_threshold:
                    docs.append({
                        "content": hit.payload.get("content", ""),
                        "metadata": hit.payload or {},
                        "score": score,
                    })

            logger.info(f"检索完成: 原始={len(results.points)}, 过滤后={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []
