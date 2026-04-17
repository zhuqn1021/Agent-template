"""
配置数据类 - 使用 dataclass 定义所有配置结构
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class LLMConfig:
    """LLM 模型配置"""
    provider: str = "dashscope"           # dashscope / openai
    model: str = "qwen-plus"
    api_key: str = ""
    base_url: Optional[str] = None        # OpenAI 兼容接口地址
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    timeout: int = 60

    # 多模型配置：可在运行时通过 key 切换
    # 例: models.deepseek -> provider=openai, model=deepseek-chat, base_url=...
    extra_models: dict = field(default_factory=dict)


@dataclass
class RAGConfig:
    """RAG 检索增强配置"""
    enabled: bool = False
    # Embedding
    embedding_provider: str = "dashscope"  # dashscope / openai
    embedding_model: str = "text-embedding-v3"
    embedding_api_key: str = ""
    embedding_dimension: int = 1024
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_use_grpc: bool = True
    collection_name: str = "agent_docs"
    # 检索参数
    top_k: int = 5
    score_threshold: float = 0.5


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    backend: str = "local"                 # local / redis
    max_history: int = 20
    # Redis 配置（backend=redis 时生效）
    redis_url: str = "redis://localhost:6379/0"
    redis_prefix: str = "agent:memory:"
    ttl_seconds: int = 86400               # 24小时


@dataclass
class ServerConfig:
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class Settings:
    """全局配置（顶层聚合）"""
    app_name: str = "Agent Template"
    version: str = "1.0.0"
    env: str = "development"               # development / production

    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    # 全局 prompts 目录
    prompts_dir: str = "prompts"
    # 日志级别
    log_level: str = "INFO"
