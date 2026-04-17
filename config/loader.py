"""
配置加载器 - 支持 YAML 文件 + 环境变量覆盖

加载优先级（从低到高）：
  1. dataclass 默认值
  2. settings.yaml
  3. settings.{env}.yaml
  4. 环境变量（AGENT_ 前缀）
"""

import os
import yaml
from pathlib import Path
from typing import Optional

from .settings import Settings, LLMConfig, RAGConfig, MemoryConfig, ServerConfig


class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("config")

    def load(self, env: Optional[str] = None) -> Settings:
        """
        加载配置，合并 YAML + 环境变量

        Args:
            env: 环境名称，默认从 AGENT_ENV 环境变量读取
        """
        env = env or os.getenv("AGENT_ENV", "development")

        # 1. 加载基础 YAML
        base_data = self._load_yaml("settings.yaml")
        # 2. 加载环境 YAML（覆盖）
        env_data = self._load_yaml(f"settings.{env}.yaml")
        # 3. 合并
        merged = self._deep_merge(base_data, env_data)
        # 4. 环境变量覆盖
        merged = self._apply_env_overrides(merged)

        # 5. 构造 Settings 对象
        return self._build_settings(merged, env)

    def _load_yaml(self, filename: str) -> dict:
        filepath = self.config_dir / filename
        if not filepath.exists():
            return {}
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """深度合并字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _apply_env_overrides(data: dict) -> dict:
        """
        环境变量覆盖规则：
          AGENT_LLM_API_KEY  ->  data["llm"]["api_key"]
          AGENT_RAG_QDRANT_HOST  ->  data["rag"]["qdrant_host"]
        """
        prefix = "AGENT_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix):].lower().split("_", 1)
            if len(parts) == 2:
                section, field = parts[0], parts[1]
                if section not in data:
                    data[section] = {}
                data[section][field] = value
            elif len(parts) == 1:
                data[parts[0]] = value
        return data

    @staticmethod
    def _build_settings(data: dict, env: str) -> Settings:
        """从字典构建 Settings 对象"""
        llm_data = data.get("llm", {})
        rag_data = data.get("rag", {})
        memory_data = data.get("memory", {})
        server_data = data.get("server", {})

        return Settings(
            app_name=data.get("app_name", "Agent Template"),
            version=data.get("version", "1.0.0"),
            env=env,
            llm=LLMConfig(**{k: v for k, v in llm_data.items() if k in LLMConfig.__dataclass_fields__}),
            rag=RAGConfig(**{k: v for k, v in rag_data.items() if k in RAGConfig.__dataclass_fields__}),
            memory=MemoryConfig(**{k: v for k, v in memory_data.items() if k in MemoryConfig.__dataclass_fields__}),
            server=ServerConfig(**{k: v for k, v in server_data.items() if k in ServerConfig.__dataclass_fields__}),
            prompts_dir=data.get("prompts_dir", "prompts"),
            log_level=data.get("log_level", "INFO"),
        )


# ---------- 快捷函数 ----------

_settings_cache: Optional[Settings] = None


def get_settings(config_dir: Optional[str] = None, env: Optional[str] = None) -> Settings:
    """获取全局配置（单例缓存）"""
    global _settings_cache
    if _settings_cache is None:
        loader = ConfigLoader(config_dir)
        _settings_cache = loader.load(env)
    return _settings_cache


def reload_settings(config_dir: Optional[str] = None, env: Optional[str] = None) -> Settings:
    """强制重新加载配置"""
    global _settings_cache
    loader = ConfigLoader(config_dir)
    _settings_cache = loader.load(env)
    return _settings_cache
