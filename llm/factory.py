"""
LLM 工厂

根据配置自动创建对应的 LLM 实例，实现一键切换模型。

使用方式：
    from llm.factory import create_llm

    # 从配置字典创建
    llm = create_llm({
        "provider": "dashscope",
        "model": "qwen-max",
        "api_key": "sk-xxxx",
    })

    # 从 LLMConfig 数据类创建
    llm = create_llm_from_config(config.llm)
"""
import logging
from typing import Union

from llm.base import LLMBase

logger = logging.getLogger(__name__)

# Provider 注册表
_PROVIDER_REGISTRY: dict[str, type[LLMBase]] = {}


def register_provider(name: str, cls: type[LLMBase]):
    """
    注册 LLM Provider

    :param name: Provider 名称（如 "dashscope", "openai"）
    :param cls: LLM 实现类
    """
    _PROVIDER_REGISTRY[name] = cls
    logger.debug(f"注册 LLM Provider: {name} -> {cls.__name__}")


def _ensure_default_providers():
    """确保默认 Provider 已注册"""
    if not _PROVIDER_REGISTRY:
        from llm.dashscope_llm import DashScopeLLM
        from llm.openai_llm import OpenAILLM

        register_provider("dashscope", DashScopeLLM)
        register_provider("openai", OpenAILLM)


def create_llm(config: Union[dict, object]) -> LLMBase:
    """
    根据配置创建 LLM 实例

    :param config: 配置字典或配置对象，必须包含:
        - provider: LLM 提供商 ("dashscope" / "openai")
        - model: 模型名称
        - api_key: API 密钥
        可选:
        - base_url: API 地址（OpenAI 兼容模型必填）
        - temperature: 温度参数
        - max_tokens: 最大 token 数
        - top_p: 核采样参数
        - timeout: 超时时间
    :return: LLMBase 实例
    """
    _ensure_default_providers()

    # 支持字典和对象两种配置格式
    if isinstance(config, dict):
        provider = config.get("provider", "dashscope")
        model = config.get("model", "qwen-turbo")
        api_key = config.get("api_key", "")
        kwargs = {
            k: v for k, v in config.items()
            if k not in ("provider", "model", "api_key") and v is not None
        }
    else:
        provider = getattr(config, "provider", "dashscope")
        model = getattr(config, "model", "qwen-turbo")
        api_key = getattr(config, "api_key", "")
        kwargs = {}
        for attr in ("base_url", "temperature", "max_tokens", "top_p", "timeout"):
            val = getattr(config, attr, None)
            if val is not None:
                kwargs[attr] = val

    # 查找 Provider
    cls = _PROVIDER_REGISTRY.get(provider)
    if cls is None:
        available = ", ".join(_PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"不支持的 LLM Provider: '{provider}', 可用: [{available}]"
        )

    logger.info(f"创建 LLM: provider={provider}, model={model}")
    return cls(model=model, api_key=api_key, **kwargs)


def list_providers() -> list[str]:
    """列出所有已注册的 Provider"""
    _ensure_default_providers()
    return list(_PROVIDER_REGISTRY.keys())
