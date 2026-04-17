"""
Prompt 模板加载器 - 支持变量替换
"""

from pathlib import Path
from typing import Optional

from .logger import get_logger

logger = get_logger("utils.prompt_loader")


def load_prompt(
    filename: str,
    prompts_dir: str = "prompts",
    **variables
) -> str:
    """
    加载 Prompt 模板文件并替换变量

    Args:
        filename: 模板文件名，如 "chat_system.txt"
        prompts_dir: 模板目录，默认 "prompts"
        **variables: 模板中 {variable_name} 的替换值

    Returns:
        替换后的 prompt 字符串

    Example:
        >>> load_prompt("greet.txt", name="Alice", role="助手")
        # greet.txt 内容: "你好 {name}，我是你的{role}。"
        # 返回: "你好 Alice，我是你的助手。"
    """
    filepath = Path(prompts_dir) / filename

    if not filepath.exists():
        logger.error(f"Prompt 文件不存在: {filepath}")
        raise FileNotFoundError(f"Prompt 文件不存在: {filepath}")

    template = filepath.read_text(encoding="utf-8")

    if variables:
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.warning(f"Prompt 模板变量缺失: {e}，返回原始模板")
            return template

    return template
