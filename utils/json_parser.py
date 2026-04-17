"""
JSON 容错解析器 

处理 LLM 返回的不规范 JSON：
  1. 提取 ```json ... ``` 代码块
  2. 定位 { } 边界
  3. 修复尾随逗号
  4. 解析并返回
"""

import re
import json
from typing import Any, Optional

from .logger import get_logger

logger = get_logger("utils.json_parser")


def safe_parse_json(text: str, default: Any = None) -> Any:
    """
    安全解析可能不规范的 JSON 文本

    Args:
        text: LLM 原始输出文本
        default: 解析失败时的默认返回值

    Returns:
        解析后的 Python 对象，或 default
    """
    if not text or not text.strip():
        return default

    # Step 1: 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 2: 提取 markdown 代码块
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            text = code_block_match.group(1)

    # Step 3: 定位 JSON 边界
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        # 尝试数组
        start = text.find("[")
        end = text.rfind("]")

    if start != -1 and end > start:
        json_str = text[start:end + 1]

        # Step 4: 修复尾随逗号 (,} 或 ,])
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 边界提取后仍解析失败: {e}")
            logger.debug(f"提取内容: {json_str[:200]}...")

    logger.error(f"JSON 解析彻底失败，原文前200字符: {text[:200]}")
    return default
