"""
日志工具 - 统一日志格式，支持彩色终端输出
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    获取标准化 Logger

    Args:
        name: 日志名称，建议用模块路径如 "agent.llm"
        level: 日志级别，默认 INFO

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level = (level or "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level, logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # 防止日志向上传播导致重复输出
    logger.propagate = False

    return logger
