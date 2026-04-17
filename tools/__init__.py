"""
Tools 工具链系统

提供 Function Calling 的工具注册与执行框架。

使用方式：
    from tools import ToolRegistry, tool

    registry = ToolRegistry()

    @registry.register
    class MyTool(Tool):
        name = "my_tool"
        description = "我的工具"
        ...
"""

from tools.base import Tool, ToolResult
from tools.registry import ToolRegistry

__all__ = ["Tool", "ToolResult", "ToolRegistry"]
