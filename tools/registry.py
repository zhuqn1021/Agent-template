"""
工具注册中心

管理所有可用工具的注册、查找和执行。
支持装饰器注册和手动注册两种方式。

使用方式：
    registry = ToolRegistry()

    # 方式1: 装饰器注册
    @registry.register
    class MyTool(Tool): ...

    # 方式2: 手动注册
    registry.add(MyTool())

    # 获取所有工具的 Function Schema（传给 LLM）
    schemas = registry.get_function_schemas()

    # 执行工具
    result = await registry.call("my_tool", city="北京")
"""
import json
import logging
from typing import Optional

from tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    工具注册中心

    管理所有工具的生命周期，提供：
    - 工具注册（装饰器/手动）
    - 工具查找
    - 工具 Schema 生成（OpenAI Function Calling 格式）
    - 工具执行（按名调用）
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool_cls_or_instance):
        """
        注册工具（支持类或实例）

        可作为装饰器使用：
            @registry.register
            class MyTool(Tool): ...

        也可直接传入实例：
            registry.register(MyTool())
        """
        if isinstance(tool_cls_or_instance, type):
            # 传入的是类，实例化后注册
            instance = tool_cls_or_instance()
        else:
            instance = tool_cls_or_instance

        if not instance.name:
            raise ValueError(f"工具必须定义 name 属性: {type(instance).__name__}")

        self._tools[instance.name] = instance
        logger.debug(f"注册工具: {instance.name} - {instance.description}")
        return tool_cls_or_instance  # 返回原始类（支持装饰器用法）

    def add(self, tool: Tool):
        """手动注册工具实例"""
        self.register(tool)

    def get(self, name: str) -> Optional[Tool]:
        """按名获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """列出所有已注册的工具名"""
        return list(self._tools.keys())

    def get_function_schemas(self) -> list[dict]:
        """
        获取所有工具的 Function Calling Schema

        返回格式兼容 OpenAI / DashScope Function Calling API。

        :return: Schema 列表
        """
        return [tool.get_function_schema() for tool in self._tools.values()]

    async def call(self, name: str, **kwargs) -> ToolResult:
        """
        按名执行工具

        :param name: 工具名称
        :param kwargs: 工具参数
        :return: ToolResult
        """
        tool = self._tools.get(name)
        if not tool:
            available = ", ".join(self._tools.keys())
            return ToolResult(
                success=False,
                error=f"工具不存在: '{name}', 可用工具: [{available}]",
            )

        logger.info(f"执行工具: {name}, 参数: {kwargs}")
        result = await tool.safe_execute(**kwargs)
        logger.info(f"工具执行完成: {name}, 成功: {result.success}")
        return result

    async def call_from_llm_response(self, tool_call: dict) -> ToolResult:
        """
        从 LLM 的 tool_call 响应中执行工具

        :param tool_call: LLM 返回的 tool_call 字典
            {
                "id": "call_xxx",
                "function": {"name": "get_weather", "arguments": '{"city": "北京"}'}
            }
        :return: ToolResult
        """
        func_info = tool_call.get("function", {})
        name = func_info.get("name", "")
        arguments_str = func_info.get("arguments", "{}")

        try:
            kwargs = json.loads(arguments_str)
        except json.JSONDecodeError:
            return ToolResult(success=False, error=f"工具参数解析失败: {arguments_str}")

        return await self.call(name, **kwargs)

    def __len__(self):
        return len(self._tools)

    def __contains__(self, name: str):
        return name in self._tools
