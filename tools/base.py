"""
Tool 基类定义

定义工具的统一接口，支持 OpenAI Function Calling 格式。
"""
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool = True
    output: Any = None
    error: Optional[str] = None

    def to_str(self) -> str:
        """转为字符串（用于传回 LLM）"""
        if self.success:
            if isinstance(self.output, (dict, list)):
                return json.dumps(self.output, ensure_ascii=False, indent=2)
            return str(self.output)
        return f"[工具执行失败] {self.error}"


class Tool(ABC):
    """
    工具基类

    所有自定义工具必须继承此类并实现 execute() 方法。

    使用示例：
        class WeatherTool(Tool):
            name = "get_weather"
            description = "查询城市天气"
            parameters = {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                },
                "required": ["city"],
            }

            async def execute(self, city: str) -> ToolResult:
                # 调用天气 API
                return ToolResult(success=True, output={"city": city, "temp": "25C"})
    """

    # 子类必须定义这三个属性
    name: str = ""
    description: str = ""
    parameters: dict = field(default_factory=dict) if False else {}

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        执行工具

        :param kwargs: 工具参数（由 LLM Function Calling 传入）
        :return: ToolResult
        """
        ...

    def get_function_schema(self) -> dict:
        """
        生成 OpenAI Function Calling 格式的 Schema

        :return: Function Schema 字典
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters or {"type": "object", "properties": {}},
        }

    async def safe_execute(self, **kwargs) -> ToolResult:
        """
        安全执行（自动捕获异常）

        :param kwargs: 工具参数
        :return: ToolResult
        """
        try:
            return await self.execute(**kwargs)
        except Exception as e:
            return ToolResult(success=False, error=f"{self.name} 执行失败: {str(e)}")
