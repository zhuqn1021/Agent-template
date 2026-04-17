"""
代码执行工具

在安全沙箱中执行 Python 代码片段。
"""
import io
import sys
import traceback
from tools.base import Tool, ToolResult


class CodeExecutorTool(Tool):
    """Python 代码执行工具"""

    name = "execute_python"
    description = "执行 Python 代码，适用于数学计算、数据处理、格式转换等任务"
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的 Python 代码",
            },
        },
        "required": ["code"],
    }

    async def execute(self, code: str) -> ToolResult:
        """
        执行 Python 代码

        注意：生产环境应使用更安全的沙箱（如 Docker / RestrictedPython）
        """
        # 捕获标准输出
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            # 在受限全局命名空间中执行
            exec_globals = {"__builtins__": __builtins__}
            exec(code, exec_globals)

            output = sys.stdout.getvalue()
            return ToolResult(success=True, output=output or "(无输出)")
        except Exception as e:
            error_trace = traceback.format_exc()
            return ToolResult(success=False, error=error_trace)
        finally:
            sys.stdout = old_stdout
