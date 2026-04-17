"""
文件读取工具

支持读取本地文本文件内容。
"""
import os
from tools.base import Tool, ToolResult


class FileReaderTool(Tool):
    """文件读取工具"""

    name = "read_file"
    description = "读取本地文件内容，支持 txt、json、csv、md 等文本文件"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件路径",
            },
            "encoding": {
                "type": "string",
                "description": "文件编码",
                "default": "utf-8",
            },
        },
        "required": ["file_path"],
    }

    async def execute(self, file_path: str, encoding: str = "utf-8") -> ToolResult:
        """读取文件内容"""
        if not os.path.exists(file_path):
            return ToolResult(success=False, error=f"文件不存在: {file_path}")

        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()

            # 限制返回内容长度，避免超出 LLM token 限制
            max_length = 10000
            if len(content) > max_length:
                content = content[:max_length] + f"\n...[截断，原文共 {len(content)} 字符]"

            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, error=f"读取文件失败: {str(e)}")
