"""
网络搜索工具

示例工具，演示如何实现一个 Tool。
实际使用时需替换为真实的搜索 API（如 SerpAPI、Bing Search 等）。
"""
from tools.base import Tool, ToolResult


class WebSearchTool(Tool):
    """网络搜索工具"""

    name = "web_search"
    description = "搜索互联网获取实时信息，适用于查询新闻、天气、百科等"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "max_results": {
                "type": "integer",
                "description": "最大返回结果数",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def execute(self, query: str, max_results: int = 5) -> ToolResult:
        """
        执行网络搜索

        注意：这是示例实现，实际使用时需要接入搜索 API。
        """
        # TODO: 替换为真实搜索 API 调用
        # 示例：SerpAPI / Bing Search / DuckDuckGo
        return ToolResult(
            success=True,
            output={
                "query": query,
                "results": [
                    {"title": f"搜索结果示例 - {query}", "snippet": "这是一个示例搜索结果，请接入真实搜索 API。"}
                ],
                "note": "请在 web_search.py 中接入真实搜索 API",
            },
        )
