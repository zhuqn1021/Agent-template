"""
示例 2: 工具调用 Agent (ReAct)

演示如何创建一个带工具的任务型 Agent，支持自动调用搜索/代码执行等工具。
运行: python -m examples.tool_agent_example
"""

import asyncio
from llm import LLMFactory
from memory import LocalMemory
from tools import ToolRegistry
from tools.builtin import WebSearchTool, CodeExecutorTool
from agents import TaskAgent
from config.loader import get_settings


async def main():
    # 1. 加载配置
    settings = get_settings(config_dir="config")

    # 2. 创建 LLM (使用支持 function calling 的模型)
    llm = LLMFactory.create(
        provider=settings.llm.provider,
        model=settings.llm.model,
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        temperature=0.3,  # 工具调用场景建议低温度
    )

    # 3. 注册工具
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    registry.register(CodeExecutorTool())

    # 4. 创建记忆
    memory = LocalMemory(max_history=10)

    # 5. 组装 TaskAgent
    agent = TaskAgent(
        llm=llm,
        memory=memory,
        tool_registry=registry,
        system_prompt="你是一个能力强大的 AI 助手。你可以搜索网络信息、执行 Python 代码来帮助用户解决问题。",
        max_tool_rounds=5,
    )

    # 6. 执行任务
    print("=== 工具 Agent 演示 ===\n")

    tasks = [
        "帮我计算斐波那契数列的第 20 项",
        "搜索一下 Python 3.12 有什么新特性",
    ]

    session_id = "tool_demo"
    for task in tasks:
        print(f"任务: {task}")
        response = await agent.chat(task, session_id=session_id)
        print(f"结果: {response}\n")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
