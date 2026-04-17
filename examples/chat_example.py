"""
示例 1: 基础对话 Agent

演示如何快速创建一个带记忆的对话 Agent。
运行: python -m examples.chat_example
"""

import asyncio
from llm import LLMFactory
from memory import LocalMemory
from agents import ChatAgent
from config.loader import get_settings


async def main():
    # 1. 加载配置
    settings = get_settings(config_dir="config")

    # 2. 创建 LLM 实例
    llm = LLMFactory.create(
        provider=settings.llm.provider,
        model=settings.llm.model,
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
    )

    # 3. 创建记忆
    memory = LocalMemory(max_history=settings.memory.max_history)

    # 4. 组装 Agent
    agent = ChatAgent(
        llm=llm,
        memory=memory,
        system_prompt="你是一个友好的 AI 助手，用中文回答用户的问题。",
    )

    # 5. 多轮对话
    session_id = "demo_session"
    print("=== 对话 Agent 演示 (输入 quit 退出) ===\n")

    while True:
        user_input = input("你: ")
        if user_input.lower() in ("quit", "exit", "q"):
            break

        response = await agent.chat(user_input, session_id=session_id)
        print(f"AI: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
