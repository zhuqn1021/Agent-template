"""
示例 3: 模型热切换

演示如何在运行时切换不同的 LLM Provider / 模型。
运行: python -m examples.model_switch_example
"""

import asyncio
from llm import LLMFactory
from agents import ChatAgent
from memory import LocalMemory


async def main():
    memory = LocalMemory(max_history=5)

    # ==========================================
    # 方式 1: 通义千问 (DashScope)
    # ==========================================
    print("=== 使用通义千问 ===")
    qwen_llm = LLMFactory.create(
        provider="dashscope",
        model="qwen-plus",
        api_key="your-dashscope-api-key",
    )
    agent = ChatAgent(llm=qwen_llm, memory=memory, system_prompt="你是通义千问助手")
    resp = await agent.chat("你好，你是什么模型？", session_id="s1")
    print(f"回复: {resp}\n")

    # ==========================================
    # 方式 2: DeepSeek (OpenAI 兼容)
    # ==========================================
    print("=== 切换到 DeepSeek ===")
    deepseek_llm = LLMFactory.create(
        provider="openai",
        model="deepseek-chat",
        api_key="your-deepseek-api-key",
        base_url="https://api.deepseek.com/v1",
    )
    agent.llm = deepseek_llm  # 运行时热切换！
    resp = await agent.chat("你好，你是什么模型？", session_id="s2")
    print(f"回复: {resp}\n")

    # ==========================================
    # 方式 3: 本地 Ollama
    # ==========================================
    print("=== 切换到本地 Ollama ===")
    ollama_llm = LLMFactory.create(
        provider="openai",
        model="qwen2:7b",
        api_key="ollama",  # Ollama 不需要真实 key
        base_url="http://localhost:11434/v1",
    )
    agent.llm = ollama_llm
    resp = await agent.chat("你好，你是什么模型？", session_id="s3")
    print(f"回复: {resp}\n")

    # ==========================================
    # 方式 4: GLM-4 (智谱)
    # ==========================================
    print("=== 切换到 GLM-4 ===")
    glm_llm = LLMFactory.create(
        provider="openai",
        model="glm-4",
        api_key="your-zhipu-api-key",
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    agent.llm = glm_llm
    resp = await agent.chat("你好，你是什么模型？", session_id="s4")
    print(f"回复: {resp}\n")


if __name__ == "__main__":
    asyncio.run(main())
