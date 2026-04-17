"""
示例 4: RAG 知识库问答 Agent

演示如何创建带 RAG 检索增强的知识库问答 Agent。
运行: python -m examples.rag_example
"""

import asyncio
from llm import LLMFactory
from memory import LocalMemory
from rag import EmbeddingModel, QdrantVectorStore, Retriever
from agents import ChatAgent
from config.loader import get_settings


async def main():
    settings = get_settings(config_dir="config")

    # 1. 创建 LLM
    llm = LLMFactory.create(
        provider=settings.llm.provider,
        model=settings.llm.model,
        api_key=settings.llm.api_key,
    )

    # 2. 创建 Embedding + 向量库 + 检索器
    embedding = EmbeddingModel(
        provider=settings.rag.embedding_provider,
        model=settings.rag.embedding_model,
        api_key=settings.rag.embedding_api_key,
    )

    vector_store = QdrantVectorStore(
        host=settings.rag.qdrant_host,
        port=settings.rag.qdrant_port,
        collection_name=settings.rag.collection_name,
        dimension=settings.rag.embedding_dimension,
    )

    retriever = Retriever(
        embedding=embedding,
        vector_store=vector_store,
        top_k=settings.rag.top_k,
    )

    # 3. 写入一些文档 (实际项目中从文件/数据库加载)
    docs = [
        {"text": "Python 3.12 引入了类型参数语法 (PEP 695)，使泛型定义更简洁。", "metadata": {"source": "python_docs"}},
        {"text": "FastAPI 是一个现代化的 Python Web 框架，基于 Starlette 和 Pydantic。", "metadata": {"source": "fastapi_docs"}},
        {"text": "Qdrant 是一个高性能的向量数据库，支持过滤搜索和 gRPC 协议。", "metadata": {"source": "qdrant_docs"}},
    ]

    print("正在写入文档到向量库...")
    for doc in docs:
        vector = await embedding.embed(doc["text"])
        await vector_store.upsert(
            text=doc["text"],
            vector=vector,
            metadata=doc["metadata"],
        )
    print(f"已写入 {len(docs)} 条文档\n")

    # 4. 创建带 RAG 的 Agent
    memory = LocalMemory(max_history=10)
    agent = ChatAgent(
        llm=llm,
        memory=memory,
        retriever=retriever,
        system_prompt="你是一个知识库助手。根据检索到的上下文回答问题，如果上下文不包含答案就如实说明。",
    )

    # 5. 问答
    questions = [
        "Python 3.12 有什么新特性？",
        "FastAPI 是基于什么框架构建的？",
        "Qdrant 支持什么协议？",
    ]

    session_id = "rag_demo"
    for q in questions:
        print(f"问: {q}")
        resp = await agent.chat(q, session_id=session_id)
        print(f"答: {resp}\n")


if __name__ == "__main__":
    asyncio.run(main())
