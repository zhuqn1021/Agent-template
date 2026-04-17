# Agent Template 项目指南

> 一个现代化 AI Agent 项目模板，支持多模型切换、工具调用、RAG 检索增强、对话记忆等能力。

---

## 目录结构

```
agent_template/
├── llm/                    # LLM 多模型抽象层
│   ├── base.py             # 统一接口定义 (ChatMessage, LLMResponse, LLMBase)
│   ├── dashscope_llm.py    # 通义千问 DashScope 实现
│   ├── openai_llm.py       # OpenAI 兼容实现 (支持 DeepSeek/GLM/Ollama)
│   └── factory.py          # 工厂模式，按配置创建 LLM 实例
│
├── tools/                  # 工具链系统 (Function Calling)
│   ├── base.py             # Tool 基类 + ToolResult
│   ├── registry.py         # 工具注册中心
│   └── builtin/            # 内置工具
│       ├── web_search.py   # 网络搜索
│       ├── file_reader.py  # 文件读取
│       └── code_executor.py# Python 代码执行
│
├── skills/                 # 技能系统 (可组合的 LLM 能力单元)
│   ├── base.py             # Skill 基类
│   ├── json_extractor.py   # JSON 容错解析
│   ├── document_parser.py  # 文档解析
│   └── report_generator.py # 报告生成
│
├── agents/                 # Agent 核心层
│   ├── base.py             # AgentBase 基类
│   ├── chat_agent.py       # 对话型 Agent
│   └── task_agent.py       # 任务型 Agent (ReAct 工具调用循环)
│
├── rag/                    # RAG 检索增强
│   ├── embeddings.py       # Embedding 模型 (DashScope/OpenAI)
│   ├── vector_store.py     # Qdrant 向量数据库
│   └── retriever.py        # 检索器
│
├── memory/                 # 对话记忆系统
│   ├── base.py             # Memory 基类
│   ├── local_memory.py     # 本地内存实现
│   └── redis_memory.py     # Redis 滑动窗口实现
│
├── config/                 # 配置管理
│   ├── settings.py         # 配置数据类 (dataclass)
│   ├── loader.py           # YAML + ENV 配置加载器
│   ├── settings.yaml       # 基础配置
│   └── settings.development.yaml
│
├── utils/                  # 工具函数
│   ├── logger.py           # 统一日志
│   ├── json_parser.py      # JSON 容错解析
│   └── prompt_loader.py    # Prompt 模板加载
│
├── examples/               # 示例代码
│   ├── chat_example.py     # 基础对话
│   ├── tool_agent_example.py # 工具调用
│   ├── model_switch_example.py # 模型切换
│   └── rag_example.py      # RAG 知识库问答
│
├── prompts/                # Prompt 模板目录
├── main.py                 # FastAPI 入口
├── requirements.txt        # Python 依赖
└── .env.example            # 环境变量模板
```

---

## 快速开始

### 1. 安装依赖

```bash
cd agent_template
pip install -r requirements.txt
```

### 2. 配置

```bash
# 复制环境变量文件
cp .env.example .env

# 编辑 .env，填入你的 API Key
AGENT_LLM_API_KEY=your-api-key
```

或直接修改 `config/settings.yaml`。

### 3. 启动服务

```bash
uvicorn main:app --reload --port 8000
```

### 4. 测试对话

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好", "session_id": "test"}'
```

---

## 模块详解

### 1. LLM 多模型抽象层 (`llm/`)

统一封装不同 LLM Provider，通过工厂模式创建，支持运行时热切换。

#### 支持的 Provider

| Provider | 模型示例 | 配置方式 |
|---------|---------|---------|
| DashScope | qwen-plus, qwen-turbo, qwen-max | `provider: "dashscope"` |
| OpenAI | gpt-4o, gpt-3.5-turbo | `provider: "openai"` |
| DeepSeek | deepseek-chat, deepseek-coder | `provider: "openai"` + `base_url` |
| GLM (智谱) | glm-4, glm-3-turbo | `provider: "openai"` + `base_url` |
| Ollama (本地) | qwen2:7b, llama3 | `provider: "openai"` + `base_url` |
| Moonshot | moonshot-v1-8k | `provider: "openai"` + `base_url` |

#### 使用方式

```python
from llm import LLMFactory

# 通义千问
llm = LLMFactory.create(provider="dashscope", model="qwen-plus", api_key="sk-xxx")

# DeepSeek
llm = LLMFactory.create(
    provider="openai", model="deepseek-chat",
    api_key="sk-xxx", base_url="https://api.deepseek.com/v1"
)

# 本地 Ollama
llm = LLMFactory.create(
    provider="openai", model="qwen2:7b",
    api_key="ollama", base_url="http://localhost:11434/v1"
)

# 调用
from llm.base import ChatMessage
response = await llm.generate([ChatMessage(role="user", content="你好")])
print(response.content)
```

#### 运行时切换模型

```python
agent.llm = LLMFactory.create(provider="openai", model="deepseek-chat", ...)
```

---

### 2. Tools 工具链系统 (`tools/`)

基于 OpenAI Function Calling 标准的工具系统，支持 LLM 自动调用。

#### 自定义工具

```python
from tools.base import Tool, ToolResult

class MyTool(Tool):
    name = "my_tool"
    description = "我的自定义工具"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "查询内容"}
        },
        "required": ["query"]
    }

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        result = f"处理了: {query}"
        return ToolResult(success=True, data=result)
```

#### 注册与使用

```python
from tools import ToolRegistry

registry = ToolRegistry()
registry.register(MyTool())

# 获取 Function Calling Schema (传给 LLM)
schemas = registry.get_function_schemas()

# 从 LLM 响应中执行工具
result = await registry.call_from_llm_response(tool_call)
```

#### 装饰器注册

```python
@registry.register_decorator
class AnotherTool(Tool):
    ...
```

---

### 3. Skills 技能系统 (`skills/`)

可组合的 LLM 能力单元，封装特定任务的 Prompt + 后处理逻辑。

**Tool vs Skill 区别：**
- **Tool**: 执行外部操作（搜索、读文件、执行代码），由 LLM 通过 Function Calling 触发
- **Skill**: 封装 LLM 自身能力（JSON 提取、文档解析、报告生成），由代码直接调用

#### 使用示例

```python
from skills import JsonExtractorSkill

skill = JsonExtractorSkill()
result = await skill.execute(
    llm=llm,
    text='用户说: {"name": "张三", "age": 25}',
    schema={"name": "str", "age": "int"}
)
print(result.data)  # {"name": "张三", "age": 25}
```

#### 自定义 Skill

```python
from skills.base import Skill, SkillResult

class SummarySkill(Skill):
    name = "summary"
    description = "文本摘要"

    async def execute(self, llm, **kwargs) -> SkillResult:
        text = kwargs["text"]
        messages = [ChatMessage(role="user", content=f"请总结: {text}")]
        response = await llm.generate(messages)
        return SkillResult(success=True, data=response.content)
```

---

### 4. Agent 核心层 (`agents/`)

两种 Agent 类型：

#### ChatAgent - 对话型

```python
from agents import ChatAgent

agent = ChatAgent(
    llm=llm,
    memory=memory,              # 可选
    retriever=retriever,        # 可选 (RAG)
    system_prompt="你是助手",
)

reply = await agent.chat("你好", session_id="user_123")
```

#### TaskAgent - 任务型 (ReAct)

自动循环：LLM 生成 → 检测 tool_calls → 执行工具 → 结果传回 LLM → 直到得到最终答案。

```python
from agents import TaskAgent

agent = TaskAgent(
    llm=llm,
    memory=memory,
    tool_registry=registry,     # 注册的工具集
    system_prompt="你是助手",
    max_tool_rounds=5,          # 最大工具调用轮次
)

reply = await agent.chat("帮我搜索 Python 最新版本", session_id="task_1")
```

---

### 5. RAG 检索增强 (`rag/`)

向量化文档 → 语义检索 → 注入上下文 → LLM 回答。

```python
from rag import EmbeddingModel, QdrantVectorStore, Retriever

# 初始化
embedding = EmbeddingModel(provider="dashscope", model="text-embedding-v3", api_key="sk-xxx")
vector_store = QdrantVectorStore(host="localhost", port=6333, collection_name="docs", dimension=1024)
retriever = Retriever(embedding=embedding, vector_store=vector_store, top_k=5)

# 写入文档
vector = await embedding.embed("FastAPI 是一个现代化的 Web 框架")
await vector_store.upsert(text="...", vector=vector, metadata={"source": "docs"})

# 检索
context = await retriever.retrieve("FastAPI 是什么？")

# 传给 Agent
agent = ChatAgent(llm=llm, retriever=retriever, ...)
```

---

### 6. Memory 记忆系统 (`memory/`)

#### LocalMemory - 内存存储

```python
from memory import LocalMemory
memory = LocalMemory(max_history=20)
```

#### RedisMemory - Redis 持久化

```python
from memory import RedisMemory
memory = RedisMemory(
    redis_url="redis://localhost:6379/0",
    max_history=20,
    ttl_seconds=86400,  # 24小时过期
)
```

两者接口完全一致，通过配置切换。

---

### 7. 配置系统 (`config/`)

#### 优先级（从低到高）

1. `settings.py` 中的 dataclass 默认值
2. `config/settings.yaml` 基础配置
3. `config/settings.{env}.yaml` 环境配置
4. `AGENT_` 前缀的环境变量

#### 环境变量映射规则

```
AGENT_LLM_API_KEY     → settings.llm.api_key
AGENT_RAG_QDRANT_HOST → settings.rag.qdrant_host
AGENT_ENV             → 当前环境名称
```

---

### 8. JSON 容错解析 (`utils/json_parser.py`)

从 HR 项目实战中提炼的 LLM 输出 JSON 解析器，5 步容错：

```python
from utils import safe_parse_json

# 能处理以下情况:
safe_parse_json('```json\n{"name": "test"}\n```')  # Markdown 代码块
safe_parse_json('回答: {"name": "test"} 以上')       # JSON 夹杂在文本中
safe_parse_json('{"a": 1, "b": 2,}')                # 尾随逗号
```

---

## 架构设计理念

```
┌─────────────────────────────────────────┐
│              Application                │
│         (main.py / FastAPI)             │
├─────────────────────────────────────────┤
│              Agent Layer                │
│     ChatAgent  /  TaskAgent (ReAct)     │
├──────┬──────┬──────┬──────┬─────────────┤
│ LLM  │Tools │Skills│Memory│    RAG      │
│ 抽象  │工具链 │技能  │记忆  │  检索增强    │
├──────┴──────┴──────┴──────┴─────────────┤
│          Config + Utils                 │
│     配置加载 / 日志 / JSON解析           │
└─────────────────────────────────────────┘
```

**核心思想：**
- **分层解耦**: 每层独立可替换，LLM/Memory/RAG 都可独立切换实现
- **工厂模式**: LLM 通过工厂创建，配置驱动，运行时可热切换
- **注册表模式**: Tool 通过注册中心管理，Agent 自动获取可用工具
- **ReAct 循环**: TaskAgent 实现"思考-行动-观察"循环，自动调用工具
- **容错设计**: JSON 解析等关键环节都有多级 fallback

---

## 扩展指南

### 添加新的 LLM Provider

1. 在 `llm/` 下新建文件，继承 `LLMBase`
2. 实现 `generate()` 和 `stream()` 方法
3. 在 `LLMFactory` 中注册新 provider

### 添加新的 Tool

1. 在 `tools/builtin/` 下新建文件，继承 `Tool`
2. 定义 `name`, `description`, `parameters`
3. 实现 `execute()` 方法
4. 注册到 `ToolRegistry`

### 添加新的 Skill

1. 在 `skills/` 下新建文件，继承 `Skill`
2. 实现 `execute(llm, **kwargs)` 方法

### 添加新的 Memory 后端

1. 在 `memory/` 下新建文件，继承 `MemoryBase`
2. 实现 `add()`, `get_history()`, `clear()` 方法
