# langchain-demo

[English](README.md)

LangChain 生态（**langchain 1.4 / langchain-core 1.7 / LangGraph 1.2 / Deep Agents 0.7 / MCP**）的教学 demo，同时也是可直接复用的项目脚手架。代码注释是中文。

- 33 个可独立运行的 demo，覆盖模型、LCEL、工具、结构化输出、RAG、create_agent、中间件、LangGraph 全部核心机制、多智能体、Deep Agents、MCP、部署和评估
- 所有 demo 已在 2026-10 用最新版依赖实测通过
- 项目用 Claude Code 开发，测试结果都来自真实运行

## 快速开始

```bash
uv sync                                  # 安装依赖（Python 3.12，由 uv 管理）
cp .env.example .env                     # 然后填入模型地址和 key
uv run langchain-demo                    # 环境自检：打印配置，测试主模型和快模型的连通性
uv run demos/01_core/01_chat_model.py    # 运行单个 demo
uv run demos/run_all.py --fast -j 6      # 全量冒烟（快模型，6 个并发）
uv run --group dev pytest -q             # 离线单测（不调用模型）
```

### 模型配置

实测走的是火山方舟的 **Anthropic 兼容协议**，其他服务没有测过：

| 变量 | 值 | 说明 |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | 改成 `openai` 时用 OpenAI 兼容端点 |
| `LLM_MODEL` | `glm-latest` | 主模型，推理模型，**思考模式关不掉**，单次 3~30 秒 |
| `LLM_FAST_MODEL` | `deepseek-v4-flash` | 快模型，约 2 秒，用于摘要、路由、评估、循环 |
| `EMBEDDING_PROVIDER` | `local` | 默认用本地哈希向量；可以改成任意 OpenAI 兼容服务 |

换别的服务只改 `.env`，参考 `.env.example`。`.env` 已被 gitignore。

## 目录

```text
src/langchain_demo/   脚手架公共模块：配置、模型工厂、向量模型、RAG 组件、示例工具
demos/                33 个 demo（按章节编号）+ run_all.py 批量运行器
tests/                离线单测（Fake 模型）
data/knowledge/       RAG 示例语料（虚构的公司制度）
langgraph.json        LangGraph Server / Studio 入口
```

## Demo 清单

| 章节 | Demo | 主题 |
|---|---|---|
| **01 核心** | `01_chat_model` | init_chat_model、invoke / batch / stream / async、content_blocks、可配置模型 |
| | `02_messages_and_prompts` | 消息类型、多模态块、ChatPromptTemplate、few-shot、trim_messages |
| | `03_lcel_runnables` | LCEL 管道、并发、路由、with_retry / with_fallbacks、astream_events |
| | `04_structured_output` | with_structured_output（Pydantic / TypedDict / JSON Schema）、include_raw、OutputParser |
| | `05_tools` | @tool、args_schema、手写工具循环、InjectedToolArg、content_and_artifact、错误处理 |
| | `06_callbacks_cache_ratelimit` | 回调、token 统计、缓存、限流、LangSmith |
| **02 RAG** | `01_indexing` | 加载 → 切分 → 向量化 → 入库 → MMR / 过滤 / 持久化 |
| | `02_rag_chain` | 两段式 RAG、引用来源、查询改写、拒答、流式 |
| | `03_agentic_rag` | 检索即工具（多跳问题）、@dynamic_prompt 注入上下文 |
| **03 Agent** | `01_create_agent_basics` | create_agent、流式、多轮记忆 |
| | `02_structured_response` | response_format：Auto / Tool / Provider 三种策略 |
| | `03_builtin_middleware` | 摘要、上下文编辑、限次、PII、降级、重试、工具异常、工具筛选、规划 |
| | `04_custom_middleware` | 6 种钩子、装饰器与类、跳转、动态模型、动态 prompt、执行顺序 |
| | `05_context_state_memory` | Context / State / Store、ToolRuntime、Command |
| | `06_human_in_the_loop` | HITL：approve / edit / reject |
| **04 LangGraph** | `01_state_graph_basics` | State / Node / Edge / Reducer / 超步 / Overwrite |
| | `02_conditional_routing` | 条件边、反思循环、Command 路由、recursion_limit |
| | `03_react_agent_from_scratch` | MessagesState + ToolNode + tools_condition |
| | `04_persistence_time_travel` | SqliteSaver、历史快照、重放、分叉 |
| | `05_interrupt_hitl` | interrupt()、审批路由、校验循环 |
| | `06_streaming` | values / updates / messages / custom / v2 / astream |
| | `07_send_map_reduce` | Send 动态并行（编排者-工作者） |
| | `08_subgraphs` | 子图的两种用法、subgraphs=True |
| | `09_functional_api` | @entrypoint / @task / previous |
| | `10_store_long_term_memory` | Store 语义记忆、跨线程召回 |
| | `11_reliability_and_cache` | RetryPolicy、CachePolicy、timeout、error_handler、依赖注入、断点续跑 |
| **05 多智能体** | `01_supervisor_subagents` | 子 Agent 即工具（官方首推） |
| | `02_handoffs` | Command.PARENT 转接 |
| | `03_router` | 结构化拆分 + Send 并行专家 |
| **06 Deep Agents** | `01_deep_agent` | 后端、AGENTS.md 记忆、SKILL.md 技能、子代理、规划 |
| **07 MCP** | `01_mcp_agent` | langchain.mcp.MCPAdapter、stdio / 进程内 / 多 server、elicitation 转 interrupt |
| **08 工程化** | `01_server_app` | LangGraph Server 部署入口（配合 `langgraph dev` / Studio） |
| | `02_evaluation` | 离线评估：规则加 LLM-as-judge |

## 部署与 Studio

```bash
uv run --group server langgraph dev      # 启动本地 Agent Server（:2024），并打开 Studio
uv run demos/08_production/_sdk_client.py  # 另开一个终端，用 SDK 调用
```

## 说明

- 推理模型在开放式创作任务上可能思考数分钟。反思循环、评估裁判这类节点在 demo 里用快模型，原因写在对应的注释里；
- `langchain-community` 和 `langchain-mcp-adapters` 已停止维护，本项目都没有使用：RAG 用自实现的 `BaseLoader`，MCP 用 `langchain[mcp]`；
- 运行产物（SQLite、向量索引、Deep Agent 工作区、日志）都在 `.runtime/`，可以随时删除；
- 很多注释提到 `docs/`，那是我的私人学习笔记，不在这个仓库里。
