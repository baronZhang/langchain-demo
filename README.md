# LangChain Demo

33 runnable demos covering the LangChain ecosystem (LangChain 1.4, LangChain Core 1.7, LangGraph 1.2, Deep Agents 0.7, and MCP), plus a small scaffold you can reuse for new projects.

> **Learning project.** It is not a production system. Code comments are in Chinese. Many mention `docs/`, which holds private study notes and is not in this repository.

[中文说明](README.zh-CN.md)

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is managed by uv.

```bash
uv sync                                  # install dependencies
cp .env.example .env                     # then fill in your model endpoint and key
uv run langchain-demo                    # print the config and test model connectivity
uv run demos/01_core/01_chat_model.py    # run one demo
uv run demos/run_all.py --fast -j 6      # run all demos with the fast model, 6 at a time
uv run --group dev pytest -q             # offline tests, no model calls
```

## Model configuration

Everything is set in `.env`. See `.env.example` for all options.

| Variable | Tested value | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | Protocol: `anthropic` or `openai` |
| `LLM_MODEL` | `glm-latest` | Main model for reasoning and agent loops |
| `LLM_FAST_MODEL` | `deepseek-v4-flash` | Fast model for summaries, routing, and evaluation |
| `EMBEDDING_PROVIDER` | `local` | Built-in hash vectors, or any OpenAI-compatible embedding service |

The demos were tested through Volcano Ark's Anthropic-compatible endpoint. No other provider has been tested.

## Demos

| Chapter | Count | Topics |
|---|---|---|
| Core | 6 | Chat models, messages and prompts, LCEL, structured output, tools, callbacks, caching, rate limits |
| RAG | 3 | Indexing, two-step RAG with citations, query rewriting, agentic RAG |
| Agents | 6 | `create_agent`, structured responses, built-in and custom middleware, context and memory, human in the loop |
| LangGraph | 11 | State graphs, routing, a ReAct agent from scratch, persistence and time travel, interrupts, streaming, `Send`, subgraphs, Functional API, long-term memory, retries and caching |
| Multi-agent | 3 | Supervisor with subagents, handoffs, router |
| Deep Agents | 1 | Backends, `AGENTS.md` memory, `SKILL.md` skills, subagents, planning |
| MCP | 1 | `langchain.mcp.MCPAdapter`, stdio and in-process servers, elicitation mapped to interrupts |
| Production | 2 | LangGraph Server deployment, offline evaluation with rules and an LLM judge |

## Test status

| Check | Result | Last run |
|---|---|---|
| `uv run --group dev pytest -q` | 6 passed (offline, fake model) | 2026-11-04 |
| All 33 demos against a real model | Passed | 2026-10 |

The full demo run calls a paid model, so it was not repeated on 2026-11-04.

## Deployment and Studio

```bash
uv run --group server langgraph dev        # local Agent Server on :2024, opens Studio
uv run demos/08_production/_sdk_client.py  # call it with the SDK from another terminal
```

## Layout

```text
src/langchain_demo/   shared code: config, model factory, embeddings, RAG components, sample tools
demos/                33 demos by chapter, plus the run_all.py batch runner
tests/                offline tests with a fake model
data/knowledge/       sample RAG documents (a fictional company's policies)
langgraph.json        LangGraph Server and Studio entry point
```

## Notes

- A reasoning model can spend minutes on open-ended writing. Reflection loops and evaluation judges use the fast model for this reason.
- `langchain-community` and `langchain-mcp-adapters` are no longer maintained, so this project uses neither.
- Runtime output (SQLite, vector indexes, agent workspaces, logs) goes to `.runtime/` and can be deleted at any time.
- This project was developed with Claude Code. The results above come from real test runs.
