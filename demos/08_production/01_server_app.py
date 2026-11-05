"""
Demo 08-01 · 部署入口：LangGraph Server（LangSmith Deployment）/ Studio
====================================================================
生产部署 Agent 的标准路径：
  1. 代码里暴露一个“编译好的图”（create_agent / StateGraph.compile() 的返回值），本文件中的 `agent`
  2. 项目根目录的 langgraph.json 声明图的位置、依赖与环境变量
  3. 本地开发：uv run --group server langgraph dev
     → 启动 Agent Server（内存模式）+ 打开 Studio 可视化调试（看图、单步、改 state、时间旅行）
  4. 上线：langgraph build 打 Docker 镜像自托管，或推到 LangSmith Deployment 托管
     Agent Server 自动提供：REST/SSE 流式 API、线程与持久化、后台任务、定时任务、
     人在回路、双重提交处理、A2A 与 MCP 端点等
  ⚠ 由 Server 托管时不要在代码里传 checkpointer/store，平台会注入自己的持久化层。
客户端调用见 demos/08_production/_sdk_client.py（需先启动 langgraph dev）。
对应文档：docs/04-工程篇/01-脚手架-可观测-部署.md
运行（本地自测，不启动服务）：uv run demos/08_production/01_server_app.py
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, SummarizationMiddleware

from langchain_demo import get_chat_model
from langchain_demo.tools import DEMO_TOOLS

# ↓↓↓ langgraph.json 中 "graphs": {"assistant": "./demos/08_production/01_server_app.py:agent"} 指向这里
agent = create_agent(
    get_chat_model(),
    tools=DEMO_TOOLS,
    system_prompt="你是一个部署在 LangGraph Server 上的中文助手，回答简洁。",
    middleware=[
        SummarizationMiddleware(model=get_chat_model(fast=True), trigger=("tokens", 6000), keep=("messages", 10)),
        ModelCallLimitMiddleware(run_limit=10),  # 生产必备：防止失控循环
    ],
    name="assistant",
)


if __name__ == "__main__":
    from langchain_demo.console import banner, show

    banner("Demo 08-01 · 部署入口自测", "直接调用即将被 LangGraph Server 托管的 agent")
    result = agent.invoke({"messages": [{"role": "user", "content": "现在几点？北京天气如何？"}]})
    show("回答", result["messages"][-1].text)
    print("\n下一步：uv run --group server langgraph dev  （浏览器会打开 Studio）")
