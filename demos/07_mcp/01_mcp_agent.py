"""
Demo 07-01 · MCP：用 langchain.mcp.MCPAdapter 把 MCP Server 接进 Agent
====================================================================
知识点
  1. MCP 三要素：tools / resources / prompts（服务端代码见 _shop_server.py）
  2. MCPAdapter(target) 支持多种连接目标：
       - FastMCP 实例             → 进程内直连（单测最方便）
       - Path("server.py")        → 本地脚本，stdio 子进程
       - "https://host/mcp"       → 远程 Streamable HTTP
       - MCPConfig / dict         → 多个 server（与 Claude Desktop 的 mcpServers 配置格式一致）
     ⚠ 字符串只接受 http(s) URL：防止来自配置/模型的字符串被当成本地脚本执行（安全设计）
  3. await adapter.list_tools() → LangChain 工具（异步），直接传给 create_agent，用 ainvoke 调用
  4. Elicitation：MCP 工具执行中途向用户索取输入 → 自动变成 LangGraph interrupt → Command(resume=...) 回答
     注意 MCP 2026-07-28 版协议改为无状态多轮往返（SEP-2322）：服务端返回 InputRequiredResult，
     客户端带答案重新调用；旧的 `await ctx.elicit()` 在新协议连接上会报错（服务端写法见 _shop_server.py）
  5. 新旧对照：langchain 1.4 起 `langchain.mcp`（beta，基于 FastMCP，pip install "langchain[mcp]"）
     取代了 langchain-mcp-adapters 的 MultiServerMCPClient（已不再维护）。旧写法：
         client = MultiServerMCPClient({"shop": {"command": "python", "args": ["server.py"], "transport": "stdio"}})
         tools = await client.get_tools()
对应文档：docs/03-详设篇/06-多智能体-DeepAgents-MCP.md
运行：uv run demos/07_mcp/01_mcp_agent.py
"""

import asyncio
import sys
import warnings
from pathlib import Path

from langchain_core._api import LangChainBetaWarning

warnings.filterwarnings("ignore", category=LangChainBetaWarning)  # langchain.mcp 处于 beta，导入时会告警一次

from _shop_server import mcp as shop_server  # noqa: E402  同目录的 MCP Server（进程内连接用）
from langchain.agents import create_agent  # noqa: E402
from langchain.mcp import MCPAdapter  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from langchain_demo import get_chat_model  # noqa: E402
from langchain_demo.console import banner, section, show, show_messages  # noqa: E402

SERVER_SCRIPT = Path(__file__).with_name("_shop_server.py")
llm = get_chat_model()


async def demo_discovery() -> None:
    section("1. 进程内连接：发现 MCP Server 暴露的工具 / 资源 / 提示词")
    async with MCPAdapter(shop_server) as adapter:
        tools = await adapter.list_tools()
        for t in tools:
            show(f"工具 {t.name}", f"{t.description.splitlines()[0]} | 参数 {list(t.args)}")
        client = adapter.client  # 底层的 fastmcp.Client，可以做资源、提示词等高级操作
        show("资源", [str(r.uri) for r in await client.list_resources()])
        show("读取资源 shop://policy/refund", (await client.read_resource("shop://policy/refund"))[0].text)
        prompt = await client.get_prompt("customer_service", {"tone": "专业"})
        show("提示词 customer_service", prompt.messages[0].content.text)


async def demo_agent_stdio() -> None:
    section("2. stdio 子进程 + create_agent：Agent 调用 MCP 工具")
    async with MCPAdapter(SERVER_SCRIPT) as adapter:  # Path → 以子进程方式启动服务端脚本
        agent = create_agent(llm, await adapter.list_tools(), system_prompt="你是星河商城客服，涉及订单先查询再回答，回答简洁。")
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "我的订单 SN001 到哪了？"}]})  # MCP 工具是异步的
        show_messages(result["messages"], max_chars=200)


async def demo_elicitation() -> None:
    section("3. Elicitation → interrupt：MCP 工具中途请求用户确认")
    async with MCPAdapter(SERVER_SCRIPT) as adapter:
        agent = create_agent(
            llm,
            await adapter.list_tools(),
            system_prompt="用户要求取消订单时直接调用 cancel_order。该工具会自行向用户二次确认，你不需要先询问用户。",
            checkpointer=InMemorySaver(),  # interrupt 需要 checkpointer
        )
        config = {"configurable": {"thread_id": "mcp-elicit"}}
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "帮我取消订单 SN002"}]}, config)
        if "__interrupt__" not in result:
            show("模型没有调用 cancel_order", result["messages"][-1].text)
            return
        payload = result["__interrupt__"][0].value  # {"type": "mcp_elicitation", "tool_name": ..., "requests": [...]}
        request = payload["requests"][0]
        show("中断类型 / 工具", f"{payload['type']} / {payload['tool_name']}")
        show("服务端的问题", request["message"])
        show("要求的答案 schema", request["requested_schema"])

        # 回答格式：按 request 的 key 返回 accept / decline / cancel；accept 时 content 要符合 requested_schema
        fields = list(request["requested_schema"].get("properties", {}))
        content = {fields[0]: True} if fields else None
        resume = {"responses": {request["key"]: {"action": "accept", "content": content}}}
        result = await agent.ainvoke(Command(resume=resume), config)
        show("工具结果", next(m.text for m in reversed(result["messages"]) if m.type == "tool"))
        show("最终回答", result["messages"][-1].text)


async def demo_multi_server() -> None:
    section("4. 多个 Server：MCPConfig（与 Claude Desktop 的 mcpServers 格式一致）")
    config = {
        "mcpServers": {
            "shop": {"command": sys.executable, "args": [str(SERVER_SCRIPT)]},
            "backup_shop": {"command": sys.executable, "args": [str(SERVER_SCRIPT)]},
        }
    }
    tools = await MCPAdapter(config).list_tools()
    show("合并后的工具（多 server 时自动加前缀避免重名）", [t.name for t in tools])


async def main() -> None:
    banner("Demo 07-01 · MCP", "MCPAdapter：进程内 / stdio / 多 server / elicitation → interrupt")
    await demo_discovery()
    await demo_agent_stdio()
    await demo_elicitation()
    await demo_multi_server()


if __name__ == "__main__":
    asyncio.run(main())
