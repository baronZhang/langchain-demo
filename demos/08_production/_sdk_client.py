"""
LangGraph SDK 客户端示例（需先在另一个终端执行 `uv run --group server langgraph dev`）
================================================================================
Agent Server 把图暴露为 REST + SSE API，核心资源：
  - assistant：图的一份“配置实例”（同一个图可以有多个不同配置的 assistant）
  - thread   ：会话（服务端负责持久化 checkpoint）
  - run      ：一次执行（可同步等待、流式、后台运行、定时）
运行：uv run demos/08_production/_sdk_client.py
"""

import asyncio

from langgraph_sdk import get_client


async def main() -> None:
    client = get_client(url="http://127.0.0.1:2024")  # langgraph dev 默认端口
    assistants = await client.assistants.search()
    print("assistants:", [a["graph_id"] for a in assistants])

    thread = await client.threads.create()  # 服务端创建会话，持久化由平台负责
    async for part in client.runs.stream(
        thread["thread_id"],
        "assistant",  # 对应 langgraph.json 里 graphs 的 key
        input={"messages": [{"role": "user", "content": "深圳天气怎么样？"}]},
        stream_mode="updates",
    ):
        print(part.event, str(part.data)[:120])

    state = await client.threads.get_state(thread["thread_id"])
    content = state["values"]["messages"][-1]["content"]  # 走 HTTP 后消息是 dict；content 可能是块列表（含 thinking）
    if isinstance(content, list):
        content = "".join(block.get("text", "") for block in content if block.get("type") == "text")
    print("最终回答:", content)


if __name__ == "__main__":
    asyncio.run(main())
