"""
Demo 04-06 · 流式输出全解
========================
知识点
  1. stream_mode 一览
     values   每个超步结束后的完整 state
     updates  每个节点返回的增量 {节点名: 增量}
     messages LLM 逐 token：(消息块, 元数据)，无论 LLM 在图里哪个节点、哪层子图被调用
     custom   节点/工具里用 get_stream_writer() 主动推送的任意数据（进度条、中间结果）
     debug / tasks / checkpoints：调试与审计用的详细事件
  2. 多模式同时订阅：stream_mode=["updates", "custom"] → 产出 (mode, chunk) 元组
  3. version="v2"（LangGraph 1.2）：统一产出 {"type", "ns", "data"} 字典，类型友好，前端协议更清晰
  4. 异步版本 astream；更细粒度的事件流 astream_events
对应文档：docs/03-详设篇/05-LangGraph运行时.md
运行：uv run demos/04_langgraph/06_streaming.py
"""

import asyncio
import time
from typing import NotRequired, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show

llm = get_chat_model()


class State(TypedDict):
    topic: str
    facts: NotRequired[list[str]]
    article: NotRequired[str]


def research(state: State) -> dict:
    writer = get_stream_writer()  # 拿到当前运行的 custom 流写入器
    facts = []
    for i, source in enumerate(["百科", "新闻", "论文"], 1):
        time.sleep(0.2)  # 模拟检索耗时
        facts.append(f"{state['topic']}在{source}中的资料")
        writer({"progress": f"{i}/3", "source": source})  # 推送进度给前端
    return {"facts": facts}


def write(state: State) -> dict:
    article = llm.invoke(f"基于这些资料，用两句话介绍{state['topic']}：{state['facts']}")
    return {"article": article.text}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("research", research)
    builder.add_node("write", write)
    builder.add_edge(START, "research")
    builder.add_edge("research", "write")
    builder.add_edge("write", END)
    return builder.compile()


graph = build_graph()
INPUT = {"topic": "大熊猫"}


def demo_values_and_updates() -> None:
    section("1. values：每步后的完整 state")
    for state in graph.stream(INPUT, stream_mode="values"):
        show("state keys", list(state))

    section("2. updates：每个节点的增量")
    for chunk in graph.stream(INPUT, stream_mode="updates"):
        for node, update in chunk.items():
            show(node, {k: (v[:40] + "…" if isinstance(v, str) and len(v) > 40 else v) for k, v in update.items()})


def demo_multi_mode() -> None:
    section("3. 多模式：custom（进度）+ messages（逐 token）")
    for mode, chunk in graph.stream(INPUT, stream_mode=["custom", "messages"]):
        if mode == "custom":
            print(f"\n  [进度] {chunk}")
        elif mode == "messages":
            token, metadata = chunk
            if token.text:
                print(token.text, end="", flush=True)  # metadata["langgraph_node"] == "write"
    print()


def demo_v2_format() -> None:
    section("4. version='v2'：统一的 StreamPart（type / ns / data）")
    for part in graph.stream(INPUT, stream_mode=["updates", "custom"], version="v2"):
        # ns 是命名空间：根图为 ()，子图事件形如 ("子图节点:task_id",)
        show(f"type={part['type']} ns={part['ns']}", list(part["data"]) if part["type"] == "updates" else part["data"])


async def demo_async() -> None:
    section("5. astream：异步流式（Web 服务里最常用，配合 SSE / WebSocket 推给前端）")
    count = 0
    async for token, _ in graph.astream(INPUT, stream_mode="messages"):
        count += 1
    show("收到的消息块数量", count)


def main() -> None:
    banner("Demo 04-06 · 流式输出", "values / updates / messages / custom / v2 / astream")
    demo_values_and_updates()
    demo_multi_mode()
    demo_v2_format()
    asyncio.run(demo_async())


if __name__ == "__main__":
    main()
