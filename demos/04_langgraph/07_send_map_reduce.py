"""
Demo 04-07 · Send API：运行时动态并行（Map-Reduce / Orchestrator-Worker）
=====================================================================
知识点
  1. 普通边的并行度在“画图时”就固定了；Send 让并行度在“运行时”由数据决定
  2. 条件边返回 [Send("节点名", 私有输入), ...]：每个 Send 生成一个并行任务，输入可以不是完整 state
  3. worker 的返回值通过 reducer（operator.add）聚合回主 state
  4. 典型场景：多文档并行摘要、子问题并行检索、批量评测、深度研究的多路调研
对应文档：docs/03-详设篇/05-LangGraph运行时.md
运行：uv run demos/04_langgraph/07_send_map_reduce.py
"""

import operator
import time
from typing import Annotated, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show

llm = get_chat_model()


class Outline(BaseModel):
    sections: list[str] = Field(description="3 个互不重叠的小节标题")


class ReportState(TypedDict):
    topic: str
    sections: NotRequired[list[str]]
    drafts: Annotated[list[str], operator.add]  # 多个 worker 并发写入 → 必须有 reducer
    report: NotRequired[str]


class WorkerInput(TypedDict):
    """worker 的私有输入：只包含它需要的字段。"""

    topic: str
    section: str


def orchestrator(state: ReportState) -> dict:
    outline = llm.with_structured_output(Outline).invoke(f"为《{state['topic']}》设计一份简短报告的 3 个小节标题")
    return {"sections": outline.sections}


def assign_workers(state: ReportState) -> list[Send]:
    # 返回几个 Send，就并行执行几次 write_section —— 数量由上一步的结果决定
    return [Send("write_section", {"topic": state["topic"], "section": s}) for s in state["sections"]]


def write_section(state: WorkerInput) -> dict:
    text = llm.invoke(f"报告主题：{state['topic']}。请用 2 句话写小节「{state['section']}」，只输出正文").text.strip()
    return {"drafts": [f"## {state['section']}\n{text}"]}


def synthesize(state: ReportState) -> dict:
    return {"report": f"# {state['topic']}\n\n" + "\n\n".join(state["drafts"])}


def main() -> None:
    banner("Demo 04-07 · Send API", "orchestrator → N 个并行 worker → reducer 聚合")
    builder = StateGraph(ReportState)
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("write_section", write_section)
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", assign_workers, ["write_section"])
    builder.add_edge("write_section", "synthesize")  # 所有并行 worker 完成后才进入 synthesize
    builder.add_edge("synthesize", END)
    graph = builder.compile()

    section("1. 图结构")
    print(graph.get_graph().draw_mermaid())

    section("2. 运行（观察 write_section 的多个实例在同一超步完成）")
    start = time.perf_counter()
    for chunk in graph.stream({"topic": "大模型应用工程师的核心技能", "drafts": []}, stream_mode="updates"):
        for node in chunk:
            show(f"{time.perf_counter() - start:5.1f}s", node)
    final = graph.invoke({"topic": "大模型应用工程师的核心技能", "drafts": []})

    section("3. 最终报告")
    print(final["report"])


if __name__ == "__main__":
    main()
