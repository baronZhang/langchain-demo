"""
Demo 04-01 · StateGraph 基础：State / Node / Edge / Reducer
==========================================================
知识点
  1. 图 = State（共享状态）+ Node（函数：读 state，返回“增量”）+ Edge（控制流）
  2. Reducer：决定节点返回的增量如何合并进 state
     - 默认（无注解）：后写覆盖
     - Annotated[list, operator.add]：列表追加
     - add_messages：按消息 id 追加/替换/删除（RemoveMessage）
     - 自定义 reducer 函数；Overwrite(...) 可以绕过 reducer 强制覆盖
  3. 执行模型（Pregel / BSP）：按“超步（super-step）”推进，同一超步内的多个节点并行执行，
     超步结束时统一合并增量 —— 所以并行节点写同一个 key 必须有 reducer，否则报 InvalidUpdateError
  4. input_schema / output_schema：对外只暴露必要的输入输出字段
  5. 可视化：get_graph().draw_mermaid()
本 demo 不调用 LLM：LangGraph 本身与 LLM 无关，是一个通用的“有状态、可持久化的工作流运行时”。
对应文档：docs/03-详设篇/05-LangGraph运行时.md
运行：uv run demos/04_langgraph/01_state_graph_basics.py
"""

import operator
from typing import Annotated, TypedDict

from langchain.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Overwrite

from langchain_demo.console import banner, section, show


def merge_scores(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """自定义 reducer：签名固定为 (旧值, 新值) -> 合并后的值。"""
    return {**left, **right}


class State(TypedDict):
    topic: str  # 无 reducer：后写覆盖
    logs: Annotated[list[str], operator.add]  # 追加
    scores: Annotated[dict[str, int], merge_scores]  # 自定义合并
    total: int


class InputState(TypedDict):
    topic: str


class OutputState(TypedDict):
    logs: list[str]
    total: int


def plan(state: State) -> dict:
    return {"logs": [f"plan：确定选题《{state['topic']}》"]}  # 只返回要更新的 key


def review_a(state: State) -> dict:
    return {"logs": ["review_a：打 8 分"], "scores": {"a": 8}}


def review_b(state: State) -> dict:
    return {"logs": ["review_b：打 6 分"], "scores": {"b": 6}}


def summarize(state: State) -> dict:
    total = sum(state["scores"].values())
    return {"logs": [f"summarize：总分 {total}"], "total": total}


def build_graph():
    builder = StateGraph(State, input_schema=InputState, output_schema=OutputState)
    builder.add_node("plan", plan)
    builder.add_node("review_a", review_a)
    builder.add_node("review_b", review_b)
    builder.add_node("summarize", summarize)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "review_a")  # 扇出：plan 之后 review_a、review_b 在同一超步并行
    builder.add_edge("plan", "review_b")
    builder.add_edge(["review_a", "review_b"], "summarize")  # 扇入：等两个都完成才执行 summarize
    builder.add_edge("summarize", END)
    return builder.compile()


def main() -> None:
    banner("Demo 04-01 · StateGraph 基础", "State / Node / Edge / Reducer / 超步 / 输入输出 schema")
    graph = build_graph()

    section("1. 图结构（Mermaid）")
    print(graph.get_graph().draw_mermaid())

    section("2. invoke：输入只需 topic（input_schema），输出只有 logs + total（output_schema）")
    show("结果", graph.invoke({"topic": "LangGraph 入门"}))

    section("3. stream_mode='updates'：观察超步 —— review_a 与 review_b 在同一步执行")
    for step, chunk in enumerate(graph.stream({"topic": "LangGraph 入门"}, stream_mode="updates"), 1):
        show(f"第 {step} 次推送", list(chunk))

    section("4. Overwrite：绕过 reducer，直接覆盖")
    builder = StateGraph(State)
    builder.add_node("append", lambda s: {"logs": ["新日志"]})
    builder.add_node("reset", lambda s: {"logs": Overwrite(["日志已重置"])})
    builder.add_edge(START, "append")
    builder.add_edge("append", "reset")
    show("结果", builder.compile().invoke({"topic": "x", "logs": ["旧日志"], "scores": {}, "total": 0})["logs"])

    section("5. add_messages：消息列表专用 reducer（MessagesState 内置）")
    msgs = add_messages([HumanMessage("你好", id="1")], [AIMessage("你好！有什么可以帮你？", id="2")])
    show("追加", [m.text for m in msgs])
    msgs = add_messages(msgs, [HumanMessage("你好（已编辑）", id="1")])  # 相同 id → 替换
    show("同 id 替换", [m.text for m in msgs])
    msgs = add_messages(msgs, [RemoveMessage(id="2")])  # RemoveMessage → 删除
    show("删除", [m.text for m in msgs])


if __name__ == "__main__":
    main()
