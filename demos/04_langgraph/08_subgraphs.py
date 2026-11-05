"""
Demo 04-08 · 子图（Subgraph）：模块化、复用与多团队协作
====================================================
知识点
  1. 方式 A —— 子图直接作为节点：add_node("qc", compiled_subgraph)
     父子 state 中同名的 key 自动传入/传出（子图可以有自己私有的 key）
  2. 方式 B —— 在节点函数里调用子图：父子 state 结构完全不同时，手动做输入/输出映射
  3. 父图的 checkpointer 会自动传给子图；get_state(config, subgraphs=True) 可查看子图状态
  4. stream(..., subgraphs=True)：事件带命名空间，能看到子图内部每一步
  多智能体系统里，每个 Agent 往往就是一个子图（create_agent 的返回值本身就是一个编译好的图）。
本 demo 不调用 LLM。
对应文档：docs/03-详设篇/05-LangGraph运行时.md
运行：uv run demos/04_langgraph/08_subgraphs.py
"""

import operator
from typing import Annotated, NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from langchain_demo.console import banner, section, show


# ======================= 子图 A：质检（与父图共享 draft / issues 两个 key） =======================
class QCState(TypedDict):
    draft: str  # 与父图共享
    issues: Annotated[list[str], operator.add]  # 与父图共享
    word_count: NotRequired[int]  # 子图私有，父图看不到


def count_words(state: QCState) -> dict:
    return {"word_count": len(state["draft"])}


def check_rules(state: QCState) -> dict:
    issues = []
    if state["word_count"] < 10:
        issues.append(f"字数过少（{state['word_count']} 字）")
    if "最" in state["draft"]:
        issues.append("含广告法禁用词“最”")
    return {"issues": issues}


qc_builder = StateGraph(QCState)
qc_builder.add_node("count_words", count_words)
qc_builder.add_node("check_rules", check_rules)
qc_builder.add_edge(START, "count_words")
qc_builder.add_edge("count_words", "check_rules")
qc_subgraph = qc_builder.compile()  # 不需要单独传 checkpointer，会继承父图的


# ======================= 子图 B：翻译（state 结构与父图完全不同） =======================
class TranslateState(TypedDict):
    source_text: str
    target_text: NotRequired[str]


translate_builder = StateGraph(TranslateState)
translate_builder.add_node("translate", lambda s: {"target_text": f"[EN] {s['source_text']}"})
translate_builder.add_edge(START, "translate")
translate_subgraph = translate_builder.compile()


# ======================= 父图 =======================
class ArticleState(TypedDict):
    draft: str
    issues: Annotated[list[str], operator.add]
    english: NotRequired[str]
    status: NotRequired[str]


def call_translate(state: ArticleState) -> dict:
    """方式 B：手动映射 父 state → 子 state → 父 state。"""
    result = translate_subgraph.invoke({"source_text": state["draft"]})
    return {"english": result["target_text"]}


def publish(state: ArticleState) -> dict:
    return {"status": "退回修改" if state["issues"] else "已发布"}


builder = StateGraph(ArticleState)
builder.add_node("qc", qc_subgraph)  # 方式 A：编译好的子图直接当节点
builder.add_node("translate", call_translate)
builder.add_node("publish", publish)
builder.add_edge(START, "qc")
builder.add_edge("qc", "translate")
builder.add_edge("translate", "publish")
builder.add_edge("publish", END)
graph = builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    banner("Demo 04-08 · 子图", "子图即节点 / 节点内调用子图 / subgraphs=True / 子图状态")

    section("1. 父图结构（xray=True 展开子图内部节点）")
    print(graph.get_graph(xray=True).draw_mermaid())

    section("2. stream(subgraphs=True)：命名空间区分父图与子图事件")
    config = {"configurable": {"thread_id": "article-1"}}
    for namespace, chunk in graph.stream({"draft": "全网最好用的笔记软件", "issues": []}, config, stream_mode="updates", subgraphs=True):
        where = "父图" if not namespace else f"子图 {namespace[0].split(':')[0]}"
        show(where, chunk)

    section("3. 结果（注意子图私有的 word_count 不会出现在父图 state 里）")
    show("父图最终 state", graph.get_state(config).values)


if __name__ == "__main__":
    main()
