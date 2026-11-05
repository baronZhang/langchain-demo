"""
Demo 04-02 · 条件路由与循环：add_conditional_edges / Command / recursion_limit
=========================================================================
知识点
  1. add_conditional_edges(源节点, 路由函数, path_map)：路由函数读 state 返回下一个节点名
  2. 循环：边可以指回前面的节点（“生成 → 评审 → 不合格重写”就是最经典的反思循环）
  3. Command(goto=..., update=...)：在节点内部同时完成“更新状态 + 决定去向”，不需要单独写路由函数
     节点返回类型注解 Command[Literal[...]] 让编译器知道可能的去向（用于校验与可视化）
  4. recursion_limit：防止死循环的安全阀（默认 25 个超步），超过抛 GraphRecursionError
对应文档：docs/03-详设篇/05-LangGraph运行时.md
运行：uv run demos/04_langgraph/02_conditional_routing.py
"""

from typing import Literal, NotRequired, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show

llm = get_chat_model()
# 反思循环会反复调用模型做“开放式创作 + 挑刺”。实测 glm-5.3 这类推理模型在这种任务上
# 单次可能思考 2 万字、耗时 3 分钟以上，所以循环里用快模型；推理模型留给真正需要深度推理的节点
loop_llm = get_chat_model(fast=True)


# ======================= 1. 反思循环：写 → 评 → 改 =======================
class Review(BaseModel):
    score: int = Field(ge=1, le=10, description="1~10 分")
    feedback: str = Field(description="一句具体的改进意见")


class SloganState(TypedDict):
    product: str
    draft: NotRequired[str]
    score: NotRequired[int]
    feedback: NotRequired[str]
    revisions: NotRequired[int]


def write(state: SloganState) -> dict:
    prompt = f"为产品「{state['product']}」写一句 20 字以内的中文广告语，只输出广告语。"
    if state.get("feedback"):
        prompt += f"\n上一版：{state['draft']}\n评审意见：{state['feedback']}\n请据此改进。"
    return {"draft": loop_llm.invoke(prompt).text.strip(), "revisions": state.get("revisions", 0) + 1}


def review(state: SloganState) -> dict:
    reviewer = loop_llm.with_structured_output(Review)
    result = reviewer.invoke(f"你是挑剔的广告总监，严格评审这句广告语（多数初稿只值 5~7 分）：{state['draft']}")
    return {"score": result.score, "feedback": result.feedback}


def route_after_review(state: SloganState) -> Literal["write", "__end__"]:
    if state["score"] >= 8 or state["revisions"] >= 3:  # 达标或最多改 3 版
        return END
    return "write"


def demo_reflection_loop() -> None:
    section("1. 条件边 + 循环：生成 → 评审 → 不达标则重写")
    builder = StateGraph(SloganState)
    builder.add_node("write", write)
    builder.add_node("review", review)
    builder.add_edge(START, "write")
    builder.add_edge("write", "review")
    builder.add_conditional_edges("review", route_after_review, ["write", END])  # path_map 声明可能的去向
    graph = builder.compile()
    print(graph.get_graph().draw_mermaid())
    for chunk in graph.stream({"product": "一款能自动写周报的 AI 助手"}, stream_mode="updates"):
        for node, update in chunk.items():
            show(node, update)


# ======================= 2. Command：节点内路由 =======================
class TicketState(TypedDict):
    text: str
    category: NotRequired[str]
    reply: NotRequired[str]


class Category(BaseModel):
    category: Literal["billing", "tech", "other"]


def triage(state: TicketState) -> Command[Literal["billing", "tech", "other"]]:
    category = llm.with_structured_output(Category).invoke(f"对客服工单分类（billing 账单/tech 技术/other 其他）：{state['text']}").category
    return Command(goto=category, update={"category": category})  # 更新 + 跳转 一步完成


def demo_command_routing() -> None:
    section("2. Command(goto, update)：不需要 add_conditional_edges")
    builder = StateGraph(TicketState)
    builder.add_node("triage", triage)
    builder.add_node("billing", lambda s: {"reply": "已转财务组，24 小时内处理退款/发票问题"})
    builder.add_node("tech", lambda s: {"reply": "已转技术支持，工程师将远程协助排查"})
    builder.add_node("other", lambda s: {"reply": "已转人工客服"})
    builder.add_edge(START, "triage")
    graph = builder.compile()
    for text in ["上个月被重复扣费了两次", "APP 登录一直提示网络错误"]:
        result = graph.invoke({"text": text})
        show(text, f"{result['category']} → {result['reply']}")


def demo_recursion_limit() -> None:
    section("3. recursion_limit：死循环保护")
    builder = StateGraph(TicketState)
    builder.add_node("ping", lambda s: {})
    builder.add_node("pong", lambda s: {})
    builder.add_edge(START, "ping")
    builder.add_edge("ping", "pong")
    builder.add_edge("pong", "ping")  # 故意制造无出口的循环
    try:
        builder.compile().invoke({"text": "loop"}, {"recursion_limit": 6})
    except GraphRecursionError as exc:
        show("捕获异常", f"{type(exc).__name__}: {str(exc)[:80]}…")


def main() -> None:
    banner("Demo 04-02 · 条件路由与循环", "conditional edges / 反思循环 / Command / recursion_limit")
    demo_reflection_loop()
    demo_command_routing()
    demo_recursion_limit()


if __name__ == "__main__":
    main()
