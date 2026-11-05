"""
Demo 05-03 · 多智能体 ③：Router（路由分发）模式
============================================
知识点
  1. 路由器用结构化输出把问题拆成若干“领域子问题”（可能涉及多个领域）
  2. Send 并行分发给领域专家（每个专家只检索自己领域的文档），reducer 收集结果
  3. 汇总节点把多位专家的回答合成一份最终答案
  4. 与 Supervisor 的区别：Router 是“一次性决策”的工作流（可预测、低延迟）；
     Supervisor 是在循环里“动态决策”的 Agent（更灵活、更贵）
对应文档：docs/03-详设篇/06-多智能体-DeepAgents-MCP.md
运行：uv run demos/05_multi_agent/03_router.py
"""

import operator
from typing import Annotated, Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show
from langchain_demo.rag import build_vector_store, format_docs

llm = get_chat_model()
store = build_vector_store()
Domain = Literal["hr", "finance", "it"]
DOMAIN_DOCS = {"hr": "employee_handbook.md", "finance": "expense_policy.md", "it": "it_security.md"}
DOMAIN_NAMES = {"hr": "人事专家", "finance": "财务专家", "it": "IT 安全专家"}


class SubQuestion(BaseModel):
    domain: Domain = Field(description="hr=考勤/假期/远程办公；finance=报销/差旅；it=账号/数据/AI 工具规范")
    question: str = Field(description="只属于该领域的、可独立回答的子问题")


class RoutePlan(BaseModel):
    sub_questions: list[SubQuestion]


class RouterState(TypedDict):
    question: str
    plan: NotRequired[list[SubQuestion]]
    answers: Annotated[list[str], operator.add]
    final: NotRequired[str]


class ExpertInput(TypedDict):
    domain: Domain
    question: str


def route(state: RouterState) -> dict:
    plan = llm.with_structured_output(RoutePlan).invoke(f"把员工的问题按领域拆分成子问题：{state['question']}")
    return {"plan": plan.sub_questions}


def dispatch(state: RouterState) -> list[Send]:
    return [Send("expert", {"domain": q.domain, "question": q.question}) for q in state["plan"]]


def expert(state: ExpertInput) -> dict:
    # 领域隔离：只在本领域文档里检索
    docs = store.similarity_search(state["question"], k=2, filter=lambda d: d.metadata["source"] == DOMAIN_DOCS[state["domain"]])
    answer = llm.invoke(f"你是{DOMAIN_NAMES[state['domain']]}，只依据资料用一两句话回答。\n资料：\n{format_docs(docs)}\n问题：{state['question']}")
    return {"answers": [f"【{DOMAIN_NAMES[state['domain']]}】{answer.text.strip()}"]}


def synthesize(state: RouterState) -> dict:
    final = llm.invoke(f"把以下专家意见整合成给员工的一段完整回答（100 字以内）：\n" + "\n".join(state["answers"]))
    return {"final": final.text.strip()}


builder = StateGraph(RouterState)
builder.add_node("route", route)
builder.add_node("expert", expert)
builder.add_node("synthesize", synthesize)
builder.add_edge(START, "route")
builder.add_conditional_edges("route", dispatch, ["expert"])
builder.add_edge("expert", "synthesize")
builder.add_edge("synthesize", END)
graph = builder.compile()


def main() -> None:
    banner("Demo 05-03 · Router 路由分发", "结构化拆分 → Send 并行专家 → 汇总")
    question = "我下周想远程办公两天，期间要去杭州出差住两晚，住宿能报多少？出差时能用公共 AI 工具处理客户数据吗？"
    result = graph.invoke({"question": question, "answers": []})
    section("1. 路由计划")
    for q in result["plan"]:
        show(q.domain, q.question)
    section("2. 各专家回答（并行）")
    for answer in result["answers"]:
        print("  " + answer)
    section("3. 汇总答案")
    print(result["final"])


if __name__ == "__main__":
    main()
