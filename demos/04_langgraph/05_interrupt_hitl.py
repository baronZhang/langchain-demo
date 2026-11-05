"""
Demo 04-05 · interrupt()：图级别的人在回路
=========================================
知识点
  1. 在节点里调用 interrupt(payload)：图暂停并持久化，payload 通过 result["__interrupt__"] 交给调用方
  2. Command(resume=value) 恢复：⚠ 节点会从头重新执行，这一次 interrupt() 直接返回 value
     → interrupt 之前的代码要幂等（或把副作用放到 interrupt 之后 / 单独的节点）
  3. 一个节点里多次 interrupt：按调用顺序与 resume 值一一匹配（可做“输入校验循环”）
  4. 常见模式：审批后路由（Command goto）、人工编辑 state、补充信息
  5. 前提：必须有 checkpointer + thread_id
对应文档：docs/03-详设篇/05-LangGraph运行时.md
运行：uv run demos/04_langgraph/05_interrupt_hitl.py
"""

from typing import Literal, NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show

llm = get_chat_model()


# ======================= 1. 审批 + 编辑 + 路由 =======================
class DeployState(TypedDict):
    request: str
    plan: NotRequired[str]
    result: NotRequired[str]


def make_plan(state: DeployState) -> dict:
    return {"plan": llm.invoke(f"针对运维请求“{state['request']}”，给出一句话执行方案").text.strip()}


def human_review(state: DeployState) -> Command[Literal["execute", "cancel"]]:
    # interrupt 的参数可以是任意可序列化对象，通常放“给人看的信息”
    decision = interrupt({"question": "是否批准该方案？可修改 plan 后批准", "plan": state["plan"]})
    if decision["approved"]:
        return Command(goto="execute", update={"plan": decision.get("plan", state["plan"])})
    return Command(goto="cancel")


def build_deploy_graph():
    builder = StateGraph(DeployState)
    builder.add_node("make_plan", make_plan)
    builder.add_node("human_review", human_review)
    builder.add_node("execute", lambda s: {"result": f"✅ 已执行：{s['plan']}"})
    builder.add_node("cancel", lambda s: {"result": "❌ 已取消"})
    builder.add_edge(START, "make_plan")
    builder.add_edge("make_plan", "human_review")
    builder.add_edge("execute", END)
    builder.add_edge("cancel", END)
    return builder.compile(checkpointer=InMemorySaver())


def demo_approval() -> None:
    graph = build_deploy_graph()

    section("1. 运行到 interrupt 暂停")
    config = {"configurable": {"thread_id": "deploy-1"}}
    result = graph.invoke({"request": "把订单服务扩容到 10 个实例"}, config)
    show("中断内容", result["__interrupt__"][0].value)
    show("当前停在", graph.get_state(config).next)

    section("2. 人工修改方案后批准 → 走 execute 分支")
    result = graph.invoke(Command(resume={"approved": True, "plan": "先扩容到 6 个实例，观察 30 分钟再扩到 10 个"}), config)
    show("结果", result["result"])

    section("3. 另一个线程：拒绝 → 走 cancel 分支")
    config = {"configurable": {"thread_id": "deploy-2"}}
    graph.invoke({"request": "删除所有测试环境数据库"}, config)
    show("结果", graph.invoke(Command(resume={"approved": False}), config)["result"])


# ======================= 2. 同一节点多次 interrupt：输入校验循环 =======================
class ProfileState(TypedDict):
    age: NotRequired[int]


def ask_age(state: ProfileState) -> dict:
    question = "请输入你的年龄（整数）"
    while True:
        answer = interrupt(question)  # 第 n 次 interrupt 对应第 n 次 resume 的值
        if isinstance(answer, int) or (isinstance(answer, str) and answer.isdigit()):
            return {"age": int(answer)}
        question = f"“{answer}”不是合法年龄，请重新输入整数"


def demo_validation_loop() -> None:
    section("4. 输入校验循环：非法输入会再次 interrupt")
    builder = StateGraph(ProfileState)
    builder.add_node("ask_age", ask_age)
    builder.add_edge(START, "ask_age")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "profile-1"}}

    result = graph.invoke({}, config)
    for user_input in ["二十八", "28"]:
        show("系统提问", result["__interrupt__"][0].value)
        show("用户输入", user_input)
        result = graph.invoke(Command(resume=user_input), config)
    show("最终 state", result)


def main() -> None:
    banner("Demo 04-05 · interrupt 人在回路", "interrupt() / Command(resume) / 审批路由 / 校验循环")
    demo_approval()
    demo_validation_loop()


if __name__ == "__main__":
    main()
