"""
Demo 05-02 · 多智能体 ②：Handoffs（转接）模式
===========================================
知识点
  1. 每个 Agent 是父图里的一个节点（create_agent 返回的图直接当子图节点用）
  2. Agent 通过“转接工具”把对话控制权交给另一个 Agent：
     工具返回 Command(goto="目标节点", graph=Command.PARENT, update=...) —— 从子图内部跳到父图的另一个节点
  3. 共享对话历史：接手的 Agent 能看到完整上下文（与 supervisor 模式的“上下文隔离”正好相反）
  4. 用 state 里的 active_agent 记住“当前是谁在服务”，下一轮用户消息直接路由给它
  适合：客服分诊（售前 ↔ 售后）、需要专家直接与用户多轮交互的场景
对应文档：docs/03-详设篇/06-多智能体-DeepAgents-MCP.md
运行：uv run demos/05_multi_agent/02_handoffs.py
"""

from typing import NotRequired

from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.types import Command

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show

llm = get_chat_model()


def make_handoff_tool(target: str, label: str, description: str):
    """生成一个“转接给 target”的工具。"""

    @tool(f"transfer_to_{target}", description=description)
    def handoff(runtime: ToolRuntime) -> Command:
        # 这条回执会被接手的 Agent 看到：要明确告诉它“现在轮到你直接服务用户了”，否则它可能误以为自己还是转出方
        done = ToolMessage(f"【系统】对话已转接给{label}，由{label}直接处理用户的诉求。", tool_call_id=runtime.tool_call_id)
        return Command(
            goto=target,
            graph=Command.PARENT,  # 关键：跳转发生在父图，而不是当前 Agent 子图内部
            # 把本 Agent 的消息（含发起转接的 AIMessage）和工具回执一并交给父图，保证历史完整、tool_call 闭合
            update={"messages": [*runtime.state["messages"], done], "active_agent": target},
        )

    return handoff


@tool
def get_product_price(product: str) -> str:
    """查询商品价格与库存。"""
    return f"{product}：现价 1299 元，库存充足，支持 7 天无理由退货"


@tool
def create_refund(order_id: str, reason: str) -> str:
    """为订单创建退款单。"""
    return f"退款单 RF-{order_id[-4:]} 已创建，原因：{reason}，预计 3 个工作日原路退回"


sales = create_agent(
    llm,
    tools=[get_product_price, make_handoff_tool("after_sales", "售后专员", "用户咨询退货、退款、维修等售后问题时，转接给售后专员")],
    system_prompt=(
        "你是售前顾问，只负责产品咨询和推荐（查价格用 get_product_price）。遇到售后问题必须立刻转接，不要自己处理。"
        "如果你是被转接过来的，直接回答用户最新的购买咨询。回答简洁。"
    ),
    name="sales",
)
after_sales = create_agent(
    llm,
    tools=[create_refund, make_handoff_tool("sales", "售前顾问", "用户想购买或咨询新产品时，转接给售前顾问")],
    system_prompt=(
        "你是售后专员，负责退货退款：用户要退货时直接调用 create_refund 创建退款单。用户有购买新品的需求时转接给售前顾问。"
        "如果你是被转接过来的，直接处理用户最新的售后诉求。回答简洁。"
    ),
    name="after_sales",
)


class SupportState(MessagesState):
    active_agent: NotRequired[str]


def route_to_active(state: SupportState) -> str:
    return state.get("active_agent", "sales")  # 新会话默认由售前接待


builder = StateGraph(SupportState)
builder.add_node("sales", sales, destinations=("after_sales",))  # destinations 只用于画图
builder.add_node("after_sales", after_sales, destinations=("sales",))
builder.add_conditional_edges(START, route_to_active, ["sales", "after_sales"])
graph = builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    banner("Demo 05-02 · Handoffs 转接", "Command(graph=Command.PARENT) / 共享历史 / active_agent 路由")
    print(graph.get_graph().draw_mermaid())
    config = {"configurable": {"thread_id": "customer-1"}}
    turns = [
        "你好，降噪耳机 X1 多少钱？",
        "我上周买的订单 SN20261020 的耳机有电流声，想退货",
        "退款知道了。那你们有没有新款音箱推荐？",
    ]
    for i, text in enumerate(turns, 1):
        section(f"第 {i} 轮：{text}")
        for chunk in graph.stream({"messages": [{"role": "user", "content": text}]}, config, stream_mode="updates"):
            for node in chunk:
                print(f"  → 由 {node} 处理")
        state = graph.get_state(config).values
        show("当前负责人", state.get("active_agent", "sales"))
        show("回复", state["messages"][-1].text)


if __name__ == "__main__":
    main()
