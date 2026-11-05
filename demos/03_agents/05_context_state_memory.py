"""
Demo 03-05 · 上下文工程：Runtime Context / 自定义 State / 短期与长期记忆
=====================================================================
三种“数据”要分清（面试高频）
  ┌──────────────┬───────────────────────────┬──────────────┬────────────────────────┐
  │              │ 是什么                    │ 生命周期     │ 典型内容               │
  ├──────────────┼───────────────────────────┼──────────────┼────────────────────────┤
  │ Context      │ 调用方传入的只读运行时参数│ 单次调用     │ user_id、权限、租户、DB连接│
  │ State        │ 图的可变状态（含 messages）│ 单个 thread │ 对话历史、购物车、中间结果│
  │ Store        │ 跨 thread 的长期记忆      │ 永久         │ 用户偏好、画像、知识    │
  └──────────────┴───────────────────────────┴──────────────┴────────────────────────┘
知识点
  1. context_schema + invoke(..., context=...)：上下文不进对话历史、模型看不到，安全传递身份信息
  2. ToolRuntime：工具里拿到 state / context / store / stream_writer / tool_call_id
  3. 工具返回 Command(update=...)：工具直接写 Agent 的 state
  4. state_schema：在 messages 之外加业务字段
  5. checkpointer（短期记忆，按 thread_id）+ store（长期记忆，按 namespace）
对应文档：docs/03-详设篇/04-Agent与中间件.md
运行：uv run demos/03_agents/05_context_state_memory.py
"""

import operator
import uuid
from dataclasses import dataclass
from typing import Annotated

from langchain.agents import AgentState, create_agent
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show, show_messages


@dataclass
class UserContext:
    """运行时上下文：由业务系统在调用时注入（比如从登录态里取），模型无法篡改。"""

    user_id: str
    user_name: str
    vip_level: int = 0


class ShopState(AgentState):
    """在默认的 messages 之外，增加一个购物车字段。

    ⚠ 必须带 reducer：模型可能在一轮里并行调用两次 add_to_cart，两个工具在同一超步写 cart，
    没有 reducer（默认 LastValue）会抛 InvalidUpdateError: Can receive only one value per step。
    """

    cart: Annotated[list[str], operator.add]


@tool
def get_my_discount(runtime: ToolRuntime[UserContext]) -> str:
    """查询当前登录用户可享受的折扣。"""
    ctx = runtime.context  # 名为 runtime、类型为 ToolRuntime 的参数会被自动注入，不会出现在工具 schema 里
    discount = {0: "无折扣", 1: "95 折", 2: "88 折"}.get(ctx.vip_level, "8 折")
    return f"{ctx.user_name}（VIP{ctx.vip_level}）可享受 {discount}"


@tool
def add_to_cart(item: str, runtime: ToolRuntime[UserContext, ShopState]) -> Command:
    """把商品加入购物车。

    Args:
        item: 商品名称
    """
    already = len(runtime.state.get("cart", []))  # runtime.state 是本超步开始时的 state 快照（只读）
    # 返回 Command 可以直接更新 state；此时必须自己补上 ToolMessage，保证每个 tool_call 都有结果
    return Command(
        update={
            "cart": [item],  # 只返回增量，由 operator.add 追加
            "messages": [ToolMessage(f"已加入「{item}」（加入前购物车有 {already} 件）", tool_call_id=runtime.tool_call_id)],
        }
    )


@tool
def save_preference(preference: str, runtime: ToolRuntime[UserContext]) -> str:
    """记住用户的长期偏好（跨会话有效）。

    Args:
        preference: 用户偏好的描述，例如“喜欢黑色的电子产品”
    """
    namespace = ("users", runtime.context.user_id, "preferences")  # namespace 类似目录，用于隔离不同用户
    runtime.store.put(namespace, str(uuid.uuid4()), {"text": preference})
    return f"已记住：{preference}"


@tool
def recall_preferences(runtime: ToolRuntime[UserContext]) -> str:
    """读取用户的长期偏好。"""
    items = runtime.store.search(("users", runtime.context.user_id, "preferences"))
    return "；".join(item.value["text"] for item in items) or "暂无偏好记录"


def main() -> None:
    banner("Demo 03-05 · 上下文工程", "Context / State / Store / ToolRuntime / Command")
    store = InMemoryStore()  # 长期记忆；生产换 PostgresStore / RedisStore 等
    agent = create_agent(
        get_chat_model(),
        tools=[get_my_discount, add_to_cart, save_preference, recall_preferences],
        system_prompt="你是电商购物助手。用户提到喜好时要用 save_preference 记住；做推荐前先用 recall_preferences 查偏好。回答简洁。",
        state_schema=ShopState,
        context_schema=UserContext,
        checkpointer=InMemorySaver(),
        store=store,
    )
    ctx = UserContext(user_id="u_1001", user_name="小王", vip_level=2)

    section("1. 会话 A：context 注入身份 + 工具用 Command 写 state + 写长期记忆")
    thread_a = {"configurable": {"thread_id": "session-A"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我有什么折扣？帮我把机械键盘和 27 寸显示器加入购物车。顺便记住：我喜欢黑色的电子产品。"}]},
        thread_a,
        context=ctx,
    )
    show_messages(result["messages"], max_chars=100)
    show("state['cart']", result.get("cart"))

    section("2. 会话 B（新 thread，短期记忆为空）：通过 store 召回长期记忆")
    thread_b = {"configurable": {"thread_id": "session-B"}}
    result = agent.invoke({"messages": [{"role": "user", "content": "帮我推荐一款鼠标，颜色怎么选？"}]}, thread_b, context=ctx)
    show("回答", result["messages"][-1].text)
    show("会话 B 的购物车（与会话 A 隔离）", result.get("cart"))

    section("3. 直接查看 store 里的长期记忆")
    for item in store.search(("users", "u_1001", "preferences")):
        show(f"{item.namespace} / {item.key[:8]}", item.value)


if __name__ == "__main__":
    main()
