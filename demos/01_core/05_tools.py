"""
Demo 01-05 · 工具（Tools）与工具调用循环
======================================
知识点
  1. @tool：函数签名 + docstring -> name / description / args_schema
  2. args_schema：用 Pydantic 精细约束参数（枚举、范围、默认值、描述）
  3. bind_tools：把工具 schema 交给模型；模型只“提议”调用，真正执行在我们的代码里
  4. 手写工具调用循环 —— 这就是 Agent 的本质（create_agent 只是把它工程化、图化）
  5. 并行工具调用、tool_choice 强制调用
  6. 工具工程细节：错误处理、content_and_artifact、InjectedToolArg（对模型隐藏的参数）
对应文档：docs/03-详设篇/02-工具与结构化输出.md
运行：uv run demos/01_core/05_tools.py
"""

from typing import Annotated, Literal

from langchain.messages import AnyMessage, HumanMessage
from langchain.tools import InjectedToolArg, ToolException, tool
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show, show_messages
from langchain_demo.tools import calculator, get_weather

llm = get_chat_model()


# ---------- 用 Pydantic 定义参数：比纯类型注解多了枚举、范围、默认值 ----------
class SearchFlightsInput(BaseModel):
    origin: str = Field(description="出发城市")
    destination: str = Field(description="到达城市")
    cabin: Literal["economy", "business"] = Field(default="economy", description="舱位：经济舱或商务舱")
    max_price: int | None = Field(default=None, ge=0, description="最高票价（元），不限则不填")


@tool(args_schema=SearchFlightsInput)
def search_flights(origin: str, destination: str, cabin: str = "economy", max_price: int | None = None) -> str:
    """搜索两个城市之间的航班。"""
    price = 680 if cabin == "economy" else 2380
    if max_price is not None and price > max_price:
        return f"{origin}->{destination} 没有低于 {max_price} 元的{cabin}航班"
    return f"{origin}->{destination} CA1234 08:30 起飞，{cabin} 票价 {price} 元"


# ---------- content_and_artifact + InjectedToolArg ----------
@tool(response_format="content_and_artifact")
def query_orders(
    status: Literal["paid", "shipped", "all"],
    user_id: Annotated[str, InjectedToolArg],  # 对模型隐藏：schema 里没有它，由我们的代码在执行前注入
) -> tuple[str, list[dict]]:
    """查询当前用户的订单。

    Args:
        status: 订单状态过滤：paid 已支付、shipped 已发货、all 全部
    """
    orders = [
        {"id": "A001", "user": user_id, "status": "paid", "amount": 199},
        {"id": "A002", "user": user_id, "status": "shipped", "amount": 59},
    ]
    hits = [o for o in orders if status == "all" or o["status"] == status]
    # 返回 (content, artifact)：content 给模型看（简短），artifact 给程序用（完整数据，不进上下文）
    return f"共 {len(hits)} 个订单，总额 {sum(o['amount'] for o in hits)} 元", hits


def demo_tool_definition() -> None:
    section("1. @tool 自动生成的元数据")
    show("name", get_weather.name)
    show("description", get_weather.description)
    show("args", get_weather.args)
    show("search_flights 发给模型的 schema", search_flights.tool_call_schema.model_json_schema())
    show("query_orders 的 schema（注意没有 user_id）", list(query_orders.tool_call_schema.model_json_schema()["properties"]))
    show("工具也是 Runnable，可以直接 invoke", get_weather.invoke({"city": "深圳"}))


def run_tool_loop(question: str, tools: list[BaseTool], *, user_id: str = "u_1001", max_steps: int = 5) -> list[AnyMessage]:
    """手写 Agent 循环：模型提议 -> 执行工具 -> 结果回填 -> 再问模型，直到模型不再调用工具。"""
    model_with_tools = llm.bind_tools(tools)  # 把工具 schema 随请求一起发给模型
    tools_by_name = {t.name: t for t in tools}
    messages: list[AnyMessage] = [HumanMessage(question)]
    for _ in range(max_steps):
        ai_msg = model_with_tools.invoke(messages)
        messages.append(ai_msg)
        if not ai_msg.tool_calls:  # 没有工具调用 = 模型认为可以直接回答了
            return messages
        for call in ai_msg.tool_calls:  # 一轮里可能有多个调用（并行工具调用）
            if call["name"] == "query_orders":
                call["args"]["user_id"] = user_id  # 注入对模型隐藏的参数（权限相关参数绝不能让模型填）
            # 传入完整的 ToolCall（而不是 args），返回的就是带 tool_call_id 的 ToolMessage
            messages.append(tools_by_name[call["name"]].invoke(call))
    raise RuntimeError(f"超过最大步数 {max_steps}，可能陷入了循环")


def demo_manual_loop() -> None:
    section("2. 手写工具调用循环（并行调用 + 多步推理）")
    messages = run_tool_loop("北京和深圳今天天气如何？两地最高气温相差几度？", [get_weather, calculator])
    show_messages(messages)

    section("3. InjectedToolArg + content_and_artifact")
    messages = run_tool_loop("我已支付的订单有哪些？", [query_orders])
    tool_msg = next(m for m in messages if m.type == "tool")
    show("ToolMessage.content（模型看到的）", tool_msg.content)
    show("ToolMessage.artifact（程序拿到的完整数据）", tool_msg.artifact)
    show("最终回答", messages[-1].text)


def demo_tool_choice() -> None:
    section("4. tool_choice：强制模型调用指定工具")
    forced = llm.bind_tools([search_flights], tool_choice="search_flights")
    ai_msg = forced.invoke("我想下周从深圳飞北京，商务舱，预算 3000 以内")
    show("模型生成的调用", ai_msg.tool_calls)


def demo_tool_error() -> None:
    section("5. 工具错误处理：ToolException + handle_tool_error")

    def divide(a: float, b: float) -> float:
        """计算 a 除以 b。"""
        if b == 0:
            raise ToolException("除数不能为 0，请让用户提供非零除数")
        return a / b

    # StructuredTool.from_function：@tool 之外的另一种构造方式，可以设置更多参数
    safe_divide = StructuredTool.from_function(divide, handle_tool_error=True)
    result = safe_divide.invoke({"type": "tool_call", "id": "c1", "name": "divide", "args": {"a": 1, "b": 0}})
    # 异常被转换成 status="error" 的 ToolMessage 回给模型，模型就能自我纠正，而不是整个程序崩溃
    show("ToolMessage", {"status": result.status, "content": result.content})


def main() -> None:
    banner("Demo 01-05 · 工具与工具调用循环", "@tool / bind_tools / 手写 Agent 循环 / 注入参数 / 错误处理")
    demo_tool_definition()
    demo_manual_loop()
    demo_tool_choice()
    demo_tool_error()


if __name__ == "__main__":
    main()
