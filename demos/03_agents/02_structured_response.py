"""
Demo 03-02 · Agent 结构化输出：response_format
=============================================
知识点
  1. response_format=Schema：Agent 跑完工具循环后，把最终结果按 schema 输出到 result["structured_response"]
  2. 三种策略
     - ToolStrategy    ：把 schema 伪装成一个“输出工具”，模型调用它即视为给出最终答案；任何支持工具调用的模型都能用
     - ProviderStrategy：使用厂商原生结构化输出（约束解码，可靠性最高），但必须厂商支持
     - AutoStrategy    ：直接传 schema 时的默认值；按模型 profile 自动选，未知模型回落到 ToolStrategy
  3. ToolStrategy 支持 Union（多选一的输出类型）和 handle_errors（校验失败自动让模型重试）
对应文档：docs/03-详设篇/04-Agent与中间件.md
运行：uv run demos/03_agents/02_structured_response.py
"""

from typing import Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from pydantic import BaseModel, Field

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show
from langchain_demo.tools import get_weather

llm = get_chat_model()


class WeatherReport(BaseModel):
    """天气播报"""

    city: str
    condition: str = Field(description="天气状况")
    temp_high: int = Field(description="最高气温（℃）")
    temp_low: int = Field(description="最低气温（℃）")
    advice: str = Field(description="一句出行建议")


class Refund(BaseModel):
    """退款申请"""

    kind: Literal["refund"] = "refund"
    order_id: str = Field(description="订单号")
    reason: str = Field(description="退款原因")


class Complaint(BaseModel):
    """投诉"""

    kind: Literal["complaint"] = "complaint"
    target: str = Field(description="投诉对象")
    severity: int = Field(ge=1, le=5, description="严重程度 1~5")


def demo_auto_strategy() -> None:
    section("1. 直接传 schema（AutoStrategy）：工具循环 + 结构化最终答案")
    agent = create_agent(llm, tools=[get_weather], response_format=WeatherReport)
    result = agent.invoke({"messages": [{"role": "user", "content": "查一下杭州天气，并给出出行建议"}]})
    show("structured_response", result["structured_response"].model_dump())
    show("消息序列（最后的 tool 消息就是“输出工具”的回执）", [m.type for m in result["messages"]])


def demo_tool_strategy_union() -> None:
    section("2. ToolStrategy + Union：让模型在多种输出类型中选一种")
    agent = create_agent(
        llm,
        tools=[],
        system_prompt="你是客服工单分类器，把用户诉求转成退款申请或投诉。",
        response_format=ToolStrategy(Refund | Complaint, handle_errors=True),  # 校验失败时把错误回传给模型重试
    )
    for text in ["订单 SN20261027 的耳机有杂音，我要退款", "你们的快递员态度极差，还把我的包裹扔在门口！"]:
        result = agent.invoke({"messages": [{"role": "user", "content": text}]})
        resp = result["structured_response"]
        show(type(resp).__name__, resp.model_dump())


def demo_provider_strategy() -> None:
    section("3. ProviderStrategy：依赖厂商原生结构化输出")
    agent = create_agent(llm, tools=[], response_format=ProviderStrategy(WeatherReport))
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": "编一份北京晴天 10~20℃ 的天气播报"}]})
        show("structured_response", result["structured_response"].model_dump())
    except Exception as exc:  # noqa: BLE001 —— 演示用：展示不支持时的失败形态
        show("失败", f"{type(exc).__name__}: {str(exc)[:120]}…")
        print("结论：兼容端点往往不支持原生结构化输出，这正是 AutoStrategy 对未知模型默认选 ToolStrategy 的原因。")


def main() -> None:
    banner("Demo 03-02 · Agent 结构化输出", "response_format: AutoStrategy / ToolStrategy / ProviderStrategy")
    demo_auto_strategy()
    demo_tool_strategy_union()
    demo_provider_strategy()


if __name__ == "__main__":
    main()
