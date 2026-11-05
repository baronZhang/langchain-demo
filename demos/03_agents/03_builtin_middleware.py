"""
Demo 03-03 · 内置中间件（Middleware）巡礼
======================================
中间件是 LangChain 1.x Agent 最核心的扩展机制：在不改 Agent 主循环的前提下，
在“调用模型前后 / 调用工具前后”插入横切逻辑（类似 Web 框架的中间件、Spring 的 AOP）。

本 demo 覆盖的内置中间件（按用途分组）
  上下文管理：SummarizationMiddleware（历史摘要压缩）、ContextEditingMiddleware（清理旧工具结果）
  成本与安全：ModelCallLimitMiddleware / ToolCallLimitMiddleware（调用次数熔断）、PIIMiddleware（敏感信息脱敏）
  可靠性    ：ModelFallbackMiddleware（模型降级）、ToolRetryMiddleware（工具重试）、ToolErrorMiddleware（工具异常转消息）
  能力增强  ：LLMToolSelectorMiddleware（工具太多时先筛选）、TodoListMiddleware（任务规划清单）
未演示但值得知道：HumanInTheLoopMiddleware（见 03-06）、ModelRetryMiddleware、LLMToolEmulator、
  ShellToolMiddleware、FilesystemFileSearchMiddleware、ProviderToolSearchMiddleware
对应文档：docs/03-详设篇/04-Agent与中间件.md
运行：uv run demos/03_agents/03_builtin_middleware.py
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    LLMToolSelectorMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
    ToolCallRequest,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
    wrap_model_call,
)
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain_demo import get_chat_model
from langchain_demo.console import banner, clip, section, show, show_messages
from langchain_demo.tools import DEMO_TOOLS, calculator, get_current_time, get_weather

llm = get_chat_model()
fast_llm = get_chat_model(fast=True)


def ask(agent, text: str, config: dict | None = None) -> dict:
    return agent.invoke({"messages": [{"role": "user", "content": text}]}, config)


def demo_summarization() -> None:
    section("1. SummarizationMiddleware：历史超过阈值自动摘要（用快模型做摘要省钱）")
    agent = create_agent(
        llm,
        checkpointer=InMemorySaver(),
        middleware=[SummarizationMiddleware(model=fast_llm, trigger=("messages", 6), keep=("messages", 2))],
    )
    config = {"configurable": {"thread_id": "summary"}}
    for text in ["我叫小李，是一名 Java 后端", "我最近在学 LangGraph", "我下周要面试 AI 应用工程师", "帮我总结一下我的情况"]:
        result = ask(agent, text + "（回答 30 字以内）", config)
        show(f"消息数 {len(result['messages']):>2}", clip(result["messages"][-1].text, 60))
    show("第一条消息已被替换为摘要", clip(result["messages"][0].text, 150))


def demo_context_editing() -> None:
    section("2. ContextEditingMiddleware：只保留最近 N 个工具结果，旧的替换成占位符")
    agent = create_agent(
        llm,
        tools=[get_weather],
        middleware=[ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=50, keep=1)])],  # trigger 单位是 token
    )
    result = ask(agent, "依次查询北京、上海、深圳的天气，每次只查一个城市，最后汇总")
    # 注意：编辑只作用于“发给模型的请求”，state 里的完整历史不受影响
    show("最终回答", clip(result["messages"][-1].text, 120))


def demo_call_limits() -> None:
    section("3. ModelCallLimit / ToolCallLimit：防止 Agent 死循环烧钱")
    agent = create_agent(
        llm,
        tools=[get_weather],
        middleware=[
            ModelCallLimitMiddleware(run_limit=3, exit_behavior="end"),  # 单次运行最多调用 3 次模型
            ToolCallLimitMiddleware(tool_name="get_weather", run_limit=2),  # 天气工具单次运行最多 2 次，超出的调用被拦截
        ],
    )
    result = ask(agent, "依次查询北京、上海、深圳、杭州、成都的天气，每次只查一个城市")
    show("工具实际执行次数", sum(1 for m in result["messages"] if m.type == "tool" and m.status != "error"))
    show("被拦截的调用", [clip(m.text, 60) for m in result["messages"] if m.type == "tool" and m.status == "error"])
    show("最后一条消息", clip(result["messages"][-1].text, 100))


def demo_pii() -> None:
    section("4. PIIMiddleware：内置类型 + 自定义正则，四种策略 block / redact / mask / hash")
    agent = create_agent(
        llm,
        middleware=[
            PIIMiddleware("email", strategy="redact"),  # 内置：email / credit_card / ip / mac_address / url
            PIIMiddleware("cn_phone", detector=r"1[3-9]\d{9}", strategy="mask"),  # 自定义：中国大陆手机号
        ],
    )
    result = ask(agent, "我的邮箱是 zhangsan@example.com，手机 13812345678，帮我写一句自我介绍")
    show("进入模型前的用户消息（已脱敏）", result["messages"][0].text)
    show("回答", clip(result["messages"][-1].text, 100))


def demo_reliability() -> None:
    section("5. ModelFallbackMiddleware：主模型失败自动切换备用模型")
    broken = get_chat_model("model-that-does-not-exist", max_retries=0)
    agent = create_agent(broken, middleware=[ModelFallbackMiddleware(fast_llm)])
    result = ask(agent, "只回复：降级成功")
    show("回答", f"{result['messages'][-1].text}（实际模型 {result['messages'][-1].response_metadata.get('model_name')}）")

    section("6. ToolRetryMiddleware + ToolErrorMiddleware：工具重试与异常转消息")
    calls = {"n": 0}

    @tool
    def flaky_stock_price(symbol: str) -> str:
        """查询股票实时价格。"""
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("行情服务超时")
        return f"{symbol} 当前价格 168.8 元"

    @tool
    def risky_transfer(amount: int) -> str:
        """内部转账接口（演示用，必定失败）。"""
        raise ValueError(f"账户余额不足，无法转出 {amount} 元")

    def on_tool_error(exc: Exception, request: ToolCallRequest) -> str | None:
        # 只把“业务异常”告诉模型；其他未知异常返回 None 继续抛出，避免内部细节泄露给模型/用户
        if isinstance(exc, ValueError):
            return f"工具 {request.tool_call['name']} 执行失败：{exc}"
        return None

    agent = create_agent(
        llm,
        tools=[flaky_stock_price, risky_transfer],
        middleware=[
            ToolErrorMiddleware(on_error=on_tool_error),  # 外层：兜底把异常变成 ToolMessage(status="error")
            ToolRetryMiddleware(max_retries=2, tools=["flaky_stock_price"], initial_delay=0.2, on_failure="error"),
        ],
    )
    result = ask(agent, "查一下 600519 的股价，然后帮我转账 100 万元")
    show("flaky_stock_price 实际被调用次数", calls["n"])
    show_messages(result["messages"][1:], max_chars=120)


def demo_tool_selector_and_todo() -> None:
    section("7. LLMToolSelectorMiddleware：工具很多时，先用快模型挑出相关工具")

    @wrap_model_call
    def log_visible_tools(request, handler):
        # 放在筛选中间件之后（更内层），所以能看到筛选后的工具列表
        print("  主模型本次可见的工具:", [getattr(t, "name", t) for t in request.tools])
        return handler(request)

    @tool
    def translate(text: str, target: str) -> str:
        """把文本翻译成目标语言。"""
        return f"[{target}] {text}"

    @tool
    def search_news(keyword: str) -> str:
        """搜索新闻。"""
        return f"关于“{keyword}”的新闻：……"

    agent = create_agent(
        llm,
        tools=[*DEMO_TOOLS, translate, search_news],
        middleware=[LLMToolSelectorMiddleware(model=fast_llm, max_tools=2), log_visible_tools],
    )
    result = ask(agent, "深圳今天天气怎么样？")
    show("回答", clip(result["messages"][-1].text, 80))

    section("8. TodoListMiddleware：给 Agent 一个 write_todos 工具做任务规划")
    agent = create_agent(llm, tools=[get_weather, calculator, get_current_time], middleware=[TodoListMiddleware()])
    result = ask(agent, "帮我准备明天去北京出差：查天气、算一下 3 晚住宿按每晚 600 元一共多少钱、告诉我现在几点。请先列计划再执行。")
    show("最终 todos（存在 state['todos'] 里）", result.get("todos"))
    show("回答", clip(result["messages"][-1].text, 150))


def main() -> None:
    banner("Demo 03-03 · 内置中间件", "摘要 / 上下文编辑 / 限流熔断 / PII / 降级 / 重试 / 工具筛选 / 规划")
    demo_summarization()
    demo_context_editing()
    demo_call_limits()
    demo_pii()
    demo_reliability()
    demo_tool_selector_and_todo()


if __name__ == "__main__":
    main()
