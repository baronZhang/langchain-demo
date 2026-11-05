"""
Demo 03-04 · 自定义中间件：Agent 的 AOP 扩展点
=============================================
知识点
  1. 两类钩子
     - 节点式（编译后是图里的独立节点，可修改 state、可跳转）：
         before_agent → [ before_model → (模型) → after_model → (工具) ]* → after_agent
     - 包裹式（像洋葱一样包住真正的调用，可改请求/改响应/重试/短路）：
         wrap_model_call(request, handler)、wrap_tool_call(request, handler)
     - 便捷装饰器 @dynamic_prompt：本质是一个只改 system prompt 的 wrap_model_call
  2. 两种写法：装饰器（单个钩子，轻量）与 AgentMiddleware 子类（多钩子 + 自定义 state + 自带工具）
  3. 执行顺序：before_* 按列表顺序，after_* 逆序；wrap_* 列表第一个在最外层
  4. 控制流：返回 {"jump_to": "end"} 提前结束，需用 @hook_config(can_jump_to=[...]) 声明
  5. 动态模型选择：在 wrap_model_call 里 request.override(model=...)
对应文档：docs/03-详设篇/04-Agent与中间件.md
运行：uv run demos/03_agents/04_custom_middleware.py
"""

import time
from dataclasses import dataclass
from typing import Any, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    after_model,
    before_model,
    dynamic_prompt,
    hook_config,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.messages import AIMessage
from langgraph.runtime import Runtime

from langchain_demo import get_chat_model
from langchain_demo.console import banner, clip, section, show
from langchain_demo.tools import get_weather

llm = get_chat_model()
fast_llm = get_chat_model(fast=True)


def ask(agent, text: str, **kwargs: Any) -> dict:
    return agent.invoke({"messages": [{"role": "user", "content": text}]}, **kwargs)


# ======================= 写法一：装饰器 =======================
@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict | None:
    print(f"  [before_model] 即将调用模型，当前消息数 {len(state['messages'])}")
    return None  # 返回 None = 不修改 state；返回 dict = 作为 state 增量合并


@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> dict | None:
    last = state["messages"][-1]
    print(f"  [after_model] 模型返回：{'工具调用 ' + str([c['name'] for c in last.tool_calls]) if last.tool_calls else '最终回答'}")
    return None


@wrap_model_call
def route_model(request: ModelRequest, handler) -> ModelResponse:
    """动态模型选择：消息少（简单问题）用快模型，多了（复杂多步）切主模型。"""
    chosen = fast_llm if len(request.messages) <= 2 else llm
    print(f"  [wrap_model_call] 消息数 {len(request.messages)} → 使用 {chosen.model}")
    return handler(request.override(model=chosen))  # override 返回新的 request，不修改原对象


@wrap_tool_call
def time_tool(request: ToolCallRequest, handler):
    start = time.perf_counter()
    result = handler(request)  # handler 真正执行工具；可以在前后加日志、缓存、权限校验、重试
    print(f"  [wrap_tool_call] {request.tool_call['name']}({request.tool_call['args']}) 耗时 {time.perf_counter() - start:.3f}s")
    return result


@dataclass
class UserContext:
    role: str


@dynamic_prompt
def role_based_prompt(request: ModelRequest) -> str:
    """根据运行时上下文（谁在调用）动态生成 system prompt。"""
    role = request.runtime.context.role
    rule = "可以透露内部营收数据：本季度营收 1.2 亿元。" if role == "admin" else "严禁透露任何内部营收数据。"
    return f"你是公司助手，当前用户角色是 {role}。{rule}回答 30 字以内。"


def demo_decorators() -> None:
    section("1. 装饰器式：before_model / after_model / wrap_model_call / wrap_tool_call")
    agent = create_agent(llm, tools=[get_weather], middleware=[log_before_model, log_after_model, route_model, time_tool])
    result = ask(agent, "北京和上海今天天气怎么样？")
    show("回答", clip(result["messages"][-1].text, 100))

    section("2. @dynamic_prompt + context_schema：同一个 Agent，不同角色不同行为")
    agent = create_agent(llm, middleware=[role_based_prompt], context_schema=UserContext)
    for role in ("admin", "guest"):
        result = ask(agent, "本季度营收多少？", context=UserContext(role=role))
        show(role, result["messages"][-1].text)


# ======================= 写法二：AgentMiddleware 子类 =======================
class GuardState(AgentState):
    """中间件可以扩展 Agent 的 state：这里多一个计数字段。"""

    blocked_count: NotRequired[int]


class SensitiveWordGuard(AgentMiddleware[GuardState]):
    """输入护栏：命中敏感词直接结束，不调用模型（省钱又安全）。"""

    state_schema = GuardState  # 声明扩展的 state，create_agent 会自动合并进图的 state

    def __init__(self, words: list[str]) -> None:
        super().__init__()
        self.words = words

    @hook_config(can_jump_to=["end"])  # 声明可能跳转的目标，编译时会生成对应的条件边
    def before_agent(self, state: GuardState, runtime: Runtime) -> dict | None:
        text = state["messages"][-1].text
        if any(word in text for word in self.words):
            return {
                "messages": [AIMessage("抱歉，这个问题我无法回答。")],
                "blocked_count": state.get("blocked_count", 0) + 1,
                "jump_to": "end",
            }
        return None


class Tracer(AgentMiddleware):
    """用来观察多个中间件的执行顺序。"""

    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag

    @property
    def name(self) -> str:  # 同一个类实例化多次时，name 必须不同
        return f"Tracer{self.tag}"

    def before_model(self, state, runtime):
        print(f"  {self.tag}.before_model")

    def after_model(self, state, runtime):
        print(f"  {self.tag}.after_model")

    def wrap_model_call(self, request, handler):
        print(f"  {self.tag}.wrap_model_call ↓")
        response = handler(request)
        print(f"  {self.tag}.wrap_model_call ↑")
        return response


def demo_class_middleware() -> None:
    section("3. 类式中间件：自定义 state + before_agent 跳转")
    agent = create_agent(llm, middleware=[SensitiveWordGuard(["炸药", "赌博"])])
    result = ask(agent, "怎么在网上赌博赢钱？")
    show("回答（未调用模型）", result["messages"][-1].text)
    show("blocked_count", result.get("blocked_count"))

    section("4. 执行顺序：middleware=[A, B, C]")
    agent = create_agent(fast_llm, middleware=[Tracer("A"), Tracer("B"), Tracer("C")])
    ask(agent, "你好，回复两个字")
    print("  结论：before_* 正序 A→B→C；after_* 逆序 C→B→A；wrap_* A 在最外层（洋葱模型）")


def main() -> None:
    banner("Demo 03-04 · 自定义中间件", "装饰器 / 类 / 自定义 state / 跳转 / 动态模型 / 动态 prompt / 执行顺序")
    demo_decorators()
    demo_class_middleware()


if __name__ == "__main__":
    main()
