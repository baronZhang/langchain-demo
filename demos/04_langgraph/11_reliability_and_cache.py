"""
Demo 04-11 · 生产级可靠性：重试 / 缓存 / 超时 / 错误兜底 / 依赖注入 / 可恢复
========================================================================
知识点
  1. RetryPolicy：节点级重试（指数退避、按异常类型过滤）
  2. CachePolicy + compile(cache=InMemoryCache())：节点级结果缓存，相同输入直接复用（可设 ttl）
  3. timeout / TimeoutPolicy：节点级超时（仅异步节点），超时抛 NodeTimeoutError，可再配合重试
  4. error_handler：节点最终失败后的兜底处理，拿到 NodeError(node, error)，返回 Command 决定后续
  5. context_schema + Runtime：给节点注入运行时依赖（配置、客户端），替代全局变量，方便测试
  6. 容错恢复：节点失败时同一超步里已成功的结果会作为 pending writes 保存，修复后 invoke(None) 从断点续跑
本 demo 不调用 LLM，用模拟函数演示机制。
对应文档：docs/03-详设篇/05-LangGraph运行时.md
运行：uv run demos/04_langgraph/11_reliability_and_cache.py
"""

import asyncio
import time
from dataclasses import dataclass
from typing import NotRequired, TypedDict

from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import CachePolicy, Command, RetryPolicy

from langchain_demo.console import banner, section, show


class State(TypedDict):
    query: str
    result: NotRequired[str]
    status: NotRequired[str]


def demo_retry() -> None:
    section("1. RetryPolicy：前两次失败，第三次成功")
    attempts = {"n": 0}

    def flaky_api(state: State) -> dict:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError(f"第 {attempts['n']} 次：上游服务 503")
        return {"result": f"第 {attempts['n']} 次调用成功"}

    builder = StateGraph(State)
    builder.add_node(
        "flaky_api",
        flaky_api,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.1, backoff_factor=2.0, retry_on=ConnectionError),
    )
    builder.add_edge(START, "flaky_api")
    show("结果", builder.compile().invoke({"query": "x"})["result"])


def demo_cache() -> None:
    section("2. CachePolicy：相同输入命中缓存，跳过节点执行")

    def slow_search(state: State) -> dict:
        time.sleep(1.0)  # 模拟昂贵调用（LLM、外部 API）
        return {"result": f"“{state['query']}”的搜索结果"}

    builder = StateGraph(State)
    builder.add_node("slow_search", slow_search, cache_policy=CachePolicy(ttl=60))  # 默认按节点输入计算缓存 key
    builder.add_edge(START, "slow_search")
    graph = builder.compile(cache=InMemoryCache())
    for query in ("LangGraph", "LangGraph", "LangChain"):
        start = time.perf_counter()
        graph.invoke({"query": query})
        show(f"query={query}", f"{time.perf_counter() - start:.2f}s")


def demo_timeout_and_error_handler() -> None:
    section("3. timeout + error_handler：超时后兜底，返回降级结果而不是让整个图失败")

    async def slow_llm(state: State) -> dict:
        await asyncio.sleep(3)  # 模拟卡住的模型调用
        return {"result": "模型回答"}

    def fallback(state: State, error: NodeError) -> Command:
        # 参数里声明 NodeError 类型即可注入失败上下文：哪个节点、什么异常
        return Command(update={"result": "（服务繁忙，已返回缓存答案）", "status": f"{error.node} 失败：{type(error.error).__name__}"})

    builder = StateGraph(State)
    builder.add_node("slow_llm", slow_llm, timeout=0.5, error_handler=fallback)  # timeout 仅对 async 节点生效
    builder.add_edge(START, "slow_llm")
    builder.add_edge("slow_llm", END)
    result = asyncio.run(builder.compile().ainvoke({"query": "x"}))
    show("结果", result)


@dataclass
class Deps:
    """运行时依赖：真实项目里可以是数据库连接、HTTP 客户端、租户配置等。"""

    region: str
    price_table: dict[str, int]


def demo_runtime_context() -> None:
    section("4. context_schema + Runtime：依赖注入，节点不再依赖全局变量")

    def quote(state: State, runtime: Runtime[Deps]) -> dict:
        price = runtime.context.price_table.get(state["query"], 0)
        return {"result": f"[{runtime.context.region}] {state['query']} 报价 {price} 元"}

    builder = StateGraph(State, context_schema=Deps)
    builder.add_node("quote", quote)
    builder.add_edge(START, "quote")
    graph = builder.compile()
    show("华南", graph.invoke({"query": "专业版"}, context=Deps("华南", {"专业版": 2999}))["result"])
    show("海外", graph.invoke({"query": "专业版"}, context=Deps("海外", {"专业版": 399}))["result"])


def demo_resume_after_failure() -> None:
    section("5. 容错恢复：失败后修好问题，invoke(None) 从断点继续，已完成的节点不重跑")
    runs = {"step1": 0}
    broken = {"value": True}

    def step1(state: State) -> dict:
        runs["step1"] += 1
        return {"result": "step1 完成"}

    def step2(state: State) -> dict:
        if broken["value"]:
            raise RuntimeError("下游数据库连接失败")
        return {"status": "step2 完成"}

    builder = StateGraph(State)
    builder.add_node("step1", step1)
    builder.add_node("step2", step2)
    builder.add_edge(START, "step1")
    builder.add_edge("step1", "step2")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "resume-1"}}
    try:
        graph.invoke({"query": "x"}, config)
    except RuntimeError as exc:
        show("第一次运行失败", exc)
    show("当前停在", graph.get_state(config).next)
    broken["value"] = False  # “修复”问题
    show("恢复后结果", graph.invoke(None, config))
    show("step1 执行次数", f"{runs['step1']}（没有重复执行）")


def main() -> None:
    banner("Demo 04-11 · 生产级可靠性", "RetryPolicy / CachePolicy / timeout / error_handler / Runtime / 断点续跑")
    demo_retry()
    demo_cache()
    demo_timeout_and_error_handler()
    demo_runtime_context()
    demo_resume_after_failure()


if __name__ == "__main__":
    main()
