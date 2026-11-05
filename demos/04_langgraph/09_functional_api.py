"""
Demo 04-09 · Functional API：@entrypoint + @task
==============================================
知识点
  1. 不画图，用普通 Python 控制流（if / for / while）写工作流，同样获得持久化、中断、流式能力
  2. @task：一个可被 checkpoint 的工作单元；调用返回 future（.result() 取值），多个 task 可并行；
     恢复执行时已完成的 task 直接读缓存结果，不会重复调用 LLM
  3. @entrypoint(checkpointer=...)：工作流入口；内部可以 interrupt()；previous 参数拿到上次的返回值
  4. entrypoint.final(value=返回值, save=保存给下次的值)：两者可以不同
  5. 怎么选：Graph API 拓扑显式、可视化好、适合复杂协作；Functional API 写起来像普通函数、改造存量代码成本低
     两者底层是同一个运行时（Pregel），可以混用
对应文档：docs/03-详设篇/05-LangGraph运行时.md
运行：uv run demos/04_langgraph/09_functional_api.py
"""

import time

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show

llm = get_chat_model()
calls = {"write": 0}


@task
def write_poem(topic: str) -> str:
    calls["write"] += 1
    return llm.invoke(f"写一句关于{topic}的七言诗，只输出诗句").text.strip()


@task
def polish(poem: str) -> str:
    return llm.invoke(f"给这句诗起一个 4 字标题，输出“《标题》诗句”：{poem}").text.strip()


@entrypoint(checkpointer=InMemorySaver())
def poem_workflow(topics: list[str]) -> dict:
    # 1) 并行：先把所有 task 发出去（拿到 future），再统一 .result()
    futures = [write_poem(t) for t in topics]
    poems = [f.result() for f in futures]

    # 2) 人工审核：普通的 if 判断就是“条件边”
    approved = interrupt({"question": "是否发布这些诗？", "poems": poems})
    if not approved:
        return {"status": "rejected", "poems": poems}

    # 3) 继续并行润色
    titled = [f.result() for f in [polish(p) for p in poems]]
    return {"status": "published", "poems": titled}


@entrypoint(checkpointer=InMemorySaver())
def accumulator(number: int, *, previous: int | None = None) -> entrypoint.final[str, int]:
    """previous = 同一 thread 上一次 save 的值，实现“函数级短期记忆”。"""
    total = (previous or 0) + number
    return entrypoint.final(value=f"本次 +{number}，累计 {total}", save=total)


def main() -> None:
    banner("Demo 04-09 · Functional API", "@entrypoint / @task / 并行 / interrupt / previous")

    section("1. 运行到 interrupt（三个 write_poem 并行执行）")
    config = {"configurable": {"thread_id": "poem-1"}}
    start = time.perf_counter()
    for chunk in poem_workflow.stream(["春雨", "秋月", "冬雪"], config, stream_mode="updates"):
        show(f"{time.perf_counter() - start:5.1f}s", chunk)

    section("2. 恢复执行：已完成的 write_poem 不会重跑（结果来自 checkpoint）")
    result = poem_workflow.invoke(Command(resume=True), config)
    show("结果", result)
    show("write_poem 实际调用次数", f"{calls['write']}（=3，恢复时没有重复调用 LLM）")

    section("3. previous + entrypoint.final：跨调用累积")
    acc_config = {"configurable": {"thread_id": "acc-1"}}
    for n in (5, 10, 20):
        show(f"invoke({n})", accumulator.invoke(n, acc_config))


if __name__ == "__main__":
    main()
