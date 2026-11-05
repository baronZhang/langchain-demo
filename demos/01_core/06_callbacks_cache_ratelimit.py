"""
Demo 01-06 · 回调、缓存、限流与可观测性
======================================
知识点
  1. Callback：在模型/链/工具的生命周期上挂逻辑（日志、计费、审计）；运行时传入会向子步骤传播
  2. get_usage_metadata_callback：统计一段代码内所有模型调用的 token 用量（按模型分组）
  3. 模型缓存：set_llm_cache(InMemoryCache())，相同请求直接命中（生产可换 Redis/SQLite 缓存）
  4. InMemoryRateLimiter：客户端令牌桶限流，保护供应商配额
  5. LangSmith：只需环境变量即可全链路追踪；@traceable 可以追踪任意 Python 函数
对应文档：docs/04-工程篇/01-脚手架-可观测-部署.md
运行：uv run demos/01_core/06_callbacks_cache_ratelimit.py
"""

import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler, get_usage_metadata_callback
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langsmith import traceable

from langchain_demo import get_chat_model, get_settings
from langchain_demo.console import banner, section, show


class AuditHandler(BaseCallbackHandler):
    """自定义回调：只覆写关心的钩子即可。常见钩子：
    on_chat_model_start / on_llm_new_token / on_llm_end / on_llm_error
    on_chain_start / on_chain_end / on_tool_start / on_tool_end / on_retriever_end ...
    """

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any) -> None:
        print(f"  [audit] 模型开始，输入 {len(messages[0])} 条消息，tags={kwargs.get('tags')}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = response.generations[0][0].message.usage_metadata
        print(f"  [audit] 模型结束，tokens: in={usage['input_tokens']} out={usage['output_tokens']}")

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        if kwargs.get("parent_run_id") is None:  # 只打印最外层链结束，避免刷屏
            print("  [audit] 整条链执行完毕")


def demo_callbacks() -> None:
    section("1. 自定义回调：运行时通过 config 传入，自动传播到链里的每一步")
    chain = ChatPromptTemplate.from_template("用一句话介绍{city}") | get_chat_model(fast=True)
    reply = chain.invoke({"city": "成都"}, config={"callbacks": [AuditHandler()], "tags": ["demo"]})
    show("回答", reply.text)


def demo_usage_tracking() -> None:
    section("2. get_usage_metadata_callback：统计多次调用的 token")
    with get_usage_metadata_callback() as cb:
        get_chat_model(fast=True).invoke("你好")
        get_chat_model(fast=True).invoke("介绍一下你自己，一句话")
    show("按模型汇总的用量", cb.usage_metadata)


def demo_cache() -> None:
    section("3. 模型缓存：第二次相同请求直接命中")
    set_llm_cache(InMemoryCache())  # 全局缓存；缓存 key = 序列化后的模型参数 + 输入
    llm = get_chat_model(fast=True)
    for i in (1, 2):
        start = time.perf_counter()
        reply = llm.invoke("给“智能客服”起一个四字中文名，只输出名字")
        show(f"第 {i} 次", f"{reply.text.strip()}，耗时 {time.perf_counter() - start:.2f}s")
    set_llm_cache(None)  # 用完关掉，避免影响后续 demo


def demo_rate_limiter() -> None:
    section("4. InMemoryRateLimiter：令牌桶限流（每 2 秒 1 个请求）")
    limiter = InMemoryRateLimiter(requests_per_second=0.5, check_every_n_seconds=0.1, max_bucket_size=1)
    llm = get_chat_model(fast=True, rate_limiter=limiter)
    start = time.perf_counter()
    for i in range(3):
        llm.invoke("只回复 ok")
        show(f"第 {i + 1} 次完成", f"{time.perf_counter() - start:.1f}s")


@traceable(name="问答业务函数", run_type="chain")
def answer(question: str) -> str:
    """@traceable：任意函数都能出现在 LangSmith trace 树里，内部的模型调用会自动成为子节点。"""
    return get_chat_model(fast=True).invoke(question).text


def demo_langsmith() -> None:
    section("5. LangSmith 可观测性")
    if get_settings().langsmith_tracing:
        show("已开启追踪", "运行后到 https://smith.langchain.com 查看 trace")
    else:
        print("当前未开启。只需在 .env 设置：")
        print("  LANGSMITH_TRACING=true\n  LANGSMITH_API_KEY=lsv2_...\n  LANGSMITH_PROJECT=langchain-demo")
        print("代码零改动，所有 LangChain/LangGraph 调用（含 token、耗时、输入输出）都会自动上报。")
    show("@traceable 函数照常可用", answer("一句话说明什么是可观测性"))


def main() -> None:
    banner("Demo 01-06 · 回调 / 缓存 / 限流 / 可观测", "Callbacks / Cache / RateLimiter / LangSmith")
    demo_callbacks()
    demo_usage_tracking()
    demo_cache()
    demo_rate_limiter()
    demo_langsmith()


if __name__ == "__main__":
    main()
