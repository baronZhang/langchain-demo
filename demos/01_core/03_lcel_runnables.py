"""
Demo 01-03 · LCEL 与 Runnable 协议
=================================
知识点
  1. 一切皆 Runnable：prompt / model / parser / retriever / 甚至整个 LangGraph 图，接口都一样
  2. `|` 管道 = RunnableSequence；dict 字面量 = RunnableParallel（并发执行）
  3. RunnablePassthrough.assign / RunnableLambda / @chain 嵌入任意 Python 逻辑
  4. 路由：RunnableBranch，或“返回 Runnable 的函数”（动态路由，更推荐）
  5. 韧性：with_retry（重试）/ with_fallbacks（降级）
  6. 观测：astream_events 看清每个子步骤；get_graph 输出链的结构图
对应文档：docs/03-详设篇/01-模型-消息-提示词-LCEL.md
运行：uv run demos/01_core/03_lcel_runnables.py
"""

import asyncio
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough, chain

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show

llm = get_chat_model()
fast_llm = get_chat_model(fast=True)
to_str = StrOutputParser()  # AIMessage -> str，流式时也能逐块透传


def demo_sequence() -> None:
    section("1. prompt | model | parser：最小的 LCEL 链")
    prompt = ChatPromptTemplate.from_template("用一句话解释：{concept}")
    explain = prompt | llm | to_str
    show("链的类型", type(explain).__name__)
    show("invoke", explain.invoke({"concept": "向量数据库"}))
    print("stream: ", end="")
    for piece in explain.stream({"concept": "知识图谱"}):  # 解析器支持流式，字符串一块块到达
        print(piece, end="", flush=True)
    print()


def demo_parallel_and_assign() -> None:
    section("2. RunnableParallel 并发 + RunnablePassthrough.assign 追加字段")
    joke = ChatPromptTemplate.from_template("讲一个关于{topic}的冷笑话，30 字内") | fast_llm | to_str
    poem = ChatPromptTemplate.from_template("写一句关于{topic}的诗，20 字内") | fast_llm | to_str
    # dict 字面量在 | 链里会被自动转换成 RunnableParallel，各分支并发执行
    combo = RunnableParallel(joke=joke, poem=poem)
    start = time.perf_counter()
    result = combo.invoke({"topic": "程序员"})
    show("并发结果", result)
    show("耗时", f"{time.perf_counter() - start:.1f}s（两个分支并行）")

    # assign：保留原输入，同时追加计算出的新字段 —— 构造 RAG 链时最常用
    enrich = RunnablePassthrough.assign(length=lambda x: len(x["text"]), upper=lambda x: x["text"].upper())
    show("assign", enrich.invoke({"text": "hello lcel"}))


def demo_lambda_and_chain_decorator() -> None:
    section("3. RunnableLambda / @chain：把普通函数变成 Runnable")
    word_count = RunnableLambda(lambda text: {"text": text, "chars": len(text)})
    show("RunnableLambda", word_count.invoke("LangChain 表达式语言"))

    @chain
    def translate_then_count(text: str) -> dict:
        """@chain 装饰的函数里可以自由调用其他 Runnable，适合写有分支/循环的复杂逻辑。"""
        english = (ChatPromptTemplate.from_template("把这句话翻译成英文，只输出译文：{t}") | fast_llm | to_str).invoke({"t": text})
        return {"english": english.strip(), "words": len(english.split())}

    show("@chain", translate_then_count.invoke("今天天气真不错"))


def demo_routing() -> None:
    section("4. 路由：RunnableBranch 与动态路由函数")
    classify = ChatPromptTemplate.from_template("问题“{q}”属于哪类？只回答 技术 / 生活 其中之一") | fast_llm | to_str
    tech = ChatPromptTemplate.from_template("你是资深工程师，用一句话专业地回答：{q}") | fast_llm | to_str
    life = ChatPromptTemplate.from_template("你是生活达人，用一句话轻松地回答：{q}") | fast_llm | to_str

    # 方式 A：RunnableBranch((条件, 分支), ..., 默认分支)
    branch = RunnableBranch((lambda x: "技术" in x["kind"], tech), life)
    router_a = RunnablePassthrough.assign(kind=classify) | branch

    # 方式 B：函数返回一个 Runnable，LCEL 会继续执行它（官方更推荐，逻辑更直观）
    def route(x: dict):
        return tech if "技术" in x["kind"] else life

    router_b = RunnablePassthrough.assign(kind=classify) | RunnableLambda(route)
    show("RunnableBranch", router_a.invoke({"q": "Redis 为什么快？"}))
    show("动态路由函数", router_b.invoke({"q": "周末去哪儿放松？"}))


def demo_retry_and_fallback() -> None:
    section("5. with_retry 重试 / with_fallbacks 降级")
    attempts = {"n": 0}

    def flaky(x: str) -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError(f"第 {attempts['n']} 次调用模拟网络抖动")
        return f"第 {attempts['n']} 次成功：{x}"

    robust = RunnableLambda(flaky).with_retry(stop_after_attempt=3, wait_exponential_jitter=False)
    show("with_retry", robust.invoke("ok"))

    # 主模型名写错 -> 报错 -> 自动切到备用模型。生产里常用于“主供应商挂了切备用供应商”
    broken = get_chat_model("model-that-does-not-exist", max_retries=0)
    safe = broken.with_fallbacks([fast_llm])
    reply = safe.invoke("只回复：降级成功")
    show("with_fallbacks", f"{reply.text.strip()}（实际模型 {reply.response_metadata.get('model_name')}）")


async def demo_stream_events() -> None:
    section("6. astream_events：观测链内每一步（构建前端进度条/调试的利器）")
    explain = (ChatPromptTemplate.from_template("用 20 字解释{c}") | fast_llm | to_str).with_config(run_name="解释链")
    seen: list[str] = []
    async for event in explain.astream_events({"c": "Embedding"}, version="v2"):
        kind = event["event"]
        if kind != "on_chat_model_stream" and kind != "on_parser_stream":
            seen.append(f"{kind}:{event['name']}")
    show("事件序列（已省略逐 token 的 stream 事件）", seen)

    section("7. get_graph：链结构可视化（Mermaid，可贴到 mermaid.live 查看）")
    print(explain.get_graph().draw_mermaid())


def main() -> None:
    banner("Demo 01-03 · LCEL 与 Runnable", "| 管道 / 并发 / 路由 / 重试降级 / 事件流")
    demo_sequence()
    demo_parallel_and_assign()
    demo_lambda_and_chain_decorator()
    demo_routing()
    demo_retry_and_fallback()
    asyncio.run(demo_stream_events())


if __name__ == "__main__":
    main()
