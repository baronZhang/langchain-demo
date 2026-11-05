"""
Demo 01-01 · 聊天模型（Chat Model）
==================================
知识点
  1. init_chat_model：按 provider + model 统一初始化（脚手架里由 get_chat_model 封装）
  2. Runnable 四件套：invoke / batch / stream / ainvoke
  3. AIMessage 解剖：content（厂商原始格式） vs content_blocks（1.0 标准内容块）、
     text、usage_metadata、response_metadata
  4. 流式块 AIMessageChunk 可以用 `+` 累加成完整消息
  5. 可配置模型：configurable_fields 让同一个对象在运行时切换模型
对应文档：docs/03-详设篇/01-模型-消息-提示词-LCEL.md
运行：uv run demos/01_core/01_chat_model.py
"""

import asyncio
import time

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from langchain_demo import get_chat_model, get_settings
from langchain_demo.console import banner, clip, section, show


def demo_invoke(llm: BaseChatModel) -> None:
    section("1. invoke 与 AIMessage 解剖")
    msg = llm.invoke(
        [
            SystemMessage("你是一名资深 Python 工程师，回答简洁。"),
            HumanMessage("用一句话解释 Python 的 GIL。"),
        ]
    )
    show("返回类型", type(msg).__name__)
    # content：厂商原始格式。Anthropic 协议下是块列表（thinking / text / tool_use），OpenAI 协议下通常是字符串
    raw_types = [b["type"] for b in msg.content] if isinstance(msg.content, list) else "str"
    show("content 原始块类型", raw_types)
    # content_blocks：LangChain 1.0 引入的标准内容块，跨厂商统一成 reasoning / text / tool_call / image ...
    # 上层代码只认 content_blocks，就不用为每家模型写解析逻辑
    show("content_blocks 标准块类型", [b["type"] for b in msg.content_blocks])
    show("text（只取正文）", msg.text)
    reasoning = [b.get("reasoning", "") for b in msg.content_blocks if b["type"] == "reasoning"]
    if reasoning:
        show("推理过程（截取）", clip(reasoning[0], 120))
    # usage_metadata：统一的 token 统计口径，做计费/限额时用它而不是 response_metadata
    show("usage_metadata", msg.usage_metadata)
    show("response_metadata（厂商原始元数据，节选）", {k: msg.response_metadata.get(k) for k in ("model_name", "stop_reason")})


def demo_input_formats_and_batch(llm: BaseChatModel) -> None:
    section("2. 三种等价的输入写法 + batch 并发")
    question = "中国的首都是哪里？只回答城市名"
    inputs = {
        "字符串（自动包装成 HumanMessage）": question,
        "消息对象列表": [HumanMessage(question)],
        "dict 列表（OpenAI 风格，适合直接接前端/数据库）": [{"role": "user", "content": question}],
    }
    start = time.perf_counter()
    # batch 内部用线程池并发执行；max_concurrency 控制并发度，防止打爆供应商限流
    results = llm.batch(list(inputs.values()), config={"max_concurrency": 3})
    for name, result in zip(inputs, results, strict=True):
        show(name, result.text.strip())
    show("batch 总耗时", f"{time.perf_counter() - start:.1f}s（3 个请求并发，约等于单次耗时）")


def demo_stream(llm: BaseChatModel) -> None:
    section("3. stream：逐块输出，chunk 可以用 + 累加")
    full = None
    reasoning_chars = 0
    for chunk in llm.stream("用三句话介绍杭州"):
        for block in chunk.content_blocks:
            if block["type"] == "reasoning":
                reasoning_chars += len(block.get("reasoning", ""))  # 推理模型先“想”再“说”
            elif block["type"] == "text":
                print(block["text"], end="", flush=True)
        full = chunk if full is None else full + chunk  # AIMessageChunk 支持 + 运算
    print()
    show("思考阶段字数", reasoning_chars)
    show("累加后的完整消息类型", type(full).__name__)
    show("累加后的 usage", full.usage_metadata)


async def demo_async(llm: BaseChatModel) -> None:
    section("4. ainvoke + asyncio.gather：异步并发")
    questions = ["用 5 个字形容春天", "用 5 个字形容夏天", "用 5 个字形容秋天"]
    start = time.perf_counter()
    answers = await asyncio.gather(*(llm.ainvoke(q) for q in questions))
    for q, a in zip(questions, answers, strict=True):
        show(q, a.text.strip())
    show("总耗时", f"{time.perf_counter() - start:.1f}s")


def demo_configurable() -> None:
    section("5. 可配置模型：同一个对象，运行时按 config 切换")
    s = get_settings()
    # configurable_fields 声明哪些构造参数允许在运行时通过 config["configurable"] 覆盖
    llm = init_chat_model(
        s.llm_model,
        model_provider=s.llm_provider,
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        max_tokens=s.llm_max_tokens,
        configurable_fields=("model",),
    )
    for name in (s.llm_model, s.llm_fast_model):
        start = time.perf_counter()
        reply = llm.invoke("1+1 等于几？只回答数字", config={"configurable": {"model": name}})
        actual = reply.response_metadata.get("model_name")
        show(f"请求 {name}", f"实际模型={actual}，回答={reply.text.strip()}，耗时={time.perf_counter() - start:.1f}s")


def main() -> None:
    banner("Demo 01-01 · 聊天模型", "init_chat_model / invoke / batch / stream / async / content_blocks")
    llm = get_chat_model()
    demo_invoke(llm)
    demo_input_formats_and_batch(llm)
    demo_stream(llm)
    asyncio.run(demo_async(llm))
    demo_configurable()


if __name__ == "__main__":
    main()
