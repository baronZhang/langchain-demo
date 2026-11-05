"""langchain-demo 脚手架公共模块。

demo 和后续业务代码只需要：
    from langchain_demo import get_chat_model, get_embeddings
就能拿到按 .env 配置好的模型，与具体供应商解耦。
"""

from __future__ import annotations

import time

from langchain_demo.embeddings import get_embeddings
from langchain_demo.llm import get_chat_model
from langchain_demo.settings import DATA_DIR, PROJECT_ROOT, RUNTIME_DIR, get_settings

__all__ = ["DATA_DIR", "PROJECT_ROOT", "RUNTIME_DIR", "get_chat_model", "get_embeddings", "get_settings"]


def main() -> None:
    """`uv run langchain-demo`：环境自检——打印当前配置并对主/快模型各做一次连通性测试。"""
    from importlib.metadata import version

    s = get_settings()
    print("== 配置 ==")
    print(f"provider     : {s.llm_provider}")
    print(f"base_url     : {s.llm_base_url}")
    print(f"主模型 / 快模型: {s.llm_model} / {s.llm_fast_model}")
    print(f"embedding    : {s.embedding_provider} {s.embedding_model or ''}")
    print(f"LangSmith    : {'开启' if s.langsmith_tracing else '关闭'}")
    print("\n== 版本 ==")
    for pkg in ("langchain", "langchain-core", "langgraph", "deepagents", f"langchain-{s.llm_provider}"):
        print(f"{pkg:<20} {version(pkg)}")
    print("\n== 连通性 ==")
    for fast in (False, True):
        llm = get_chat_model(fast=fast)
        start = time.perf_counter()
        reply = llm.invoke("只回复 pong 四个字母")
        model_name = reply.response_metadata.get("model_name") or reply.response_metadata.get("model")
        print(f"{'快模型' if fast else '主模型'}: {reply.text.strip()!r}  实际模型={model_name}  耗时={time.perf_counter() - start:.1f}s")
