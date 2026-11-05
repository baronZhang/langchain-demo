"""集中式配置：所有可变参数都从环境变量 / .env 读取（12-Factor 风格）。

设计要点
--------
1. demo 与业务代码只依赖 `get_settings()`，不直接读 os.environ；
   换模型、换供应商、换协议只改 .env，不改代码。
2. `lru_cache` 保证整个进程只解析一次；测试里可以 `get_settings.cache_clear()` 重新加载。
3. 刻意不用 pydantic-settings，让脚手架依赖最少、行为一眼能看懂。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# src/langchain_demo/settings.py -> 向上两级就是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# demo 运行时产物（SQLite checkpoint、deep agent 工作区等）统一放这里，已加入 .gitignore
RUNTIME_DIR = PROJECT_ROOT / ".runtime"
# RAG 等 demo 使用的示例语料
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class Settings:
    """不可变的配置快照。字段含义见 .env.example。"""

    llm_provider: str
    llm_base_url: str | None
    llm_api_key: str
    llm_model: str
    llm_fast_model: str
    llm_max_tokens: int
    llm_timeout: float
    embedding_provider: str
    embedding_base_url: str | None
    embedding_api_key: str | None
    embedding_model: str | None
    langsmith_tracing: bool


def _env(name: str, default: str | None = None) -> str | None:
    """读取环境变量；空字符串按“未设置”处理，避免 `KEY=` 这种写法变成空值。"""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # override=False：真实环境变量优先于 .env，方便 CI / 容器里注入覆盖
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    api_key = _env("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 LLM_API_KEY：请先执行 `cp .env.example .env` 并填写密钥")

    return Settings(
        llm_provider=_env("LLM_PROVIDER", "anthropic"),
        llm_base_url=_env("LLM_BASE_URL"),
        llm_api_key=api_key,
        llm_model=_env("LLM_MODEL", "glm-latest"),
        llm_fast_model=_env("LLM_FAST_MODEL") or _env("LLM_MODEL", "glm-latest"),
        llm_max_tokens=int(_env("LLM_MAX_TOKENS", "8192")),
        llm_timeout=float(_env("LLM_TIMEOUT", "180")),
        embedding_provider=_env("EMBEDDING_PROVIDER", "local"),
        embedding_base_url=_env("EMBEDDING_BASE_URL"),
        embedding_api_key=_env("EMBEDDING_API_KEY"),
        embedding_model=_env("EMBEDDING_MODEL"),
        langsmith_tracing=_env("LANGSMITH_TRACING", "false").lower() == "true",
    )
