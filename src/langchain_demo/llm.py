"""模型工厂：全项目唯一创建 ChatModel 的地方。

为什么要包一层？
----------------
- LangChain 1.x 推荐用 `init_chat_model` 按 provider 统一初始化模型，屏蔽各家 SDK 差异；
- 但 base_url / api_key / 超时 / 重试 这类“横切参数”不应该散落在每个 demo 里，
  所以这里再包一层：调用方只需要说“我要主模型还是快模型”。

协议说明
--------
火山方舟 Coding Plan、智谱、Kimi、DeepSeek 等国产服务普遍同时提供
Anthropic 兼容和 OpenAI 兼容两套端点。LLM_PROVIDER=anthropic 时底层是
`langchain_anthropic.ChatAnthropic`，=openai 时是 `langchain_openai.ChatOpenAI`，
上层代码完全不用改——这正是 BaseChatModel 统一抽象的价值。
"""

from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from langchain_demo.settings import get_settings


def get_chat_model(model: str | None = None, *, fast: bool = False, **overrides: Any) -> BaseChatModel:
    """创建聊天模型。

    Args:
        model: 显式指定模型名；为空时按 `fast` 取 LLM_MODEL 或 LLM_FAST_MODEL。
        fast: 是否使用“快模型”（摘要、路由、工具筛选、兜底等辅助任务）。
        **overrides: 透传给具体 ChatModel 的参数，如 `temperature`、`max_tokens`。

    Returns:
        一个 `BaseChatModel`，支持 invoke / stream / batch / bind_tools /
        with_structured_output 等全部标准接口。
    """
    s = get_settings()
    params: dict[str, Any] = {
        "model": model or (s.llm_fast_model if fast else s.llm_model),
        "model_provider": s.llm_provider,
        "api_key": s.llm_api_key,
        # 推理模型（如 glm-5.x）会先产出 thinking 再产出正文，max_tokens 太小会只剩思考没有答案
        "max_tokens": s.llm_max_tokens,
        "timeout": s.llm_timeout,
        # SDK 层面的网络重试；业务层重试/降级交给 Runnable.with_retry 或 Agent 中间件
        "max_retries": 2,
    }
    if s.llm_base_url:
        params["base_url"] = s.llm_base_url
    params.update(overrides)
    return init_chat_model(**params)
