"""终端打印小工具：让每个 demo 的输出层次清晰、便于对照代码阅读。"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

_COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def banner(title: str, subtitle: str = "") -> None:
    """demo 大标题。"""
    line = "=" * 72
    print(f"\n{_c('1;36', line)}\n{_c('1;36', title)}")
    if subtitle:
        print(_c("36", subtitle))
    print(_c("1;36", line))


def section(title: str) -> None:
    """demo 内的小节标题。"""
    print(f"\n{_c('1;33', '▶ ' + title)}")


def show(label: str, value: Any) -> None:
    """打印一个键值；dict / list 自动转成缩进 JSON。"""
    if isinstance(value, dict | list):
        value = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    print(f"{_c('32', label + ':')} {value}")


def clip(text: str, max_chars: int = 600) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars] + f" …(共 {len(text)} 字)"


def show_messages(messages: Iterable[BaseMessage], *, max_chars: int = 600) -> None:
    """按“角色 + 正文 + 工具调用”的格式打印消息列表（自动忽略 thinking 块）。"""
    for msg in messages:
        role = _c("35", f"[{msg.type}]")
        if isinstance(msg, ToolMessage):
            print(f"{role} ({msg.name}) {clip(msg.text, max_chars)}")
            continue
        text = msg.text
        if text:
            print(f"{role} {clip(text, max_chars)}")
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                args = json.dumps(call["args"], ensure_ascii=False)
                print(f"{role} {_c('34', '→ 调用工具')} {call['name']}({args})")
