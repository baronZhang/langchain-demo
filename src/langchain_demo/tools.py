"""多个 demo 共用的示例工具。

工具 = 一个“带 schema 的函数”。`@tool` 会从函数签名 + docstring 自动生成：
  - name        ：函数名
  - description ：docstring 首段（模型据此决定“要不要调用”）
  - args_schema ：参数的 JSON Schema（模型据此决定“怎么传参”）
`parse_docstring=True` 会把 Google 风格 docstring 的 Args 段解析成每个参数的描述。

注意：这里的数据都是演示用假数据。
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain.tools import tool

_FAKE_WEATHER = {
    "北京": "晴，12~22℃，北风 2 级",
    "上海": "多云，16~23℃，东风 3 级",
    "深圳": "阵雨，24~30℃，湿度 85%",
    "杭州": "小雨，15~20℃，空气质量良",
    "成都": "阴，14~19℃，微风",
}


@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """查询指定城市今天的天气。

    Args:
        city: 城市中文名，例如“北京”“深圳”。
    """
    return _FAKE_WEATHER.get(city, f"{city}：暂无数据")


# 只允许这些运算符，避免 eval 带来的代码注入风险
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式片段: {ast.dump(node)}")


@tool(parse_docstring=True)
def calculator(expression: str) -> str:
    """计算算术表达式，支持 + - * / ** % 和括号。

    Args:
        expression: 合法的算术表达式，例如 "(3 + 5) * 2"。
    """
    result = _safe_eval(ast.parse(expression, mode="eval").body)
    return str(round(result, 6))


@tool
def get_current_time() -> str:
    """获取当前北京时间（精确到分钟）。"""
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")


DEMO_TOOLS = [get_weather, calculator, get_current_time]
