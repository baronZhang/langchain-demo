"""
MCP Server 示例：星河商城（被 demos/07_mcp/01_mcp_agent.py 调用，不单独作为 demo 运行）
================================================================================
MCP（Model Context Protocol）把“工具 / 资源 / 提示词”标准化成一个协议：
  - tools     ：模型可调用的函数（≈ LangChain 的 Tool）
  - resources ：可读取的上下文数据（文件、数据库记录…），用 URI 标识
  - prompts   ：服务端预置的提示词模板
一个 MCP Server 写一次，Claude Desktop / Cursor / LangChain Agent 等任何 MCP 客户端都能直接用。

这里用 FastMCP 实现；单独启动（stdio 传输）：uv run demos/07_mcp/_shop_server.py
"""

import os

# 关掉 FastMCP 的启动横幅和 INFO 日志，避免刷屏（stdio 子进程会继承这些环境变量）
os.environ.setdefault("FASTMCP_SHOW_SERVER_BANNER", "false")
os.environ.setdefault("FASTMCP_LOG_LEVEL", "WARNING")

from fastmcp import Context, FastMCP  # noqa: E402
from mcp import types as mcp_types  # noqa: E402

mcp = FastMCP("星河商城")

_ORDERS = {
    "SN001": {"status": "已发货", "item": "降噪耳机 X1", "amount": 1299, "carrier": "顺丰 SF123456"},
    "SN002": {"status": "待发货", "item": "机械键盘 K8", "amount": 499, "carrier": None},
}


@mcp.tool
def query_order(order_id: str) -> dict:
    """查询订单状态、商品与物流信息。

    Args:
        order_id: 订单号，例如 SN001
    """
    return _ORDERS.get(order_id, {"error": f"订单 {order_id} 不存在"})


@mcp.tool
async def cancel_order(order_id: str, ctx: Context) -> str:
    """取消一个尚未发货的订单。取消前会向用户二次确认。

    Args:
        order_id: 订单号
    """
    order = _ORDERS.get(order_id)
    if order is None:
        return f"订单 {order_id} 不存在"
    if order["status"] != "待发货":
        return f"订单 {order_id} 状态为「{order['status']}」，无法取消"

    # Elicitation（向用户索取输入）。MCP 2026-07-28 版协议改成了无状态的“多轮往返”（SEP-2322）：
    #   第 1 轮：工具返回 InputRequiredResult，列出要问的问题，本次调用结束；
    #   第 2 轮：客户端带着答案重新调用同一个工具，答案在 ctx.input_responses 里。
    # 旧协议里的 `await ctx.elicit(...)`（服务端主动发请求）在新协议连接上已不可用。
    # 客户端这边，langchain.mcp 会把第 1 轮的问题转换成 LangGraph interrupt，人回答后自动发起第 2 轮。
    answers = ctx.input_responses
    if answers is None:
        return mcp_types.InputRequiredResult(
            input_requests={
                "confirm": mcp_types.ElicitRequest(
                    params=mcp_types.ElicitRequestFormParams(
                        message=f"确认取消订单 {order_id}（{order['item']}，{order['amount']} 元）吗？",
                        requested_schema={
                            "type": "object",
                            "properties": {"confirm": {"type": "boolean", "title": "确认取消"}},
                            "required": ["confirm"],
                        },
                    )
                )
            }
        )
    reply = answers.get("confirm")
    if reply is not None and reply.action == "accept" and (reply.content or {}).get("confirm"):
        order["status"] = "已取消"
        return f"订单 {order_id} 已取消，{order['amount']} 元将原路退回"
    return f"用户未确认，订单 {order_id} 保持不变"


@mcp.resource("shop://policy/refund")
def refund_policy() -> str:
    """退款政策（资源：只读上下文）。"""
    return "未发货订单可随时取消；已签收商品 7 天内可无理由退货，运费由买家承担。"


@mcp.prompt
def customer_service(tone: str = "亲切") -> str:
    """客服话术模板（提示词）。"""
    return f"你是星河商城的客服，语气{tone}，回答简洁，涉及订单必须先查询再回答。"


if __name__ == "__main__":
    mcp.run()  # 默认 stdio 传输；mcp.run(transport="http", port=8000) 可改为 Streamable HTTP
