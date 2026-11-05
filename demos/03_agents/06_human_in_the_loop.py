"""
Demo 03-06 · 人在回路（Human-in-the-Loop）
========================================
知识点
  1. HumanInTheLoopMiddleware(interrupt_on={...})：指定哪些工具调用需要人工审批
  2. 依赖 checkpointer：中断时 state 已持久化，人类决策后从断点恢复（可以跨进程、跨天）
  3. 中断信息在 result["__interrupt__"]；用 Command(resume=...) 恢复执行
  4. 四种决策：approve 批准 / edit 改参数后执行 / reject 拒绝并说明原因 / respond 由人直接给出工具结果
  5. interrupt_mode："batched"（默认，一轮模型输出一个中断，decisions 按顺序对应）
                    "per_call"（每个工具调用一个中断，按 interrupt id 回复，计划成为下个大版本默认值）
底层原理：batched 模式在 after_model 钩子里、per_call 模式在 wrap_tool_call 里调用 LangGraph 的 interrupt()
（见 demos/04_langgraph/05_interrupt_hitl.py）
对应文档：docs/03-详设篇/04-Agent与中间件.md
运行：uv run demos/03_agents/06_human_in_the_loop.py
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show
from langchain_demo.tools import get_weather


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件。

    Args:
        to: 收件人邮箱
        subject: 主题
        body: 正文
    """
    return f"邮件已发送 → {to}，主题《{subject}》，正文：{body}"


@tool
def delete_file(path: str) -> str:
    """删除服务器上的文件。

    Args:
        path: 文件路径
    """
    return f"已删除 {path}"


agent = create_agent(
    get_chat_model(),
    tools=[send_email, delete_file, get_weather],
    checkpointer=InMemorySaver(),  # HITL 必须有 checkpointer，否则中断后无处恢复
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
                "delete_file": True,  # True = 允许所有决策类型
                "get_weather": False,  # False = 不需要审批（只读工具）
            },
            description_prefix="以下操作需要人工审批",
        )
    ],
)


def run_until_interrupt(text: str, thread_id: str) -> tuple[dict, dict] | None:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
    if "__interrupt__" not in result:  # 模型没有调用需审批的工具（例如它自己判断应先向用户确认）
        show("未触发审批，模型直接回复", result["messages"][-1].text)
        return None
    interrupt = result["__interrupt__"][0]  # Interrupt(value=..., id=...)
    request = interrupt.value  # HITLRequest：action_requests（待审批的调用）+ review_configs（允许的决策）
    for action in request["action_requests"]:
        show("待审批", f"{action['name']}({action['args']})")
    show("允许的决策", [c["allowed_decisions"] for c in request["review_configs"]])
    return result, config


def main() -> None:
    banner("Demo 03-06 · 人在回路", "HumanInTheLoopMiddleware：approve / edit / reject")

    section("1. approve：直接批准")
    if pending := run_until_interrupt("给 boss@example.com 发邮件，主题“周报”，正文“本周完成了 RAG 模块”", "t-approve"):
        result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), pending[1])
        show("工具结果", result["messages"][-2].text)
        show("最终回答", result["messages"][-1].text)

    section("2. edit：人工修改参数后再执行")
    if pending := run_until_interrupt("给 team@example.com 发邮件通知明天放假", "t-edit"):
        edited = {"to": "team@example.com", "subject": "【通知】明日调休", "body": "明天调休一天，后天正常上班。——行政部"}
        result = agent.invoke(
            Command(resume={"decisions": [{"type": "edit", "edited_action": {"name": "send_email", "args": edited}}]}),
            pending[1],
        )
        # 中间件会在工具结果前加一段 edit_notice，告诉模型“参数被人改过，不要重发原调用”
        show("工具结果（最后一行是真实执行结果）", result["messages"][-2].text.splitlines()[-1])

    section("3. reject：拒绝并告诉模型原因，模型会据此调整回答")
    if pending := run_until_interrupt("帮我删除临时文件 /tmp/cache/report_2025.csv", "t-reject"):
        result = agent.invoke(
            Command(resume={"decisions": [{"type": "reject", "message": "该文件仍在审计保留期内，暂不能删除"}]}),
            pending[1],
        )
        show("最终回答", result["messages"][-1].text)

    section("4. 不需要审批的工具照常执行")
    result = agent.invoke({"messages": [{"role": "user", "content": "深圳天气如何？"}]}, {"configurable": {"thread_id": "t-free"}})
    show("是否中断", "__interrupt__" in result)
    show("回答", result["messages"][-1].text)


if __name__ == "__main__":
    main()
