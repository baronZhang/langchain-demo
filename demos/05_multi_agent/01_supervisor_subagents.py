"""
Demo 05-01 · 多智能体 ①：Supervisor + 子 Agent 即工具（官方首推）
==============================================================
知识点
  1. 把子 Agent 包装成工具：主管 Agent 通过“调用工具”来委派任务，并决定调用谁、调用几次、何时汇总
  2. 上下文隔离：子 Agent 有独立的 system prompt / 工具 / 消息历史，主管只看到子 Agent 的结论
     → 主上下文干净、token 省，这也是 Deep Agents 里 task 工具的原理
  3. 子 Agent 可以用不同模型（便宜模型干简单活）
  4. stream(subgraphs=True) 可以看到子 Agent 内部的执行过程
  背景：早期的 langgraph-supervisor 库已归档，1.x 官方推荐直接用 create_agent + 工具实现。
对应文档：docs/03-详设篇/06-多智能体-DeepAgents-MCP.md
运行：uv run demos/05_multi_agent/01_supervisor_subagents.py
"""

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_demo import get_chat_model
from langchain_demo.console import banner, clip, section, show, show_messages
from langchain_demo.tools import calculator

llm = get_chat_model()
fast_llm = get_chat_model(fast=True)

_FAKE_FINANCE = {
    "星河科技": "星河科技 2025 年营收 12.6 亿元，净利润 1.8 亿元，员工 2300 人",
    "银河科技": "银河科技 2025 年营收 8.4 亿元，净利润 0.9 亿元，员工 1500 人",
}


@tool
def search_company(name: str) -> str:
    """查询公司的公开财务数据（演示用假数据）。

    Args:
        name: 公司名称
    """
    return _FAKE_FINANCE.get(name, f"未找到 {name} 的数据")


# ---------- 两个专业子 Agent ----------
researcher = create_agent(
    fast_llm,  # 查资料是简单活，用快模型
    tools=[search_company],
    system_prompt="你是研究员，只负责查询并如实汇报数据，不做计算和评论。",
    name="researcher",
)
analyst = create_agent(
    llm,
    tools=[calculator],
    system_prompt="你是财务分析师。所有数值计算必须调用 calculator，给出计算过程和结论。",
    name="analyst",
)


# ---------- 子 Agent → 工具 ----------
@tool
def ask_researcher(task: str) -> str:
    """把“需要查资料”的子任务交给研究员。

    Args:
        task: 完整、自包含的任务描述（研究员看不到你的对话历史）
    """
    result = researcher.invoke({"messages": [{"role": "user", "content": task}]})
    return result["messages"][-1].text  # 只把结论交回主管，子 Agent 的中间过程不污染主上下文


@tool
def ask_analyst(task: str) -> str:
    """把“需要计算/分析”的子任务交给财务分析师。

    Args:
        task: 完整、自包含的任务描述，必须包含计算所需的全部数据
    """
    result = analyst.invoke({"messages": [{"role": "user", "content": task}]})
    return result["messages"][-1].text


supervisor = create_agent(
    llm,
    tools=[ask_researcher, ask_analyst],
    system_prompt="你是项目经理。先让研究员查数据，再把数据交给分析师计算，最后用 3 句话汇总结论。不要自己编数据。",
    name="supervisor",
)


def main() -> None:
    banner("Demo 05-01 · Supervisor + 子 Agent 即工具", "委派 / 上下文隔离 / 子图流式")
    question = "对比星河科技和银河科技 2025 年的营收和净利率（净利润/营收），谁更赚钱？"

    section("1. stream(subgraphs=True)：观察主管与子 Agent 的协作过程")
    for namespace, chunk in supervisor.stream(
        {"messages": [{"role": "user", "content": question}]}, stream_mode="updates", subgraphs=True
    ):
        # namespace 为空 = 主管自己的节点；非空 = 在主管 tools 节点里被调用的子 Agent（子图）
        who = "主管" if not namespace else "  └─ 子Agent"
        for node, update in chunk.items():
            if not update or "messages" not in update:
                continue
            last = update["messages"][-1]
            # 1.x 会把 create_agent(name=...) 写进 AIMessage.name，借此区分是哪个 Agent 说的话
            speaker = f"({last.name})" if last.type == "ai" and last.name else ""
            detail = f"调用 {[c['name'] for c in last.tool_calls]}" if getattr(last, "tool_calls", None) else clip(last.text, 70)
            print(f"  {who}{speaker} [{node}] {detail}")

    section("2. 最终结果（主管的消息历史里只有子 Agent 的结论）")
    result = supervisor.invoke({"messages": [{"role": "user", "content": question}]})
    show_messages(result["messages"], max_chars=300)


if __name__ == "__main__":
    main()
