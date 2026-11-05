"""
Demo 06-01 · Deep Agents：面向长任务的 Agent “整机”（harness）
=============================================================
Deep Agents（deepagents 包）= create_agent + 一套精心设计的默认中间件 + 系统提示词，
模仿 Claude Code / Manus 这类“能干长活”的 Agent。四大支柱：
  1. 文件系统（FilesystemMiddleware）：ls / read_file / write_file / edit_file / glob / grep / delete
     —— 把大段中间结果写进“文件”而不是塞进上下文（上下文卸载）
  2. 子代理（SubAgentMiddleware）：task 工具把子任务派给上下文隔离的子 Agent
  3. 上下文管理：自动摘要（SummarizationMiddleware）、大工具结果落盘
  4. 记忆与技能：memory=[AGENTS.md] 常驻提示词；skills=[目录] 按需渐进加载 SKILL.md
后端（backend）决定“文件”存在哪：
  StateBackend（默认，存在 Agent state 里，随 checkpointer 持久化）/ FilesystemBackend（本地磁盘）/
  StoreBackend（LangGraph Store，跨线程）/ CompositeBackend（按路径路由到不同后端）/ 沙箱后端（可执行命令）
注意：0.7 起 TodoListMiddleware 不再默认启用（需要规划时显式加入 middleware）。
对应文档：docs/03-详设篇/06-多智能体-DeepAgents-MCP.md
运行：uv run demos/06_deep_agents/01_deep_agent.py
"""

import shutil
from collections import Counter

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import TodoListMiddleware
from langchain.tools import tool

from langchain_demo import RUNTIME_DIR, get_chat_model
from langchain_demo.console import banner, clip, section, show

llm = get_chat_model()
WORKSPACE = RUNTIME_DIR / "deep_agent_ws"

AGENTS_MD = """# 团队约定（memory：每轮都会注入系统提示词）
- 所有产出使用简体中文，Markdown 格式
- 报告类文件统一写到 /reports/ 目录
- 文档结尾署名“—— 星河 AI 学习小组”
"""

SKILL_MD = """---
name: interview-prep
description: 当用户需要准备技术面试材料（面试题、知识点清单）时使用本技能
---
# 面试准备材料生成规范
1. 先列出该方向 3~5 个核心知识点
2. 每个知识点给出 2 个高频面试题，并附一句话答题要点
3. 最后给出 3 条临场建议
4. 用 write_file 写入 /reports/interview_prep.md，然后只向用户汇报文件路径和一句话摘要
"""


@tool
def search_tech_notes(topic: str) -> str:
    """检索内部技术笔记（演示用假数据）。

    Args:
        topic: 技术主题关键词
    """
    return (
        f"关于「{topic}」的笔记：LangGraph 以 StateGraph 建模，核心概念包括 State/Node/Edge/Reducer；"
        "执行模型借鉴 Pregel（超步 + 消息传递）；Checkpointer 提供持久化，支持 interrupt 人在回路、时间旅行；"
        "Send 实现动态并行；create_agent 是构建在 LangGraph 之上的 Agent 抽象，通过中间件扩展。"
    )


def prepare_workspace() -> None:
    shutil.rmtree(WORKSPACE, ignore_errors=True)
    (WORKSPACE / "skills" / "interview-prep").mkdir(parents=True)
    (WORKSPACE / "AGENTS.md").write_text(AGENTS_MD, encoding="utf-8")
    (WORKSPACE / "skills" / "interview-prep" / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")


def demo_state_backend() -> None:
    section("1. 默认 StateBackend：“文件”存在 Agent state 里（不落盘，适合 Web 服务多租户）")
    agent = create_deep_agent(model=llm, system_prompt="你是一个简洁的助手。")
    result = agent.invoke({"messages": [{"role": "user", "content": "把“你好，Deep Agents”写入 /hello.txt，然后读出来确认"}]})
    show("state['files'] 中的文件", list(result.get("files", {})))
    show("回答", clip(result["messages"][-1].text, 100))


def demo_filesystem_backend() -> None:
    section("2. FilesystemBackend + memory + skills + 子代理 + 规划")
    prepare_workspace()
    agent = create_deep_agent(
        model=llm,
        tools=[search_tech_notes],
        system_prompt="你是资深技术导师。复杂任务先用 write_todos 规划；调研类子任务交给 researcher 子代理。",
        # virtual_mode=True：Agent 看到的 "/" 映射到 root_dir，且无法越界访问 root_dir 之外的文件
        backend=FilesystemBackend(root_dir=WORKSPACE, virtual_mode=True),
        memory=["/AGENTS.md"],  # 常驻记忆
        skills=["/skills/"],  # 技能目录：只把技能的 name/description 放进提示词，需要时 Agent 再 read_file 全文
        middleware=[TodoListMiddleware()],  # 0.7 起规划工具需显式启用
        subagents=[
            {
                "name": "researcher",
                "description": "调研某个技术主题，返回 5 条以内的要点",
                "system_prompt": "你是技术调研员，先用 search_tech_notes 检索，再整理成不超过 5 条要点返回。",
                "tools": [search_tech_notes],
            }
        ],
    )
    task = "我下周面试 AI 应用工程师（LangGraph 方向），请帮我准备一份面试材料。"
    result = agent.invoke({"messages": [{"role": "user", "content": task}]}, {"recursion_limit": 80})

    tool_calls = Counter(call["name"] for m in result["messages"] if m.type == "ai" for call in m.tool_calls)
    show("工具调用统计", dict(tool_calls))
    show("规划（todos）", result.get("todos"))
    show("最终回答", clip(result["messages"][-1].text, 200))

    section("3. 工作区里的文件（Agent 真实写到了磁盘）")
    for path in sorted(p for p in WORKSPACE.rglob("*") if p.is_file()):
        show(str(path.relative_to(WORKSPACE)), f"{path.stat().st_size} 字节")
    report = WORKSPACE / "reports" / "interview_prep.md"
    if report.exists():
        print(clip(report.read_text(encoding="utf-8"), 800))


def main() -> None:
    banner("Demo 06-01 · Deep Agents", "文件系统 / 子代理 / 记忆 / 技能 / 后端")
    demo_state_backend()
    demo_filesystem_backend()


if __name__ == "__main__":
    main()
