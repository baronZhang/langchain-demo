"""
Demo 02-03 · Agentic RAG：让 Agent 自己决定“查不查、查几次、查什么”
================================================================
知识点
  1. 把检索封装成工具（content_and_artifact：文本给模型，原始 Document 留给程序做引用展示）
  2. create_agent + 检索工具：多跳问题会被拆成多次检索
  3. 1.x 的另一种写法：@dynamic_prompt 中间件在每次调用模型前注入检索结果
     （= 固定检索的两段式 RAG，但写在 Agent 体系里，能和其他中间件组合）
对比
  两段式 RAG：每次必检索一次，延迟低、可控；Agentic RAG：按需多次检索，能处理多跳问题，但更慢更贵
对应文档：docs/03-详设篇/03-RAG检索增强.md
运行：uv run demos/02_rag/03_agentic_rag.py
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.tools import tool
from langchain_core.documents import Document

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show, show_messages
from langchain_demo.rag import build_vector_store, format_docs

llm = get_chat_model()
store = build_vector_store()


@tool(response_format="content_and_artifact")
def search_handbook(query: str) -> tuple[str, list[Document]]:
    """检索星河科技内部资料（员工手册、报销制度、IT 安全规范、产品 FAQ）。

    Args:
        query: 检索查询，尽量使用制度文件里可能出现的关键词，一次只查一个主题。
    """
    docs = store.similarity_search(query, k=3)
    return format_docs(docs), docs


def demo_agentic_rag() -> None:
    section("1. 检索即工具：Agent 自主多次检索（多跳问题）")
    agent = create_agent(
        llm,
        tools=[search_handbook],
        system_prompt="你是星河科技员工助手。回答制度类问题前必须先检索；问题涉及多个主题时分别检索。回答注明出处文件名。",
    )
    question = "我入职满 3 年了，下个月去上海出差 3 晚，住宿最多能报销多少钱？另外我今年有几天年假？"
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    show_messages(result["messages"], max_chars=200)
    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    show("检索次数", len(tool_msgs))
    show("引用的文件（来自 artifact）", sorted({d.metadata["source"] for m in tool_msgs for d in m.artifact}))


def demo_dynamic_prompt_rag() -> None:
    section("2. @dynamic_prompt 中间件：每次调用模型前自动检索并注入 system prompt")

    @dynamic_prompt
    def rag_prompt(request: ModelRequest) -> str:
        question = request.state["messages"][-1].text  # 当前最新一条用户消息
        docs = store.similarity_search(question, k=3)
        return f"你是星河科技员工助手，只依据以下资料简洁回答，没有就说不知道：\n\n{format_docs(docs)}"

    agent = create_agent(llm, tools=[], middleware=[rag_prompt])
    result = agent.invoke({"messages": [{"role": "user", "content": "专业版套餐多少钱？API 限流多少？"}]})
    show("答", result["messages"][-1].text)


def main() -> None:
    banner("Demo 02-03 · Agentic RAG", "检索工具 + create_agent / dynamic_prompt 中间件注入上下文")
    demo_agentic_rag()
    demo_dynamic_prompt_rag()


if __name__ == "__main__":
    main()
