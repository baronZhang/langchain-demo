"""
Demo 04-03 · 从零手写 ReAct Agent（看清 create_agent 的底层）
==========================================================
知识点
  1. MessagesState：内置 state，等价于 {"messages": Annotated[list[AnyMessage], add_messages]}
  2. 模型节点：llm.bind_tools(tools).invoke(messages)，产出带 tool_calls 的 AIMessage
  3. ToolNode：并行执行最后一条 AIMessage 里的所有 tool_calls，返回对应的 ToolMessage（内置异常处理）
  4. tools_condition：最后一条消息有 tool_calls → "tools"，否则 → END
  5. tools → model 的回边形成 Reason → Act → Observe 循环（ReAct）
  6. 对照 create_agent：拓扑完全一致；create_agent 额外提供中间件插槽、结构化输出、运行时上下文等工程能力
对应文档：docs/03-详设篇/05-LangGraph运行时.md
运行：uv run demos/04_langgraph/03_react_agent_from_scratch.py
"""

from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show_messages
from langchain_demo.tools import DEMO_TOOLS

llm = get_chat_model()
model_with_tools = llm.bind_tools(DEMO_TOOLS)
SYSTEM = SystemMessage("你是一个中文助手，需要实时信息或计算时调用工具，回答简洁。")


def call_model(state: MessagesState) -> dict:
    """Reason：让模型基于完整历史决定“直接回答”还是“调用工具”。"""
    response = model_with_tools.invoke([SYSTEM, *state["messages"]])
    return {"messages": [response]}  # add_messages reducer 会把它追加到历史末尾


def build_react_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(DEMO_TOOLS))  # Act：执行工具
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)  # 有 tool_calls → tools，否则 → END
    builder.add_edge("tools", "model")  # Observe：工具结果回到模型
    return builder.compile()


def main() -> None:
    banner("Demo 04-03 · 手写 ReAct Agent", "MessagesState + ToolNode + tools_condition")
    graph = build_react_graph()

    section("1. 手写版的图结构")
    print(graph.get_graph().draw_mermaid())

    section("2. 运行：多工具、多步推理")
    result = graph.invoke({"messages": [HumanMessage("现在几点？顺便算一下 1024 * 768，再告诉我上海天气")]})
    show_messages(result["messages"])

    section("3. 对照 create_agent 的图结构（节点名同为 model / tools）")
    print(create_agent(llm, tools=DEMO_TOOLS).get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
