"""
Demo 03-01 · create_agent：LangChain 1.x 的标准 Agent
====================================================
知识点
  1. create_agent(model, tools, system_prompt)：1.x 唯一推荐的 Agent 入口，
     取代 0.x 的 AgentExecutor / initialize_agent 以及 langgraph.prebuilt.create_react_agent
  2. 返回的是编译好的 LangGraph 图（CompiledStateGraph）：可 invoke / stream / 可视化 / 持久化 / 部署
  3. 输入输出都是 state：{"messages": [...]}，最终回答是最后一条 AIMessage
  4. stream_mode="updates" 看每一步、stream_mode="messages" 拿逐 token 输出
  5. 短期记忆：checkpointer + thread_id 实现多轮对话
对应文档：docs/03-详设篇/04-Agent与中间件.md
运行：uv run demos/03_agents/01_create_agent_basics.py
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show, show_messages
from langchain_demo.tools import DEMO_TOOLS

llm = get_chat_model()
SYSTEM_PROMPT = "你是一个乐于助人的中文助手，需要实时信息或计算时调用工具。回答简洁。"


def demo_invoke() -> None:
    section("1. 创建并调用 Agent")
    # model 既可以传 BaseChatModel 实例，也可以传 "anthropic:claude-xxx" 这种字符串（内部走 init_chat_model）
    agent = create_agent(model=llm, tools=DEMO_TOOLS, system_prompt=SYSTEM_PROMPT, name="assistant")
    show("返回类型", type(agent).__name__)
    show("图中的节点", list(agent.get_graph().nodes))
    print("Mermaid 结构图：\n" + agent.get_graph().draw_mermaid())

    result = agent.invoke({"messages": [{"role": "user", "content": "现在几点了？成都天气怎么样？"}]})
    show_messages(result["messages"])


def demo_stream_updates() -> None:
    section("2. stream_mode='updates'：每个节点执行完推送一次增量")
    agent = create_agent(llm, tools=DEMO_TOOLS, system_prompt=SYSTEM_PROMPT)
    inputs = {"messages": [{"role": "user", "content": "帮我算一下 2 的 20 次方再除以 1024"}]}
    for chunk in agent.stream(inputs, stream_mode="updates"):
        for node, update in chunk.items():  # chunk 形如 {节点名: 该节点返回的 state 增量}
            last = update["messages"][-1]  # model 节点产出 AIMessage，tools 节点产出 ToolMessage（没有 tool_calls 属性）
            detail = getattr(last, "tool_calls", None) or last.text
            print(f"  [{node}] {type(last).__name__}: {detail}")

    section("3. stream_mode='messages'：逐 token 输出（聊天界面打字机效果）")
    for token, metadata in agent.stream(
        {"messages": [{"role": "user", "content": "用两句话介绍一下深圳"}]}, stream_mode="messages"
    ):
        # metadata["langgraph_node"] 告诉你 token 来自哪个节点，可用于只展示 model 节点的输出
        if metadata["langgraph_node"] == "model" and token.text:
            print(token.text, end="", flush=True)
    print()


def demo_memory() -> None:
    section("4. 短期记忆：checkpointer + thread_id")
    agent = create_agent(llm, tools=DEMO_TOOLS, system_prompt=SYSTEM_PROMPT, checkpointer=InMemorySaver())
    alice = {"configurable": {"thread_id": "user-alice"}}
    agent.invoke({"messages": [{"role": "user", "content": "我叫 Alice，我在杭州工作。"}]}, alice)
    reply = agent.invoke({"messages": [{"role": "user", "content": "我在哪个城市工作？那里今天天气如何？"}]}, alice)
    show("同一线程（记得上下文）", reply["messages"][-1].text)

    bob = {"configurable": {"thread_id": "user-bob"}}
    reply = agent.invoke({"messages": [{"role": "user", "content": "我在哪个城市工作？"}]}, bob)
    show("另一个线程（互相隔离）", reply["messages"][-1].text)

    state = agent.get_state(alice)  # checkpointer 存的就是图的 state 快照
    show("alice 线程累计消息数", len(state.values["messages"]))


def main() -> None:
    banner("Demo 03-01 · create_agent", "创建 / 调用 / 流式 / 多轮记忆")
    demo_invoke()
    demo_stream_updates()
    demo_memory()


if __name__ == "__main__":
    main()
