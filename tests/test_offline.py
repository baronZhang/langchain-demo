"""离线单元测试：不调用任何真实模型，秒级跑完，可放进 CI。

LLM 应用怎么测？（面试常问）
  1. 确定性部分（工具、切分、路由、图结构、reducer）→ 普通单测
  2. 依赖模型的流程 → 用 Fake 模型返回预设消息，验证“编排逻辑”而不是“模型能力”
  3. 模型效果 → 评估（见 demos/08_production/02_evaluation.py），而不是单测
运行：uv run --group dev pytest -q
"""

import operator
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt

from langchain_demo.embeddings import HashingEmbeddings
from langchain_demo.rag import load_markdown_dir, split_markdown
from langchain_demo.tools import calculator


class FakeToolCallingModel(GenericFakeChatModel):
    """GenericFakeChatModel 按顺序吐出预设消息；bind_tools 直接返回自身即可满足 create_agent。"""

    def bind_tools(self, tools, **kwargs):
        return self


def test_calculator_tool() -> None:
    assert calculator.invoke({"expression": "(3 + 5) * 12"}) == "96"


def test_hashing_embeddings_prefers_lexical_overlap() -> None:
    emb = HashingEmbeddings()
    query, near, far = emb.embed_documents(["年假有几天", "年假天数按司龄计算", "密码必须 90 天更换"])
    dot = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))  # noqa: E731
    assert abs(dot(query, query) - 1.0) < 1e-9  # L2 归一化
    assert dot(query, near) > dot(query, far)


def test_markdown_split_keeps_header_metadata() -> None:
    chunks = split_markdown(load_markdown_dir(), chunk_size=300, chunk_overlap=50)
    assert chunks and all("source" in c.metadata for c in chunks)
    assert any(c.metadata.get("h2") == "年假制度" for c in chunks)


def test_create_agent_runs_tool_loop_with_fake_model() -> None:
    model = FakeToolCallingModel(
        messages=iter(
            [
                AIMessage("", tool_calls=[{"name": "calculator", "args": {"expression": "6*7"}, "id": "call_1"}]),
                AIMessage("答案是 42"),
            ]
        )
    )
    agent = create_agent(model, tools=[calculator])
    result = agent.invoke({"messages": [HumanMessage("6 乘 7 等于几？")]})
    assert [m.type for m in result["messages"]] == ["human", "ai", "tool", "ai"]
    assert result["messages"][2].content == "42"
    assert result["messages"][-1].text == "答案是 42"


def test_parallel_branches_need_reducer() -> None:
    class State(TypedDict):
        logs: Annotated[list[str], operator.add]

    builder = StateGraph(State)
    builder.add_node("a", lambda s: {"logs": ["a"]})
    builder.add_node("b", lambda s: {"logs": ["b"]})
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    assert sorted(builder.compile().invoke({"logs": []})["logs"]) == ["a", "b"]


def test_interrupt_and_resume() -> None:
    class State(TypedDict):
        answer: str

    builder = StateGraph(State)
    builder.add_node("ask", lambda s: {"answer": interrupt("你的名字？")})
    builder.add_edge(START, "ask")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t"}}
    first = graph.invoke({"answer": ""}, config)
    assert first["__interrupt__"][0].value == "你的名字？"
    assert graph.invoke(Command(resume="小王"), config)["answer"] == "小王"
