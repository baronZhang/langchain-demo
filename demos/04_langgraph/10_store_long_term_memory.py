"""
Demo 04-10 · Store：跨线程长期记忆（含语义检索）
==============================================
知识点
  1. BaseStore 接口：put / get / search / delete / list_namespaces；namespace 是元组（类似文件路径）
  2. 语义检索：InMemoryStore(index={"embed": ..., "dims": ..., "fields": [...]})，search(query=...) 按相似度排序
  3. 在图中使用：compile(store=store)，节点参数声明 runtime: Runtime → runtime.store / runtime.context
  4. 记忆的三种类型（参考认知科学）：
     语义记忆（事实、偏好） / 情景记忆（过往经历、成功案例） / 程序记忆（规则、可自我更新的提示词）
  5. 写入时机：热路径（对话中实时抽取，本 demo）vs 后台（异步批量整理，延迟低、质量高）
  6. 生产：PostgresStore（pgvector）/ RedisStore；LangMem 库提供记忆抽取、合并的现成实现
对应文档：docs/03-详设篇/05-LangGraph运行时.md
运行：uv run demos/04_langgraph/10_store_long_term_memory.py
"""

import uuid
from dataclasses import dataclass

from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from langchain_demo import get_chat_model, get_embeddings
from langchain_demo.console import banner, section, show

llm = get_chat_model()


@dataclass
class Ctx:
    user_id: str


class Memories(BaseModel):
    facts: list[str] = Field(description="从用户最新一句话中提取的、值得长期记住的个人事实或偏好；没有则为空列表")


def remember(state: MessagesState, runtime: Runtime[Ctx]) -> dict:
    """热路径记忆抽取：每轮对话后，把值得记住的事实写进 store。"""
    extracted = llm.with_structured_output(Memories).invoke(f"用户说：{state['messages'][-1].text}")
    namespace = ("memories", runtime.context.user_id)
    for fact in extracted.facts:
        runtime.store.put(namespace, str(uuid.uuid4()), {"fact": fact})
        print(f"  [remember] 写入记忆：{fact}")
    return {}


def respond(state: MessagesState, runtime: Runtime[Ctx]) -> dict:
    """回答前按“当前问题”语义检索相关记忆，注入 system prompt。"""
    query = state["messages"][-1].text
    hits = runtime.store.search(("memories", runtime.context.user_id), query=query, limit=3)
    memory_text = "\n".join(f"- {h.value['fact']}（相关度 {h.score:.2f}）" for h in hits) or "（暂无）"
    print(f"  [respond] 召回记忆：\n{memory_text}")
    system = SystemMessage(f"你是贴心的私人助理，回答 50 字以内。已知的用户信息：\n{memory_text}")
    return {"messages": [llm.invoke([system, *state["messages"]])]}


def main() -> None:
    banner("Demo 04-10 · Store 长期记忆", "namespace / put / search / 语义检索 / runtime.store")

    section("1. Store 基础 API")
    store = InMemoryStore(index={"embed": get_embeddings(), "dims": 1024, "fields": ["fact"]})
    store.put(("demo", "u1"), "k1", {"fact": "喜欢喝美式咖啡"})
    show("get", store.get(("demo", "u1"), "k1").value)
    show("list_namespaces", store.list_namespaces())
    store.delete(("demo", "u1"), "k1")

    builder = StateGraph(MessagesState, context_schema=Ctx)
    builder.add_node("remember", remember)
    builder.add_node("respond", respond)
    builder.add_edge(START, "remember")
    builder.add_edge("remember", "respond")
    graph = builder.compile(checkpointer=InMemorySaver(), store=store)
    ctx = Ctx(user_id="u_1001")

    section("2. 线程 1：告诉助理一些个人信息")
    t1 = {"configurable": {"thread_id": "t1"}}
    for text in ["我对花生过敏，平时喜欢吃川菜", "我在深圳南山上班，每天骑车通勤"]:
        result = graph.invoke({"messages": [{"role": "user", "content": text}]}, t1, context=ctx)
        show("助理", result["messages"][-1].text)

    section("3. 线程 2（全新会话，没有聊天历史）：长期记忆依然可用")
    t2 = {"configurable": {"thread_id": "t2"}}
    result = graph.invoke({"messages": [{"role": "user", "content": "今晚请朋友吃饭，帮我推荐点菜"}]}, t2, context=ctx)
    show("助理", result["messages"][-1].text)

    section("4. 换一个用户：记忆按 namespace 隔离")
    result = graph.invoke({"messages": [{"role": "user", "content": "今晚请朋友吃饭，帮我推荐点菜"}]}, {"configurable": {"thread_id": "t3"}}, context=Ctx(user_id="u_2002"))
    show("助理（u_2002）", result["messages"][-1].text)


if __name__ == "__main__":
    main()
