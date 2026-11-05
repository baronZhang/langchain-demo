"""
Demo 04-04 · 持久化与时间旅行（Checkpointer）
==========================================
知识点
  1. Checkpointer 在每个超步结束后保存一次 state 快照（checkpoint）；thread_id 是“会话主键”
  2. 实现：InMemorySaver（开发）/ SqliteSaver（单机）/ PostgresSaver（生产）/ 社区 Redis、MongoDB …
  3. get_state：当前快照（values / next / config / metadata / tasks）
  4. get_state_history：该线程全部历史快照（新 → 旧）
  5. 时间旅行
     - Replay（重放）：拿历史 checkpoint 的 config 调 invoke(None, config) → 从那一刻重新执行后续步骤
     - Fork  （分叉）：update_state(历史 config, 新值) 生成一个新分支 checkpoint → 从分支继续执行
  6. durability：checkpoint 写入时机 "sync"（每步同步写，最安全）/ "async"（默认）/ "exit"（只在结束写，最快）
  持久化是 LangGraph 的地基：多轮记忆、人在回路、容错恢复、时间旅行都建立在它之上。
对应文档：docs/03-详设篇/05-LangGraph运行时.md
运行：uv run demos/04_langgraph/04_persistence_time_travel.py
"""

import uuid
from typing import NotRequired, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from langchain_demo import RUNTIME_DIR, get_chat_model
from langchain_demo.console import banner, section, show

llm = get_chat_model()


class JokeState(TypedDict):
    topic: NotRequired[str]
    joke: NotRequired[str]


def generate_topic(state: JokeState) -> dict:
    return {"topic": llm.invoke("随机给出一个适合讲冷笑话的主题，只输出主题本身，4 个字以内").text.strip()}


def write_joke(state: JokeState) -> dict:
    return {"joke": llm.invoke(f"写一个关于“{state['topic']}”的冷笑话，40 字以内，只输出笑话").text.strip()}


def build_builder() -> StateGraph:
    builder = StateGraph(JokeState)
    builder.add_node("generate_topic", generate_topic)
    builder.add_node("write_joke", write_joke)
    builder.add_edge(START, "generate_topic")
    builder.add_edge("generate_topic", "write_joke")
    builder.add_edge("write_joke", END)
    return builder


def main() -> None:
    banner("Demo 04-04 · 持久化与时间旅行", "SqliteSaver / get_state / history / replay / fork")
    RUNTIME_DIR.mkdir(exist_ok=True)
    db_path = RUNTIME_DIR / "checkpoints.sqlite"

    # SqliteSaver 是上下文管理器：负责打开/关闭数据库连接；首次使用会自动建表
    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = build_builder().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": f"joke-{uuid.uuid4().hex[:6]}"}}

        section("1. 首次运行（durability='sync'：每个超步都同步落盘）")
        result = graph.invoke({}, config, durability="sync")
        show("结果", result)

        section("2. get_state：当前快照")
        snapshot = graph.get_state(config)
        show("values", snapshot.values)
        show("next（为空表示已运行结束）", snapshot.next)

        section("3. get_state_history：所有历史快照（新 → 旧）")
        history = list(graph.get_state_history(config))
        for snap in history:
            cid = snap.config["configurable"]["checkpoint_id"]
            show(f"step={snap.metadata['step']:>2} id={cid[-8:]}", f"next={snap.next} values={snap.values}")

        before_joke = next(s for s in history if s.next == ("write_joke",))  # 写笑话之前的那个快照

        section("4. Replay：从“写笑话之前”重放 → 主题不变，笑话重新生成")
        replay = graph.invoke(None, before_joke.config)
        show("重放结果", replay)

        section("5. Fork：修改历史快照里的主题，再从那里继续")
        fork_config = graph.update_state(before_joke.config, {"topic": "程序员加班"})
        forked = graph.invoke(None, fork_config)
        show("分叉结果", forked)
        show("该线程快照总数", len(list(graph.get_state_history(config))))

    show("SQLite 文件", db_path.relative_to(RUNTIME_DIR.parent))
    print("提示：换成 PostgresSaver（langgraph-checkpoint-postgres）即可用于多实例生产部署，代码其余部分不变。")


if __name__ == "__main__":
    main()
