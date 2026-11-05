"""
Demo 08-02 · 评估（Evaluation）：让“效果好不好”可度量
==================================================
LLM 应用上线前后都要回答：改了 prompt / 换了模型 / 调了切分参数之后，效果变好还是变差？
知识点
  1. 评估三要素：数据集（输入 + 参考答案）→ 被测目标（链 / Agent）→ 评估器（规则 / LLM-as-judge / 人工）
  2. RAG 常用指标：答案正确性（vs 参考答案）、忠实度（是否只依据检索内容，防幻觉）、检索命中率
  3. LLM-as-judge：用结构化输出让裁判模型给出分数 + 理由，便于统计和复查
  4. 线上闭环：LangSmith 把 trace 沉淀为数据集，`langsmith.evaluate()` 跑实验并对比版本；
     openevals / agentevals 提供现成的评估器（含 Agent 轨迹评估）
本 demo 是不依赖 LangSmith 的离线版本，思路与 `evaluate()` 完全一致。
对应文档：docs/04-工程篇/01-脚手架-可观测-部署.md
运行：uv run demos/08_production/02_evaluation.py
"""

from concurrent.futures import ThreadPoolExecutor

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show
from langchain_demo.rag import build_vector_store, format_docs

# ---------- 1. 数据集：输入 + 参考答案 + 期望命中的文档 ----------
DATASET = [
    {"question": "入职满 3 年有几天年假？", "reference": "10 天", "source": "employee_handbook.md"},
    {"question": "去深圳出差住宿每晚能报多少？", "reference": "一线城市每晚上限 600 元", "source": "expense_policy.md"},
    {"question": "公司密码多久必须更换一次？", "reference": "每 90 天", "source": "it_security.md"},
    {"question": "专业版套餐的 API 限流是多少？", "reference": "50 QPS", "source": "product_faq.md"},
    {"question": "公司年会在哪里举办？", "reference": "资料中没有相关信息（应拒答）", "source": None},
]

# ---------- 2. 被测目标：02_rag 里的两段式 RAG 链 ----------
retriever = build_vector_store().as_retriever(search_kwargs={"k": 3})
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "只依据资料回答，资料中没有就回答“资料中没有相关信息”。回答一句话。\n资料：\n{context}"),
        ("human", "{question}"),
    ]
)
generate = (lambda x: {"context": format_docs(x["docs"]), "question": x["question"]}) | PROMPT | get_chat_model() | StrOutputParser()
target = RunnableParallel(docs=retriever, question=RunnablePassthrough()).assign(answer=generate)


# ---------- 3. 评估器 ----------
class Verdict(BaseModel):
    correctness: int = Field(ge=1, le=5, description="与参考答案的一致程度，5 为完全正确")
    faithful: bool = Field(description="回答是否完全基于给定资料，没有编造")
    reason: str = Field(description="一句话理由")


judge = get_chat_model(fast=True).with_structured_output(Verdict)  # 裁判用快模型，降低评估成本


def evaluate_one(example: dict) -> dict:
    out = target.invoke(example["question"])
    sources = [d.metadata["source"] for d in out["docs"]]
    verdict = judge.invoke(
        f"问题：{example['question']}\n参考答案：{example['reference']}\n"
        f"检索资料：\n{format_docs(out['docs'])}\n待评回答：{out['answer']}\n请评估。"
    )
    hit = example["source"] is None or example["source"] in sources  # 规则评估器：检索命中
    return {**example, "answer": out["answer"], "hit": hit, **verdict.model_dump()}


def main() -> None:
    banner("Demo 08-02 · 离线评估", "数据集 → 目标 → 评估器（规则 + LLM-as-judge）→ 指标")
    with ThreadPoolExecutor(max_workers=5) as pool:  # 用例之间相互独立，可以并发评估
        results = list(pool.map(evaluate_one, DATASET))

    section("1. 逐条结果")
    for r in results:
        show(r["question"], f"正确性 {r['correctness']}/5 | 忠实 {r['faithful']} | 命中 {r['hit']} | {r['answer'][:40]}")
        print(f"    裁判理由：{r['reason']}")

    section("2. 汇总指标")
    n = len(results)
    show("平均正确性", f"{sum(r['correctness'] for r in results) / n:.2f} / 5")
    show("忠实率", f"{sum(r['faithful'] for r in results) / n:.0%}")
    show("检索命中率", f"{sum(r['hit'] for r in results) / n:.0%}")
    print("\n接入 LangSmith 后可改用：langsmith.Client().evaluate(target_fn, data='数据集名', evaluators=[...])，结果自动入库并可做版本对比。")


if __name__ == "__main__":
    main()
