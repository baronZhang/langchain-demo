"""
Demo 02-02 · 两段式 RAG 链（Retrieve → Generate）
================================================
知识点
  1. 用 LCEL 组装 RAG：{"context": retriever | format_docs, "question": 透传} | prompt | llm | parser
  2. 答案 + 引用来源一起返回（RunnableParallel + assign）
  3. 查询改写（Query Rewriting）：先把口语化问题改写成检索友好的查询，再检索
  4. 拒答：知识库里没有就说不知道（靠 prompt 约束 + 引用编号可校验）
  5. 流式输出答案
对应文档：docs/03-详设篇/03-RAG检索增强.md
运行：uv run demos/02_rag/02_rag_chain.py
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain_demo import get_chat_model
from langchain_demo.console import banner, section, show
from langchain_demo.rag import build_vector_store, format_docs

llm = get_chat_model()
retriever = build_vector_store().as_retriever(search_kwargs={"k": 4})

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是星河科技的员工助手。只能依据 <context> 中的资料回答，资料里没有就回答“资料中没有相关信息”。"
            "回答要简洁，并在相关句子后用 [编号] 标注引用。\n<context>\n{context}\n</context>",
        ),
        ("human", "{question}"),
    ]
)


def main() -> None:
    banner("Demo 02-02 · 两段式 RAG 链", "检索 → 拼上下文 → 生成，附引用来源")

    section("1. 基础 RAG 链")
    rag_chain = (
        # dict 会被转成 RunnableParallel：context 走检索+格式化，question 原样透传
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    show("问", "我入职 4 年了，一年有几天年假？没休完怎么办？")
    show("答", rag_chain.invoke("我入职 4 年了，一年有几天年假？没休完怎么办？"))

    section("2. 返回答案 + 来源文档")
    generate = (lambda x: {"context": format_docs(x["docs"]), "question": x["question"]}) | PROMPT | llm | StrOutputParser()
    rag_with_sources = RunnableParallel(docs=retriever, question=RunnablePassthrough()).assign(answer=generate)
    out = rag_with_sources.invoke("单笔 8000 元的报销要谁审批？多久到账？")
    show("答", out["answer"])
    show("来源", sorted({d.metadata["source"] for d in out["docs"]}))

    section("3. 查询改写：口语问题 → 检索查询")
    rewrite = (
        ChatPromptTemplate.from_template("把用户问题改写成用于检索公司制度文档的简短关键词查询，只输出查询本身：{q}")
        | get_chat_model(fast=True)
        | StrOutputParser()
    )
    question = "老板让我下周去广州见客户，晚上住酒店的话公司给报多少？"
    rewritten = rewrite.invoke({"q": question})
    show("改写后的查询", rewritten)
    # rewrite 需要 {"q": ...}，用 lambda 把字符串输入适配成 dict
    chain ={"context": (lambda q: {"q": q}) | rewrite | retriever | format_docs, "question": RunnablePassthrough()} | PROMPT | llm | StrOutputParser()
    show("答", chain.invoke(question))

    section("4. 拒答：知识库里没有的问题")
    show("答", rag_chain.invoke("公司食堂几点开饭？"))

    section("5. 流式输出")
    for piece in rag_chain.stream("试用期可以远程办公吗？"):
        print(piece, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
