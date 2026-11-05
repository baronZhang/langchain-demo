"""
Demo 02-01 · RAG 索引流水线：Load → Split → Embed → Store → Retrieve
===================================================================
知识点
  1. Document：page_content + metadata，是 RAG 全链路流转的数据结构
  2. Loader：BaseLoader 接口（实现 lazy_load 即可）；脚手架自带 TextDirectoryLoader，
     PDF/网页/数据库等来源用对应的独立集成包（langchain-community 已停止维护）
  3. Splitter：MarkdownHeaderTextSplitter（按结构切）+ RecursiveCharacterTextSplitter（按长度切）
  4. Embeddings：统一接口 embed_documents / embed_query
  5. VectorStore：增删查、相似度分数、MMR 多样性检索、元数据过滤、持久化
  6. Retriever：vector_store.as_retriever() —— 检索器也是 Runnable，可以直接接进 LCEL
对应文档：docs/03-详设篇/03-RAG检索增强.md
运行：uv run demos/02_rag/01_indexing.py
"""

from langchain_core.vectorstores import InMemoryVectorStore

from langchain_demo import RUNTIME_DIR, get_embeddings
from langchain_demo.console import banner, clip, section, show
from langchain_demo.rag import load_markdown_dir, split_markdown


def main() -> None:
    banner("Demo 02-01 · RAG 索引流水线", "Load → Split → Embed → Store → Retrieve（本 demo 不调用 LLM）")

    section("1. Load：加载文档")
    docs = load_markdown_dir()
    show("文档数", len(docs))
    show("第一个 Document 的 metadata", docs[0].metadata)

    section("2. Split：先按标题切，再按长度切")
    chunks = split_markdown(docs, chunk_size=300, chunk_overlap=50)
    show("切分后 chunk 数", len(chunks))
    show("示例 chunk metadata（标题层级进入了 metadata）", chunks[3].metadata)
    show("示例 chunk 内容", clip(chunks[3].page_content, 120))

    section("3. Embed：文本 → 向量")
    embeddings = get_embeddings()
    vector = embeddings.embed_query("年假有几天")
    show("向量模型", type(embeddings).__name__)
    show("向量维度 / 前 5 维", f"{len(vector)} / {[round(v, 3) for v in vector[:5]]}")

    section("4. Store：写入向量库")
    store = InMemoryVectorStore(embedding=embeddings)
    ids = store.add_documents(chunks)  # 内部会批量调用 embed_documents
    show("写入条数", len(ids))

    section("5. 检索：相似度 / 分数 / MMR / 元数据过滤")
    query = "去上海出差住宿能报多少钱"
    for doc, score in store.similarity_search_with_score(query, k=3):
        show(f"score={score:.3f}", f"《{doc.metadata['source']}》{clip(doc.page_content, 60)}")

    # MMR（最大边际相关）：先取 fetch_k 个候选，再在“相关性”和“彼此差异性”之间权衡，避免返回一堆重复片段
    mmr_docs = store.max_marginal_relevance_search(query, k=3, fetch_k=10, lambda_mult=0.5)
    show("MMR 结果来源", [d.metadata["source"] for d in mmr_docs])

    # InMemoryVectorStore 的 filter 是一个函数；Milvus/PGVector 等则是各自的表达式语法
    only_it = store.similarity_search("密码", k=2, filter=lambda d: d.metadata["source"] == "it_security.md")
    show("过滤后仅 IT 规范", [clip(d.page_content, 40) for d in only_it])

    section("6. 持久化与 Retriever")
    path = RUNTIME_DIR / "kb_index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    store.dump(str(path))
    restored = InMemoryVectorStore.load(str(path), embedding=embeddings)
    show("持久化文件", path.relative_to(RUNTIME_DIR.parent))

    retriever = restored.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 8})
    results = retriever.batch(["远程办公要求", "企业版有什么支持"])  # Retriever 是 Runnable：invoke/batch/stream 都可以
    for q, hits in zip(["远程办公要求", "企业版有什么支持"], results, strict=True):
        show(q, [clip(d.page_content, 40) for d in hits])

    print("\n提示：本地 HashingEmbeddings 只做字面匹配，换成真正的语义向量模型（.env 里 EMBEDDING_PROVIDER=openai）效果会好很多。")


if __name__ == "__main__":
    main()
