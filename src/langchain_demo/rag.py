"""RAG 公共组件：加载 Markdown 目录 → 按结构切分 → 构建向量库。

demo 02-01 会把每一步拆开讲解；其他 demo 和业务代码直接调用这里的函数即可。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from langchain_demo.embeddings import get_embeddings
from langchain_demo.settings import DATA_DIR

KNOWLEDGE_DIR = DATA_DIR / "knowledge"


class TextDirectoryLoader(BaseLoader):
    """最小可用的目录加载器，顺便演示 Loader 接口：实现 `lazy_load()` 逐个 yield Document 即可，
    `load()` / `alazy_load()` 由基类自动提供。

    背景：langchain-community（曾经的“百宝箱”，含上百种 loader）已于 2026 年停止维护，
    官方建议改用独立集成包或 MCP。纯文本 / Markdown 这类简单来源，自己实现 BaseLoader 最可控。
    """

    def __init__(self, directory: Path, glob: str = "**/*.md", encoding: str = "utf-8") -> None:
        self.directory = Path(directory)
        self.glob = glob
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        for path in sorted(self.directory.glob(self.glob)):
            # metadata 只保留文件名，方便在回答里展示引用来源
            yield Document(page_content=path.read_text(encoding=self.encoding), metadata={"source": path.name})


def load_markdown_dir(directory: Path = KNOWLEDGE_DIR) -> list[Document]:
    """加载目录下所有 Markdown 文件，每个文件一个 Document。"""
    return TextDirectoryLoader(directory).load()


def split_markdown(docs: Sequence[Document], chunk_size: int = 300, chunk_overlap: int = 50) -> list[Document]:
    """两级切分：先按 Markdown 标题切成“语义完整的小节”，过长的小节再按字符递归切。

    - 标题信息会写进 metadata（h1/h2/h3），可用于过滤和展示；
    - strip_headers=False 让标题保留在正文里，检索时标题关键词也能命中。
    """
    header_splitter = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 中文文本要把中文标点加进分隔符，否则会在句子中间硬切
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    sections: list[Document] = []
    for doc in docs:
        for section in header_splitter.split_text(doc.page_content):
            section.metadata = {**doc.metadata, **section.metadata}
            sections.append(section)
    return text_splitter.split_documents(sections)


def build_vector_store(directory: Path = KNOWLEDGE_DIR) -> InMemoryVectorStore:
    """一步构建内存向量库。生产环境把 InMemoryVectorStore 换成 Milvus/PGVector/Chroma 等即可，接口不变。"""
    store = InMemoryVectorStore(embedding=get_embeddings())
    store.add_documents(split_markdown(load_markdown_dir(directory)))
    return store


def format_docs(docs: Sequence[Document]) -> str:
    """把检索结果拼成带编号和出处的上下文，方便模型在回答里引用 [编号]。"""
    return "\n\n".join(f"[{i}]《{d.metadata.get('source', '未知')}》\n{d.page_content}" for i, d in enumerate(docs, 1))
