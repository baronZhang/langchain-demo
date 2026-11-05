"""向量模型工厂 + 一个零依赖的本地哈希向量实现。

- EMBEDDING_PROVIDER=openai：走任意 OpenAI 兼容的 /embeddings 接口（生产推荐）。
- EMBEDDING_PROVIDER=local ：`HashingEmbeddings`，离线可跑、无需下载模型。
  它本质是“词袋 + 哈希技巧”，只能捕捉字面重叠、不懂同义词——
  正好可以拿来说明“为什么生产环境必须用真正的语义向量模型”。
"""

from __future__ import annotations

import hashlib
import math
import re

from langchain.embeddings import Embeddings, init_embeddings

from langchain_demo.settings import get_settings

_ASCII_WORD = re.compile(r"[a-z0-9_]+")
_CJK_RUN = re.compile(r"[一-鿿]+")


class HashingEmbeddings(Embeddings):
    """字符 n-gram 哈希向量（教学用）。

    实现 LangChain 的 `Embeddings` 接口只需要两个方法：
      - `embed_documents(texts)`：索引阶段批量调用
      - `embed_query(text)`     ：检索阶段对用户问题调用
    任何向量库（InMemoryVectorStore / Chroma / Milvus / PGVector ...）都只依赖这个接口，
    所以自研/私有化的向量模型也能这样无缝接入。
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    @staticmethod
    def _features(text: str) -> list[tuple[str, float]]:
        """把文本拆成 (特征, 权重)：英文按单词；中文二元组是主要信号，单字权重低，只提供“部分匹配”。"""
        text = text.lower()
        features = [(word, 1.0) for word in _ASCII_WORD.findall(text)]
        for run in _CJK_RUN.findall(text):
            features += [(char, 0.3) for char in run]
            features += [(run[i : i + 2], 1.0) for i in range(len(run) - 1)]
        return features

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feature, weight in self._features(text):
            h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
            # 带符号哈希：哈希冲突时正负相抵，而不是一味累加，减小冲突带来的偏差
            vec[h % self.dim] += weight if (h >> 63) & 1 else -weight
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]  # L2 归一化后，点积 = 余弦相似度

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings() -> Embeddings:
    """按配置返回向量模型。"""
    s = get_settings()
    if s.embedding_provider == "openai":
        if not s.embedding_model:
            raise RuntimeError("EMBEDDING_PROVIDER=openai 时必须设置 EMBEDDING_MODEL")
        return init_embeddings(
            s.embedding_model,
            provider="openai",
            base_url=s.embedding_base_url,
            api_key=s.embedding_api_key,
            # 非 OpenAI 官方服务不接受 token id 数组输入，必须关掉 tiktoken 预切分
            check_embedding_ctx_length=False,
        )
    return HashingEmbeddings()
