"""BM25 关键词索引的构建、保存、加载和检索。

BM25 和 FAISS 解决的是不同问题：

- FAISS 根据 embedding 向量做语义相似检索，适合“说法不同但意思相近”的问题。
- BM25 根据关键词匹配做稀疏检索，适合“工具名、模型名、专有名词”这类精确命中。

本模块只做本地分词和关键词打分，不调用 DashScope 或其他 AI API。
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from src.vector_store import SearchResult, read_jsonl


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_.+#-]+")


@dataclass
class Bm25BuildStats:
    """BM25 索引构建后的基础统计信息。"""

    record_count: int
    total_tokens: int
    average_tokens: float


def tokenize_text(text: str) -> list[str]:
    """把中文检索文本切成 BM25 可以使用的词项列表。

    BM25 不理解整段自然语言，它只看到一个个 token。
    对中文来说，必须先分词；对 `coze`、`qwen3.6-flash` 这类英文或模型名，
    再额外用正则保留连续片段，避免专有名词被切得过碎。
    """

    normalized = text.lower()
    tokens: list[str] = []

    for token in jieba.lcut(normalized):
        cleaned = token.strip()
        if _is_valid_token(cleaned):
            tokens.append(cleaned)

    # BM25 需要保留词频，所以不能把 jieba 分词结果整体去重。
    # 正则补充只用于保留技术名词原片段，避免把同一个片段额外重复加权。
    seen = set(tokens)
    for token in TOKEN_PATTERN.findall(normalized):
        cleaned = token.strip()
        if _is_valid_token(cleaned) and cleaned not in seen:
            tokens.append(cleaned)
            seen.add(cleaned)

    return tokens


def build_bm25_text(record: dict[str, Any]) -> str:
    """决定一条知识记录进入 BM25 索引的文本。

    这里优先使用 `retrieval_text`，因为它同时包含问题别名和正文。
    这样用户提到某个问题别名时，也能回到对应的 `content` 作为回答依据。
    """

    text = str(record.get("retrieval_text") or record.get("content") or "").strip()
    if not text:
        raise ValueError(f"记录缺少可检索文本: {record.get('chunk_id')}")
    return text


def build_bm25_metadata(record: dict[str, Any], bm25_id: int) -> dict[str, Any]:
    """提取 BM25 召回后需要展示和传给 prompt 的 metadata。

    metadata 字段尽量和 FAISS metadata 保持一致。
    这样上层 pipeline 不需要关心结果来自向量检索还是关键词检索。
    """

    return {
        "bm25_id": bm25_id,
        "doc_id": record.get("doc_id", ""),
        "chunk_id": record.get("chunk_id", ""),
        "parent_id": record.get("parent_id", ""),
        "source_type": record.get("source_type", ""),
        "source_file": record.get("source_file", ""),
        "row_index": record.get("row_index"),
        "title": record.get("title", ""),
        "questions": record.get("questions", []),
        "content": record.get("content", ""),
        "image": record.get("image", ""),
        "retrieval_text": build_bm25_text(record),
        "metadata": record.get("metadata", {}),
    }


class Bm25Store:
    """封装 BM25 索引和与之对齐的 metadata。"""

    def __init__(
        self,
        *,
        tokenized_corpus: list[list[str]],
        metadata: list[dict[str, Any]],
    ) -> None:
        if len(tokenized_corpus) != len(metadata):
            raise ValueError("BM25 语料数量和 metadata 数量不一致")
        if not tokenized_corpus:
            raise ValueError("BM25 语料为空，无法构建索引")

        self.tokenized_corpus = tokenized_corpus
        self.metadata = metadata
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.warm_up_tokenizer()

    @classmethod
    def build_from_records(cls, records: list[dict[str, Any]]) -> "Bm25Store":
        """从标准知识库记录构建 BM25 索引。"""

        tokenized_corpus: list[list[str]] = []
        metadata: list[dict[str, Any]] = []

        for bm25_id, record in enumerate(records):
            text = build_bm25_text(record)
            tokens = tokenize_text(text)
            if not tokens:
                raise ValueError(f"记录分词结果为空: {record.get('chunk_id')}")
            tokenized_corpus.append(tokens)
            metadata.append(build_bm25_metadata(record, bm25_id))

        return cls(tokenized_corpus=tokenized_corpus, metadata=metadata)

    @classmethod
    def build_from_jsonl(cls, jsonl_path: Path) -> "Bm25Store":
        """从第 1 阶段生成的标准 JSONL 文件构建 BM25 索引。"""

        return cls.build_from_records(read_jsonl(jsonl_path))

    @classmethod
    def load(cls, index_path: Path) -> "Bm25Store":
        """从磁盘加载 BM25 索引。

        文件中保存的是分词后的语料和 metadata。
        加载时重新创建 `BM25Okapi` 对象，避免不同版本 rank-bm25 的对象序列化差异。
        """

        if not index_path.exists():
            raise FileNotFoundError(f"找不到 BM25 索引文件: {index_path}")

        with index_path.open("rb") as file:
            payload = pickle.load(file)

        return cls(
            tokenized_corpus=payload["tokenized_corpus"],
            metadata=payload["metadata"],
        )

    def save(
        self,
        *,
        index_path: Path,
        manifest_path: Path,
        manifest: dict[str, Any],
    ) -> None:
        """把 BM25 索引和构建说明写入磁盘。"""

        index_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "tokenized_corpus": self.tokenized_corpus,
            "metadata": self.metadata,
        }
        with index_path.open("wb") as file:
            pickle.dump(payload, file)

        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def stats(self) -> Bm25BuildStats:
        """返回 BM25 索引的基础统计信息。"""

        token_counts = [len(tokens) for tokens in self.tokenized_corpus]
        total_tokens = sum(token_counts)
        return Bm25BuildStats(
            record_count=len(self.metadata),
            total_tokens=total_tokens,
            average_tokens=total_tokens / len(token_counts),
        )

    def search(self, *, query: str, top_k: int) -> list[SearchResult]:
        """使用 BM25 召回 top-k 条结果。

        BM25 分数是关键词相关性分数，和 FAISS 的向量相似度不是同一个量纲。
        混合检索时会在 retriever 层按各自排序做归一化，而不是直接相加原始分数。
        """

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        sorted_indexes = np.argsort(scores)[::-1]

        results: list[SearchResult] = []
        for index_id in sorted_indexes:
            score = float(scores[int(index_id)])
            if score <= 0:
                break
            results.append(
                SearchResult(
                    score=score,
                    metadata=self.metadata[int(index_id)],
                )
            )
            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def warm_up_tokenizer() -> None:
        """预热 jieba 分词器，避免首次查询时把词典加载时间算进 BM25 检索。

        jieba 默认采用懒加载：第一次调用 `jieba.lcut` 时才加载词典，
        在 Streamlit 页面里这会让第一次 BM25 检索看起来突然变慢。
        加载 BM25Store 时先执行一次很短的分词，可以把冷启动成本挪到 Pipeline 初始化阶段。
        """

        jieba.lcut("BM25 分词预热 coze qwen")


def _is_valid_token(token: str) -> bool:
    """过滤空白和纯标点，保留中文、英文、数字和常见技术符号。"""

    if not token:
        return False
    return any(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in token)
