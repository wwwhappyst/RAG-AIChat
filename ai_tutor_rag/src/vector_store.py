"""FAISS 向量库读写与检索。

FAISS 只擅长保存和搜索“数字向量”，并不适合保存标题、正文、来源等可读信息。
所以本模块采用两个文件配合：

- `knowledge_base.index`：FAISS 向量索引，只保存向量和内部编号。
- `knowledge_base_metadata.jsonl`：metadata，每一行对应同位置的一条向量。

检索时，FAISS 返回向量位置，再用位置去 metadata 中找回 chunk_id、content、source_type 等信息。
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np


@dataclass
class SearchResult:
    """一次检索召回结果。

    第 5 阶段开始，FAISS 和 BM25 会共用这个轻量结构。
    `score` 的含义由检索器决定：FAISS 是向量相似度，BM25 是关键词相关性，
    混合检索则是合并后的排序分数。
    """

    score: float
    metadata: dict[str, Any]
    score_details: dict[str, Any] = field(default_factory=dict)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取第 1 阶段产出的标准 JSONL 知识库。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到知识库文件: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 第 {line_number} 行解析失败") from exc
    return records


def build_vector_text(record: dict[str, Any]) -> str:
    """决定一条知识记录进入 embedding 模型的文本。

    优先使用第 1 阶段生成的 `retrieval_text`，因为它同时包含问题别名和正文。
    如果未来某条记录缺少该字段，则退回到 content，保证建库流程仍能运行。
    """

    text = str(record.get("retrieval_text") or record.get("content") or "").strip()
    if not text:
        raise ValueError(f"记录缺少可向量化文本: {record.get('chunk_id')}")
    return text


def build_vector_metadata(
    *,
    record: dict[str, Any],
    vector_id: int,
    vector_text: str,
) -> dict[str, Any]:
    """提取检索结果展示和后续回答需要的 metadata。

    metadata 必须和 FAISS 向量保持同顺序：第 0 个向量对应第 0 行 metadata。
    这样 FAISS 返回内部编号后，我们才能准确回到原始知识 chunk。
    """

    return {
        "vector_id": vector_id,
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
        "retrieval_text": vector_text,
        "metadata": record.get("metadata", {}),
    }


class FaissVectorStore:
    """封装 FAISS 索引和与之对齐的 metadata。"""

    def __init__(self, *, index: Any, metadata: list[dict[str, Any]]) -> None:
        self.index = index
        self.metadata = metadata

    @classmethod
    def build(
        cls,
        *,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> "FaissVectorStore":
        """从 embedding 向量构建 FAISS 索引。"""

        if len(vectors) != len(metadata):
            raise ValueError("向量数量和 metadata 数量不一致")
        if not vectors:
            raise ValueError("向量列表为空，无法构建 FAISS 索引")

        matrix = np.asarray(vectors, dtype="float32")
        dimension = matrix.shape[1]

        # 先归一化再做内积检索，是 FAISS 中实现 cosine similarity 的常见方式。
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(dimension)
        index.add(matrix)

        return cls(index=index, metadata=metadata)

    @classmethod
    def load(cls, *, index_path: Path, metadata_path: Path) -> "FaissVectorStore":
        """从磁盘加载 FAISS 索引和 metadata。"""

        if not index_path.exists():
            raise FileNotFoundError(f"找不到 FAISS 索引文件: {index_path}")
        metadata = read_jsonl(metadata_path)
        index = _read_faiss_index(faiss, index_path)

        if index.ntotal != len(metadata):
            raise ValueError(
                f"索引向量数和 metadata 行数不一致: {index.ntotal} vs {len(metadata)}"
            )

        return cls(index=index, metadata=metadata)

    def save(
        self,
        *,
        index_path: Path,
        metadata_path: Path,
        manifest_path: Path,
        manifest: dict[str, Any],
    ) -> None:
        """把 FAISS 索引、metadata 和构建说明写入磁盘。"""

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        _write_faiss_index(faiss, self.index, index_path)
        with metadata_path.open("w", encoding="utf-8") as file:
            for item in self.metadata:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")

        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def search(self, *, query_vector: list[float], top_k: int) -> list[SearchResult]:
        """使用查询向量召回 top-k 条 metadata。"""

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        query = np.asarray([query_vector], dtype="float32")
        faiss.normalize_L2(query)

        scores, indexes = self.index.search(query, top_k)
        results: list[SearchResult] = []
        for score, index_id in zip(scores[0], indexes[0]):
            if index_id < 0:
                continue
            results.append(
                SearchResult(
                    score=float(score),
                    metadata=self.metadata[int(index_id)],
                )
            )
        return results

def _write_faiss_index(faiss: Any, index: Any, index_path: Path) -> None:
    """写入 FAISS 索引，并兼容 Windows 中文路径。

    部分 FAISS Windows wheel 的 C++ 文件接口对非 ASCII 路径支持不稳定。
    项目路径包含中文，所以先写到系统临时目录中的 ASCII 文件名，再由 Python 移动到目标路径。
    Python 的文件移动对中文路径支持更可靠。
    """

    with tempfile.TemporaryDirectory(prefix="ai_tutor_faiss_") as temp_dir:
        temp_path = Path(temp_dir) / "index.faiss"
        faiss.write_index(index, str(temp_path))
        shutil.move(str(temp_path), index_path)


def _read_faiss_index(faiss: Any, index_path: Path) -> Any:
    """读取 FAISS 索引，并兼容 Windows 中文路径。"""

    with tempfile.TemporaryDirectory(prefix="ai_tutor_faiss_") as temp_dir:
        temp_path = Path(temp_dir) / "index.faiss"
        shutil.copy2(index_path, temp_path)
        return faiss.read_index(str(temp_path))
