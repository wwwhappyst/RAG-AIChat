"""清洗原始行数据，并转换成项目内部统一的文档结构。

这个文件最重要的设计思想：

- `questions` 是“检索别名”，帮助系统匹配用户的不同问法。
- `content` 是“回答依据”，最终生成答案时应该优先使用正文内容。

例如，一行 Excel 里可能有 5 个用 `||` 分隔的相似问题。
这些问题很适合提高召回率，但生成答案时应该映射回同一条原始正文。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    """把 Excel 单元格里的原始值转换成干净字符串。

    Excel 单元格可能是 None、数字、类似 NaN 的值，或者带有混乱空白的文本。
    做 RAG 索引前先统一清洗，可以减少后续检索时的脏数据干扰。
    """

    if value is None:
        return ""

    text = str(value).strip()
    if text.lower() == "nan":
        return ""

    # 把连续空白压缩成一个空格。这样既能保留中文和英文的可读性，
    # 又能去掉 Excel 换行、制表符、多余空格带来的噪声。
    return WHITESPACE_RE.sub(" ", text)


def split_questions(raw_questions: Any) -> list[str]:
    """把 Excel 的 `questions` 单元格拆成问题别名列表。

    源数据使用 `||` 分隔多个等价问题。
    保留为列表后，后续 FAISS 或 BM25 可以按需分别索引每个问题别名。
    """

    text = normalize_text(raw_questions)
    if not text:
        return []

    questions = [normalize_text(part) for part in text.split("||")]
    return [question for question in questions if question]


def build_doc_id(source_type: str, row_number: int, content: str) -> str:
    """生成一个可读、相对稳定的文档编号。

    行号方便人工排查；短 hash 可以降低不同来源、相同行号产生冲突的概率。
    """

    digest = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
    return f"{source_type}_{row_number:06d}_{digest}"


def build_knowledge_record(
    *,
    raw_row: dict[str, Any],
    source_type: str,
    source_file: Path,
    row_index: int,
) -> dict[str, Any] | None:
    """把一行原始 Excel 数据转换成一条标准 RAG 知识记录。

    第一版设计：
    - 一行 Excel 数据对应一个父文档；
    - 一个父文档暂时只有一个 chunk；
    - 后续如果正文变长，再由 `chunker.py` 进一步切分。
    """

    questions = split_questions(raw_row.get("questions"))
    content = normalize_text(raw_row.get("content"))
    image = normalize_text(raw_row.get("image"))

    # 空行既不能帮助检索，也不能作为答案依据，所以直接跳过。
    if not questions and not content:
        return None

    row_number = row_index + 1
    doc_id = build_doc_id(source_type, row_number, content or " ".join(questions))
    chunk_id = f"{doc_id}_chunk_000"

    # 短标题主要用于前端展示。对这两个数据集来说，第一个问题通常最适合当标题。
    title = questions[0] if questions else content[:80]

    retrieval_text_parts = []
    if questions:
        retrieval_text_parts.append("问题别名：" + "；".join(questions))
    if content:
        retrieval_text_parts.append("正文：" + content)

    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "parent_id": doc_id,
        "source_type": source_type,
        "source_file": source_file.name,
        "row_index": row_index,
        "title": title[:120],
        "questions": questions,
        "content": content,
        "image": image,
        # `retrieval_text` 是给后续 embedding / BM25 索引用的文本。
        # 它不是最终回答的全部上下文；最终答案仍应引用 `content` 并保留来源信息。
        "retrieval_text": "\n".join(retrieval_text_parts),
        "metadata": {
            "question_count": len(questions),
            "content_length": len(content),
            "has_image": bool(image),
        },
    }


def build_records_from_rows(
    *,
    rows: list[dict[str, Any]],
    source_type: str,
    source_file: Path,
) -> list[dict[str, Any]]:
    """把一个来源文件中的所有行转换成标准知识记录。"""

    records: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(rows):
        record = build_knowledge_record(
            raw_row=raw_row,
            source_type=source_type,
            source_file=source_file,
            row_index=row_index,
        )
        if record is not None:
            records.append(record)
    return records
