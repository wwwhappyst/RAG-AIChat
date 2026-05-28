"""为 RAG 系统构建标准化 JSONL 知识库。

在 ai_tutor_rag 目录下运行：

    python build_knowledge_base.py

这个脚本刻意保持简洁，只负责串联流程。
真正的读取逻辑在 `src/data_loader.py`，清洗逻辑在 `src/cleaner.py`。
这样做可以让每个文件的职责更清楚，也方便你逐个模块学习。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from config import (
    KNOWLEDGE_BASE_JSONL,
    KNOWLEDGE_BASE_STATS_JSON,
    PROCESSED_DATA_DIR,
    VIDEO_EXCEL_PATH,
    WECHAT_QA_EXCEL_PATH,
)
from src.cleaner import build_records_from_rows
from src.data_loader import read_excel_rows


def write_jsonl(records: list[dict], output_path: Path) -> None:
    """把记录写成 JSON Lines 格式。

    JSONL 的意思是“一行一个 JSON 对象”。
    它很适合 RAG 数据，因为后续可以逐行读取，不必一次加载一个巨大的 JSON 数组。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_stats(records: list[dict]) -> dict:
    """生成一个小型统计报告，用于验证数据质量和辅助学习。"""

    source_counts = Counter(record["source_type"] for record in records)
    question_counts = [record["metadata"]["question_count"] for record in records]
    content_lengths = [record["metadata"]["content_length"] for record in records]

    return {
        "total_records": len(records),
        "source_counts": dict(source_counts),
        "total_question_aliases": sum(question_counts),
        "avg_question_aliases_per_record": round(
            sum(question_counts) / len(question_counts), 2
        )
        if question_counts
        else 0,
        "avg_content_length": round(sum(content_lengths) / len(content_lengths), 2)
        if content_lengths
        else 0,
        "max_content_length": max(content_lengths) if content_lengths else 0,
    }


def main() -> None:
    """从两个源 Excel 文件构建清洗后的标准知识库。"""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    sources = [
        {
            "source_type": "video",
            "path": VIDEO_EXCEL_PATH,
        },
        {
            "source_type": "wechat_qa",
            "path": WECHAT_QA_EXCEL_PATH,
        },
    ]

    all_records: list[dict] = []
    for source in sources:
        rows = read_excel_rows(source["path"])
        records = build_records_from_rows(
            rows=rows,
            source_type=source["source_type"],
            source_file=source["path"],
        )
        all_records.extend(records)

    write_jsonl(all_records, KNOWLEDGE_BASE_JSONL)

    stats = build_stats(all_records)
    KNOWLEDGE_BASE_STATS_JSON.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"知识库记录数: {stats['total_records']}")
    print(f"来源分布: {stats['source_counts']}")
    print(f"问题别名总数: {stats['total_question_aliases']}")
    print(f"输出文件: {KNOWLEDGE_BASE_JSONL}")
    print(f"统计文件: {KNOWLEDGE_BASE_STATS_JSON}")


if __name__ == "__main__":
    main()
