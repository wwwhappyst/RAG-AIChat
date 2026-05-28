"""构建第 5 阶段 BM25 关键词索引。

运行方式：

    python build_bm25_index.py

这个脚本不会调用任何 AI API。
它只读取第 1 阶段标准化后的 `knowledge_base.jsonl`，对 `retrieval_text` 做中文分词，
然后把分词后的语料和 metadata 保存到 `indexes/bm25/`。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BM25_INDEX_PATH, BM25_MANIFEST_PATH, KNOWLEDGE_BASE_JSONL
from src.bm25_store import Bm25Store


def build_manifest(store: Bm25Store) -> dict[str, object]:
    """生成 BM25 索引说明文件，帮助学习者检查索引来源和规模。"""

    stats = store.stats()
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_jsonl": str(KNOWLEDGE_BASE_JSONL),
        "record_count": stats.record_count,
        "tokenizer": "jieba.lcut + 英文/数字/技术名词正则补充",
        "bm25_algorithm": "rank_bm25.BM25Okapi",
        "total_tokens": stats.total_tokens,
        "average_tokens": round(stats.average_tokens, 2),
        "index_file": str(BM25_INDEX_PATH),
        "mock_data": False,
        "uses_ai_api": False,
    }


def main() -> int:
    """执行 BM25 索引构建并打印关键结果。"""

    print("=== 构建 BM25 关键词索引 ===")
    print(f"知识库文件: {KNOWLEDGE_BASE_JSONL}")
    print("说明: BM25 是本地关键词算法，本脚本不会调用 embedding、LLM 或 rerank API。")

    store = Bm25Store.build_from_jsonl(KNOWLEDGE_BASE_JSONL)
    manifest = build_manifest(store)
    store.save(
        index_path=BM25_INDEX_PATH,
        manifest_path=BM25_MANIFEST_PATH,
        manifest=manifest,
    )

    print(f"BM25 索引文件: {BM25_INDEX_PATH}")
    print(f"BM25 manifest: {BM25_MANIFEST_PATH}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
