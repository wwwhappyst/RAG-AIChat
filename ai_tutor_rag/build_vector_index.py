"""构建第 2 阶段 FAISS 向量库。

在 ai_tutor_rag 目录下运行：

    python build_vector_index.py

建议首次学习时先跑小样本：

    python build_vector_index.py --limit 30

脚本流程：
1. 读取第 1 阶段产出的 knowledge_base.jsonl。
2. 使用配置中的 DashScope embedding 模型将 `retrieval_text` 转成向量。
3. 使用 FAISS 建立向量索引。
4. 分别保存 FAISS index、metadata 和 manifest。
5. 用一个测试问题做 top-k 召回验证。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any

from config import (
    EMBEDDING_BASE_URL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    KNOWLEDGE_BASE_JSONL,
    VECTOR_MANIFEST_PATH,
    VECTOR_METADATA_PATH,
)
from src.embeddings import DashScopeEmbeddingClient
from src.vector_store import (
    FaissVectorStore,
    build_vector_metadata,
    build_vector_text,
    read_jsonl,
)


DEFAULT_TEST_QUERY = "coze平台怎么注册并开始实践？"


def parse_args() -> argparse.Namespace:
    """解析命令行参数，方便全量建库和小样本验证共用同一个脚本。"""

    parser = argparse.ArgumentParser(description="构建 AI 助教 RAG 的 FAISS 向量库")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前 N 条记录，适合首次验证流程；不传则处理全量知识库。",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_TEST_QUERY,
        help="建库完成后用于验证召回效果的测试问题。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="测试召回时返回的结果数量。",
    )
    return parser.parse_args()


def build_index_records(limit: int | None) -> tuple[list[dict[str, Any]], list[str]]:
    """读取知识库，并抽取将要送入 embedding 的文本。"""

    records = read_jsonl(KNOWLEDGE_BASE_JSONL)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit 必须大于 0")
        records = records[:limit]

    vector_texts = [build_vector_text(record) for record in records]
    return records, vector_texts


def main() -> None:
    """执行完整的 embedding + FAISS 建库 + 查询验证流程。"""

    args = parse_args()
    records, vector_texts = build_index_records(args.limit)

    print("第 2 阶段：开始构建 FAISS 向量库")
    print(f"知识记录数: {len(records)}")
    print(f"Embedding 模型: {EMBEDDING_MODEL}")
    print(f"向量维度: {EMBEDDING_DIMENSION}")

    client = DashScopeEmbeddingClient(
        model=EMBEDDING_MODEL,
        dimension=EMBEDDING_DIMENSION,
        batch_size=EMBEDDING_BATCH_SIZE,
        base_url=EMBEDDING_BASE_URL,
    )

    embedding_result = client.embed_documents(vector_texts)

    metadata = [
        build_vector_metadata(
            record=record,
            vector_id=index,
            vector_text=vector_text,
        )
        for index, (record, vector_text) in enumerate(zip(records, vector_texts))
    ]

    store = FaissVectorStore.build(
        vectors=embedding_result.vectors,
        metadata=metadata,
    )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_jsonl": str(KNOWLEDGE_BASE_JSONL),
        "record_count": len(records),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "embedding_total_tokens": embedding_result.total_tokens,
        "faiss_metric": "IndexFlatIP + L2 normalize, approximate cosine similarity",
        "metadata_file": str(VECTOR_METADATA_PATH),
    }
    store.save(
        index_path=FAISS_INDEX_PATH,
        metadata_path=VECTOR_METADATA_PATH,
        manifest_path=VECTOR_MANIFEST_PATH,
        manifest=manifest,
    )

    print(f"FAISS 索引文件: {FAISS_INDEX_PATH}")
    print(f"Metadata 文件: {VECTOR_METADATA_PATH}")
    print(f"Manifest 文件: {VECTOR_MANIFEST_PATH}")
    print(f"Embedding token 用量: {embedding_result.total_tokens}")

    query_vector = client.embed_query(args.query)
    results = store.search(query_vector=query_vector, top_k=args.top_k)

    print("\n召回验证")
    print(f"测试问题: {args.query}")
    for rank, result in enumerate(results, start=1):
        item = result.metadata
        print(
            json.dumps(
                {
                    "rank": rank,
                    "score": round(result.score, 4),
                    "source_type": item["source_type"],
                    "chunk_id": item["chunk_id"],
                    "title": item["title"],
                    "content_preview": item["content"][:100],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
