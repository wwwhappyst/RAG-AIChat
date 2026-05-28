"""命令行提问入口：运行 AI 助教 RAG 问答链路。

在 ai_tutor_rag 目录下运行：

    python ask.py "coze平台怎么注册并开始实践？"

这个脚本用于学习和验证：
- 先把用户问题向量化；
- 再从 FAISS、BM25 或混合检索召回相关 chunk；
- 然后把召回证据交给 LLM；
- 最后打印答案、证据和流程日志。
"""

from __future__ import annotations

import argparse
import json
import sys

from src.pipeline import BasicRagPipeline
from src.retriever import RETRIEVAL_MODE_LABELS


def parse_args() -> argparse.Namespace:
    """解析命令行参数，让同一个脚本可以测试不同问题和 top-k。"""

    parser = argparse.ArgumentParser(description="向 AI 助教 RAG 系统提问")
    parser.add_argument(
        "question",
        nargs="?",
        help="用户问题；如果不传，会进入交互式输入。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="最终交给 LLM 的证据数量。",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=list(RETRIEVAL_MODE_LABELS.keys()),
        default="hybrid",
        help="检索模式：faiss=向量检索，bm25=关键词检索，hybrid=混合检索。",
    )
    parser.add_argument(
        "--show-debug",
        action="store_true",
        help="打印更完整的调试信息，适合学习 RAG 链路。",
    )
    return parser.parse_args()


def main() -> int:
    """加载 RAG Pipeline，执行一次提问并打印结果。"""

    args = parse_args()
    question = args.question or input("请输入你的问题: ").strip()

    pipeline = BasicRagPipeline.from_config()
    result = pipeline.ask(
        question=question,
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
    )

    print("\n=== AI 助教回答 ===")
    print(result.answer)

    print("\n=== 召回证据 ===")
    for evidence in result.evidences:
        print(
            json.dumps(
                {
                    "rank": evidence.rank,
                    "score": round(evidence.score, 4),
                    "retrieval_mode": result.debug.get("retrieval_mode_label"),
                    "source_type": evidence.source_type,
                    "chunk_id": evidence.chunk_id,
                    "title": evidence.title,
                    "content_preview": evidence.content[:120],
                },
                ensure_ascii=False,
            )
        )

    if args.show_debug:
        print("\n=== 调试信息 ===")
        print(json.dumps(result.debug, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
