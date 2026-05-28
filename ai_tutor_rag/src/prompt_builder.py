"""RAG 答案生成前的 prompt 构造模块。

在基础 RAG 链路中，prompt 是“检索”和“生成”之间的桥梁：

1. 检索模块从 FAISS、BM25 或混合检索找回可能相关的知识 chunk。
2. prompt_builder 把这些 chunk 整理成 LLM 容易阅读的证据材料。
3. LLM 只能基于这些证据回答，降低自由发挥导致的幻觉。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.vector_store import SearchResult


@dataclass
class Evidence:
    """传给 LLM 的一条召回证据。

    `rank` 表示召回顺序，`score` 表示当前检索模式下的排序分数，`content` 才是回答依据。
    保留 `questions` 是为了让学习者看到：命中的可能是问题别名，但回答仍应回到正文。
    `source_file` 和 `row_index` 用来生成更适合 LLM 阅读的证据位置，避免把内部 chunk 编号暴露给模型。
    """

    rank: int
    score: float
    chunk_id: str
    source_type: str
    source_file: str
    row_index: int | None
    title: str
    questions: list[str]
    content: str
    score_details: dict[str, Any]


def build_evidences(results: list[SearchResult]) -> list[Evidence]:
    """把检索原始结果转换成 prompt 构造更友好的证据结构。"""

    evidences: list[Evidence] = []
    for rank, result in enumerate(results, start=1):
        metadata = result.metadata
        questions = metadata.get("questions") or []
        if not isinstance(questions, list):
            questions = [str(questions)]

        evidences.append(
            Evidence(
                rank=rank,
                score=result.score,
                chunk_id=str(metadata.get("chunk_id", "")),
                source_type=str(metadata.get("source_type", "")),
                source_file=str(metadata.get("source_file", "")),
                row_index=_parse_row_index(metadata.get("row_index")),
                title=str(metadata.get("title", "")),
                questions=[str(question) for question in questions],
                content=str(metadata.get("content", "")).strip(),
                score_details=result.score_details,
            )
        )
    return evidences


def build_rag_prompt(
    *,
    question: str,
    evidences: list[Evidence],
    max_context_chars: int,
) -> str:
    """构造最终发送给 LLM 的 RAG prompt。

    为什么要限制上下文长度：
    - 召回内容过长会增加 token 成本。
    - 基础阶段先保留最相关的 top-k 证据，后续 rerank 阶段再做更精细筛选。
    - 截断发生在 prompt 构造层，检索 metadata 不会被破坏。
    """

    if max_context_chars <= 0:
        raise ValueError("max_context_chars 必须大于 0")

    context = _format_evidence_context(evidences)
    if len(context) > max_context_chars:
        context = context[:max_context_chars].rstrip() + "\n\n[上下文因长度限制已截断]"

    return f"""你是一个面向 RAG 初学者的 AI 助教。请严格基于【召回证据】回答【用户问题】。

【回答要求】
1. 简短回答，字数控制在50~80字内。
2. 只能使用召回证据中的信息；证据不足时，请明确说“根据当前知识库证据不足以回答”。
3. 如果引用了某条证据，请在答案的最后，告知用户"证据来源"。
4. 不要编造课程、工具、步骤或结论。

【用户问题】
{question}

【召回证据】
{context}

【最终回答】
"""


def build_prompt_debug_info(evidences: list[Evidence]) -> list[dict[str, Any]]:
    """生成可展示的证据摘要，后续 Streamlit 前端可以直接复用。"""

    return [
        {
            "rank": evidence.rank,
            "score": round(evidence.score, 4),
            "score_details": _round_score_details(evidence.score_details),
            "chunk_id": evidence.chunk_id,
            "source_type": evidence.source_type,
            "title": evidence.title,
            "content_preview": evidence.content[:120],
        }
        for evidence in evidences
    ]


def _format_evidence_context(evidences: list[Evidence]) -> str:
    """把多条证据整理成人类和 LLM 都容易阅读的文本块。"""

    if not evidences:
        return "未召回到任何证据。"

    blocks: list[str] = []
    for evidence in evidences:
        question_preview = "；".join(evidence.questions[:3])
        source_position = _format_source_position(evidence)
        blocks.append(
            "\n".join(
                [
                    f"[证据{evidence.rank}]",
                    f"排序分数: {evidence.score:.4f}",
                    f"证据来源: {source_position}",
                    f"问题别名: {question_preview}",
                    f"正文: {evidence.content}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _parse_row_index(value: Any) -> int | None:
    """把 metadata 中的行号转成整数，缺失或异常时返回 None。

    标准知识库里的 `row_index` 从 0 开始；展示给 LLM 时会转换成从 1 开始的自然行号。
    """

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_source_position(evidence: Evidence) -> str:
    """把内部 chunk 位置转换成更贴近原始资料的人类可读出处。"""

    if evidence.source_file and evidence.row_index is not None:
        return f"{evidence.source_file} 文件的第 {evidence.row_index + 1} 行"
    if evidence.source_file:
        return f"{evidence.source_file} 文件"
    return evidence.chunk_id or "未知位置"


def _round_score_details(score_details: dict[str, Any]) -> dict[str, Any]:
    """把分数明细整理成前端和调试信息更容易阅读的格式。"""

    rounded: dict[str, Any] = {}
    for key, value in score_details.items():
        if isinstance(value, float):
            rounded[key] = round(value, 4)
        else:
            rounded[key] = value
    return rounded
