"""统一检索器：把 FAISS、BM25 和混合检索收拢到同一个入口。

Pipeline 不应该直接塞满各种检索细节。
第 5 阶段增加 BM25 后，用这个模块统一处理“选哪种检索模式”和“结果如何合并去重”。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from src.bm25_store import Bm25Store
from src.embeddings import DashScopeEmbeddingClient
from src.vector_store import FaissVectorStore, SearchResult


RetrievalMode = Literal["faiss", "bm25", "hybrid"]


RETRIEVAL_MODE_LABELS: dict[RetrievalMode, str] = {
    "faiss": "FAISS 向量检索",
    "bm25": "BM25 关键词检索",
    "hybrid": "混合检索",
}


DEFAULT_CANDIDATE_MULTIPLIER = 5

# 第 5 阶段教学版混合检索权重。
# FAISS 负责语义相似，BM25 负责关键词命中；这里略偏向语义检索，
# 同时保留较高的 BM25 权重，方便专有名词和工具名被精确召回。
FAISS_HYBRID_WEIGHT = 0.55
BM25_HYBRID_WEIGHT = 0.45


@dataclass
class RetrievalOutput:
    """一次检索的结果和可解释调试信息。"""

    results: list[SearchResult]
    debug: dict[str, object]


class HybridRetriever:
    """统一管理三种检索模式。

    - `faiss`：需要先调用 embedding 模型生成查询向量。
    - `bm25`：只在本地做分词和关键词打分，不调用 AI API。
    - `hybrid`：分别召回 FAISS 和 BM25 候选，再做归一化加权融合。
    """

    def __init__(
        self,
        *,
        embedding_client: DashScopeEmbeddingClient,
        vector_store: FaissVectorStore,
        bm25_store: Bm25Store,
    ) -> None:
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    def retrieve(
        self,
        *,
        question: str,
        top_k: int,
        mode: RetrievalMode,
        source_types: list[str] | None = None,
    ) -> RetrievalOutput:
        """按指定模式召回证据候选。

        `source_types` 是前端传入的知识库范围筛选条件。
        因为 FAISS 和 BM25 都是全库索引，所以这里先多召回候选，再按 metadata 过滤来源。
        """

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if mode not in RETRIEVAL_MODE_LABELS:
            raise ValueError(f"不支持的检索模式: {mode}")

        search_top_k = self._get_search_top_k(
            answer_top_k=top_k,
            source_types=source_types,
            mode=mode,
        )

        if mode == "faiss":
            faiss_results, retrieval_timings = self._search_faiss(
                question=question,
                top_k=search_top_k,
            )
            filter_start = perf_counter()
            results = self._filter_by_source_type(
                search_results=faiss_results,
                source_types=source_types,
            )[:top_k]
            retrieval_timings.append(
                _timing_event(
                    "来源过滤与截取 top-k",
                    "本地计算",
                    filter_start,
                    "按 metadata.source_type 过滤候选，并保留最终证据数量。",
                )
            )
            return RetrievalOutput(
                results=results,
                debug={
                    "retrieval_mode": mode,
                    "retrieval_mode_label": RETRIEVAL_MODE_LABELS[mode],
                    "faiss_search_top_k": search_top_k,
                    "bm25_search_top_k": 0,
                    "faiss_candidate_count": len(faiss_results),
                    "bm25_candidate_count": 0,
                    "merged_candidate_count": len(results),
                    "retrieval_timings": retrieval_timings,
                },
            )

        if mode == "bm25":
            bm25_results, retrieval_timings = self._search_bm25(
                question=question,
                top_k=search_top_k,
            )
            filter_start = perf_counter()
            results = self._filter_by_source_type(
                search_results=bm25_results,
                source_types=source_types,
            )[:top_k]
            retrieval_timings.append(
                _timing_event(
                    "来源过滤与截取 top-k",
                    "本地计算",
                    filter_start,
                    "按 metadata.source_type 过滤候选，并保留最终证据数量。",
                )
            )
            return RetrievalOutput(
                results=results,
                debug={
                    "retrieval_mode": mode,
                    "retrieval_mode_label": RETRIEVAL_MODE_LABELS[mode],
                    "faiss_search_top_k": 0,
                    "bm25_search_top_k": search_top_k,
                    "faiss_candidate_count": 0,
                    "bm25_candidate_count": len(bm25_results),
                    "merged_candidate_count": len(results),
                    "retrieval_timings": retrieval_timings,
                },
            )

        # 混合检索里，BM25 是纯本地计算，不依赖查询向量。
        # 因此可以在等待 DashScope embedding 的同时先跑 BM25，减少串行等待。
        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(
                self._search_bm25,
                question=question,
                top_k=search_top_k,
            )
            faiss_results, retrieval_timings = self._search_faiss(
                question=question,
                top_k=search_top_k,
            )
            bm25_results, bm25_timings = bm25_future.result()
            retrieval_timings.extend(bm25_timings)

        merge_start = perf_counter()
        merged_results = self._merge_hybrid_results(
            faiss_results=faiss_results,
            bm25_results=bm25_results,
        )
        retrieval_timings.append(
            _timing_event(
                "混合分数归一化与合并",
                "本地计算",
                merge_start,
                "对两路候选做 min-max 归一化、加权求和，并按 chunk_id 去重排序。",
            )
        )
        filter_start = perf_counter()
        results = self._filter_by_source_type(
            search_results=merged_results,
            source_types=source_types,
        )[:top_k]
        retrieval_timings.append(
            _timing_event(
                "来源过滤与截取 top-k",
                "本地计算",
                filter_start,
                "按 metadata.source_type 过滤候选，并保留最终证据数量。",
            )
        )
        return RetrievalOutput(
            results=results,
            debug={
                "retrieval_mode": mode,
                "retrieval_mode_label": RETRIEVAL_MODE_LABELS[mode],
                "faiss_search_top_k": search_top_k,
                "bm25_search_top_k": search_top_k,
                "faiss_candidate_count": len(faiss_results),
                "bm25_candidate_count": len(bm25_results),
                "merged_candidate_count": len(results),
                "hybrid_strategy": "并行执行 BM25 本地检索和 FAISS 查询向量化，再对两路候选分数做 min-max 归一化、加权求和，并用 chunk_id 去重",
                "hybrid_parallel_bm25": True,
                "hybrid_faiss_weight": FAISS_HYBRID_WEIGHT,
                "hybrid_bm25_weight": BM25_HYBRID_WEIGHT,
                "retrieval_timings": retrieval_timings,
            },
        )

    def _search_faiss(
        self,
        *,
        question: str,
        top_k: int,
    ) -> tuple[list[SearchResult], list[dict[str, Any]]]:
        """把用户问题向量化后交给 FAISS 检索。"""

        embedding_start = perf_counter()
        query_vector = self.embedding_client.embed_query(question)
        cache_hit = self.embedding_client.last_query_cache_hit
        embedding_kind = "本地缓存" if cache_hit else "DashScope embedding 调用"
        embedding_description = (
            "命中本地查询向量缓存，直接复用 768 维 query vector。"
            if cache_hit
            else "把用户问题转换成 768 维向量，供 FAISS 做语义检索。"
        )
        timings = [
            _timing_event(
                "查询向量化",
                embedding_kind,
                embedding_start,
                embedding_description,
            )
        ]

        faiss_start = perf_counter()
        results = self.vector_store.search(query_vector=query_vector, top_k=top_k)
        timings.append(
            _timing_event(
                "FAISS 向量检索",
                "本地计算",
                faiss_start,
                "在本地 FAISS 索引中按向量相似度召回候选。",
            )
        )
        return results, timings

    def _search_bm25(
        self,
        *,
        question: str,
        top_k: int,
    ) -> tuple[list[SearchResult], list[dict[str, Any]]]:
        """执行 BM25 关键词检索，并返回可解释耗时。"""

        bm25_start = perf_counter()
        bm25_results = self.bm25_store.search(query=question, top_k=top_k)
        timings = [
            _timing_event(
                "BM25 关键词检索",
                "本地计算",
                bm25_start,
                "对用户问题分词，并使用 BM25Okapi 计算关键词相关性。",
            )
        ]
        return bm25_results, timings

    def _get_search_top_k(
        self,
        *,
        answer_top_k: int,
        source_types: list[str] | None,
        mode: RetrievalMode,
    ) -> int:
        """决定底层索引先召回多少候选。

        如果用户只选择一个知识库来源，过滤后候选会变少，所以需要先多取一些。
        混合检索也受益于更大的候选池，两个索引才有机会互相补充。
        """

        if mode != "hybrid" and not source_types:
            return answer_top_k
        return min(
            len(self.vector_store.metadata),
            max(answer_top_k, answer_top_k * DEFAULT_CANDIDATE_MULTIPLIER),
        )

    @staticmethod
    def _filter_by_source_type(
        *,
        search_results: list[SearchResult],
        source_types: list[str] | None,
    ) -> list[SearchResult]:
        """根据 metadata.source_type 过滤召回结果。"""

        if not source_types:
            return search_results

        allowed = set(source_types)
        return [
            result
            for result in search_results
            if result.metadata.get("source_type") in allowed
        ]

    @staticmethod
    def _merge_hybrid_results(
        *,
        faiss_results: list[SearchResult],
        bm25_results: list[SearchResult],
    ) -> list[SearchResult]:
        """合并 FAISS 和 BM25 结果。

        两类分数量纲不同，不能直接比较原始分数。
        这里先分别在本次候选池内做 min-max 归一化，再按权重求和。
        前端会展示原始分、归一分和最终分，帮助学习者观察排序依据。
        """

        faiss_scores = _score_map_by_chunk_id(faiss_results)
        bm25_scores = _score_map_by_chunk_id(bm25_results)
        faiss_norm_scores = _min_max_normalize(faiss_scores)
        bm25_norm_scores = _min_max_normalize(bm25_scores)

        faiss_ranks = _rank_map_by_chunk_id(faiss_results)
        bm25_ranks = _rank_map_by_chunk_id(bm25_results)
        result_by_chunk = _result_map_by_chunk_id(faiss_results)
        for chunk_id, result in _result_map_by_chunk_id(bm25_results).items():
            result_by_chunk.setdefault(chunk_id, result)

        merged: list[SearchResult] = []
        for chunk_id, result in result_by_chunk.items():
            faiss_norm = faiss_norm_scores.get(chunk_id, 0.0)
            bm25_norm = bm25_norm_scores.get(chunk_id, 0.0)
            final_score = (
                FAISS_HYBRID_WEIGHT * faiss_norm
                + BM25_HYBRID_WEIGHT * bm25_norm
            )
            merged.append(
                SearchResult(
                    score=final_score,
                    metadata=result.metadata,
                    score_details={
                        "faiss_raw": faiss_scores.get(chunk_id),
                        "bm25_raw": bm25_scores.get(chunk_id),
                        "faiss_norm": faiss_norm,
                        "bm25_norm": bm25_norm,
                        "faiss_rank": faiss_ranks.get(chunk_id),
                        "bm25_rank": bm25_ranks.get(chunk_id),
                        "faiss_weight": FAISS_HYBRID_WEIGHT,
                        "bm25_weight": BM25_HYBRID_WEIGHT,
                        "final_score": final_score,
                    },
                )
            )

        return sorted(
            merged,
            key=lambda result: result.score,
            reverse=True,
        )


def _score_map_by_chunk_id(results: list[SearchResult]) -> dict[str, float]:
    """把检索结果转成 chunk_id -> 原始分数，便于后续归一化。"""

    scores: dict[str, float] = {}
    for result in results:
        chunk_id = str(result.metadata.get("chunk_id") or "")
        if chunk_id and chunk_id not in scores:
            scores[chunk_id] = result.score
    return scores


def _rank_map_by_chunk_id(results: list[SearchResult]) -> dict[str, int]:
    """记录每条结果在单路检索中的原始排名。"""

    ranks: dict[str, int] = {}
    for rank, result in enumerate(results, start=1):
        chunk_id = str(result.metadata.get("chunk_id") or "")
        if chunk_id and chunk_id not in ranks:
            ranks[chunk_id] = rank
    return ranks


def _result_map_by_chunk_id(results: list[SearchResult]) -> dict[str, SearchResult]:
    """用 chunk_id 去重，保留第一次出现的检索结果。"""

    result_by_chunk: dict[str, SearchResult] = {}
    for result in results:
        chunk_id = str(result.metadata.get("chunk_id") or "")
        if chunk_id and chunk_id not in result_by_chunk:
            result_by_chunk[chunk_id] = result
    return result_by_chunk


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """把一组原始分数归一到 0~1。

    这里做的是“本次候选池内归一化”：
    - 最高分映射为 1；
    - 最低分映射为 0；
    - 如果只有一个候选或所有分数相同，就把已命中的结果记为 1。
    """

    if not scores:
        return {}

    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {chunk_id: 1.0 for chunk_id in scores}

    return {
        chunk_id: (score - min_score) / (max_score - min_score)
        for chunk_id, score in scores.items()
    }


def _timing_event(
    node: str,
    kind: str,
    start: float,
    description: str,
) -> dict[str, Any]:
    """生成一个流程耗时节点，统一前端展示格式。"""

    return {
        "node": node,
        "kind": kind,
        "elapsed_ms": round((perf_counter() - start) * 1000, 2),
        "description": description,
    }
