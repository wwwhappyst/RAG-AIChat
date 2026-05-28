"""RAG 问答链路。

这个模块把前面已经完成的能力串起来：

用户问题 -> 检索证据 -> prompt 构造 -> DashScope LLM 生成答案

第 5 阶段开始，检索证据可以选择 FAISS、BM25 或混合检索。
第 7 阶段加入本地 rerank：先多召回候选，再用本地模型重排序。
query 改写和父页面扩展暂时不进入当前链路。
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator

from config import (
    BM25_INDEX_PATH,
    DEFAULT_RETRIEVAL_TOP_K,
    EMBEDDING_BASE_URL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    LLM_BASE_URL,
    LLM_ENABLE_THINKING,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_STREAM,
    LLM_TEMPERATURE,
    MAX_CONTEXT_CHARS,
    QUERY_VECTOR_CACHE_PATH,
    RERANK_BATCH_SIZE,
    RERANK_CANDIDATE_MULTIPLIER,
    RERANK_DEVICE,
    RERANK_ENABLED,
    RERANK_MAX_CANDIDATES,
    RERANK_MAX_LENGTH,
    RERANK_MODEL_DIR,
    RERANK_MODEL_ID,
    VECTOR_METADATA_PATH,
)
from src.bm25_store import Bm25Store
from src.embeddings import DashScopeEmbeddingClient
from src.generator import DashScopeGenerator, GenerationResult
from src.prompt_builder import (
    Evidence,
    build_evidences,
    build_prompt_debug_info,
    build_rag_prompt,
)
from src.reranker import LocalReranker, RerankResult
from src.retriever import HybridRetriever, RetrievalMode
from src.vector_store import FaissVectorStore


@dataclass
class RagAnswer:
    """一次 RAG 问答的完整返回结果。

    `answer` 给用户阅读，`evidences` 和 `debug` 用来解释答案来自哪里。
    第 4 阶段做 Streamlit 前端时，可以把这些字段分别展示在不同区域。
    """

    question: str
    answer: str
    evidences: list[Evidence]
    debug: dict[str, Any]


@dataclass
class PreparedRagRequest:
    """已经完成检索和 prompt 构造、等待 LLM 生成的中间结果。"""

    question: str
    prompt: str
    evidences: list[Evidence]
    debug_base: dict[str, Any]
    timing_events: list[dict[str, Any]]
    pipeline_start: float


@dataclass
class RagStreamEvent:
    """流式问答事件。

    `progress` 表示某个 RAG 流程节点已经完成；
    `answer_delta` 是 LLM 新生成的一小段文本；
    `result` 在流结束时返回完整问答结果。
    """

    progress: dict[str, Any] | None = None
    prepared: PreparedRagRequest | None = None
    answer_delta: str = ""
    result: RagAnswer | None = None


class BasicRagPipeline:
    """AI 助教 RAG Pipeline。

    Pipeline 负责串联“检索”和“生成”，但不直接实现每一种检索算法。
    第 5 阶段把检索细节下沉到 `HybridRetriever`，后续加 rerank 时也更容易扩展。
    """

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        reranker: LocalReranker,
        generator: DashScopeGenerator,
        max_context_chars: int,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.max_context_chars = max_context_chars

    @classmethod
    def from_config(cls) -> "BasicRagPipeline":
        """按项目配置加载 embedding 客户端、FAISS 索引和生成模型。"""

        embedding_client = DashScopeEmbeddingClient(
            model=EMBEDDING_MODEL,
            dimension=EMBEDDING_DIMENSION,
            batch_size=EMBEDDING_BATCH_SIZE,
            base_url=EMBEDDING_BASE_URL,
            query_cache_path=QUERY_VECTOR_CACHE_PATH,
        )
        vector_store = FaissVectorStore.load(
            index_path=FAISS_INDEX_PATH,
            metadata_path=VECTOR_METADATA_PATH,
        )
        bm25_store = Bm25Store.load(index_path=BM25_INDEX_PATH)
        retriever = HybridRetriever(
            embedding_client=embedding_client,
            vector_store=vector_store,
            bm25_store=bm25_store,
        )
        reranker = LocalReranker(
            enabled=RERANK_ENABLED,
            model_id=RERANK_MODEL_ID,
            model_dir=RERANK_MODEL_DIR,
            batch_size=RERANK_BATCH_SIZE,
            max_length=RERANK_MAX_LENGTH,
            device=RERANK_DEVICE,
        )
        generator = DashScopeGenerator(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            base_url=LLM_BASE_URL,
            max_tokens=LLM_MAX_TOKENS,
            enable_thinking=LLM_ENABLE_THINKING,
            stream=LLM_STREAM,
        )
        return cls(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            max_context_chars=MAX_CONTEXT_CHARS,
        )

    def ask(
        self,
        question: str,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        source_types: list[str] | None = None,
        retrieval_mode: RetrievalMode = "hybrid",
        use_rerank: bool = True,
    ) -> RagAnswer:
        """执行一次完整问答。

        输入是用户原始问题；输出包括最终答案、召回证据和调试信息。
        这里特意保留每一步日志，是为了让学习者能观察 RAG 的真实工作过程。
        `source_types` 用于第 4 阶段前端的“知识库范围”筛选，例如只看课程视频或微信群问答。
        `retrieval_mode` 用于第 5 阶段选择 FAISS、BM25 或混合检索。
        """

        prepared = self._prepare(
            question=question,
            top_k=top_k,
            source_types=source_types,
            retrieval_mode=retrieval_mode,
            use_rerank=use_rerank,
        )
        generation_start = perf_counter()
        generation = self.generator.generate(prepared.prompt)
        return self._build_answer(
            prepared=prepared,
            generation=generation,
            generation_start=generation_start,
        )

    def ask_stream(
        self,
        question: str,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        source_types: list[str] | None = None,
        retrieval_mode: RetrievalMode = "hybrid",
        use_rerank: bool = True,
    ) -> Iterator[RagStreamEvent]:
        """执行一次流式问答。

        检索和 prompt 构造仍然先完成；进入 LLM 阶段后，逐步产出答案片段。
        最后一个事件会携带完整 `RagAnswer`，用于前端展示证据、日志和 token 用量。
        """

        prepared: PreparedRagRequest | None = None
        for event in self._prepare_stream(
            question=question,
            top_k=top_k,
            source_types=source_types,
            retrieval_mode=retrieval_mode,
            use_rerank=use_rerank,
        ):
            if event.prepared is not None:
                prepared = event.prepared
            else:
                yield event

        if prepared is None:
            raise RuntimeError("RAG 准备阶段结束，但没有生成中间结果")

        generation_start = perf_counter()
        yield RagStreamEvent(
            progress=_timing_event(
                "进入 LLM 流式生成",
                "DashScope LLM 流式调用",
                generation_start,
                "已完成检索和 Prompt 构造，开始向 DashScope 发起流式生成请求。",
            )
        )
        final_generation: GenerationResult | None = None
        for event in self.generator.generate_stream(prepared.prompt):
            if event.delta:
                yield RagStreamEvent(answer_delta=event.delta)
            if event.result is not None:
                final_generation = event.result
                for timing in event.result.timings:
                    yield RagStreamEvent(progress=timing)

        if final_generation is None:
            raise RuntimeError("DashScope LLM 流式返回结束，但没有生成最终结果")

        yield RagStreamEvent(
            result=self._build_answer(
                prepared=prepared,
                generation=final_generation,
                generation_start=generation_start,
            )
        )

    def _prepare_stream(
        self,
        *,
        question: str,
        top_k: int,
        source_types: list[str] | None,
        retrieval_mode: RetrievalMode,
        use_rerank: bool,
    ) -> Iterator[RagStreamEvent]:
        """带进度事件的准备阶段。

        Streamlit 前端会把这些事件展示成“已完成步骤 + 耗时”，
        让用户在最终答案出现前就能看到系统正在经历哪些 RAG 节点。
        """

        pipeline_start = perf_counter()
        timing_events: list[dict[str, Any]] = []

        input_start = perf_counter()
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("问题不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        normalized_source_types = self._normalize_source_types(source_types)
        timing = _timing_event(
            "参数校验与来源清洗",
            "本地计算",
            input_start,
            "清洗用户问题、检查 top-k，并把前端知识库范围转换成内部 source_type。",
        )
        timing_events.append(timing)
        yield RagStreamEvent(progress=timing)

        candidate_top_k = self._get_candidate_top_k(
            final_top_k=top_k,
            use_rerank=use_rerank,
        )
        yield RagStreamEvent(
            progress=_running_event(
                "检索阶段总耗时",
                "混合流程",
                "正在执行查询向量化、FAISS 检索、BM25 检索、混合合并和来源过滤。",
            )
        )
        retrieval_start = perf_counter()
        retrieval = self.retriever.retrieve(
            question=normalized_question,
            top_k=candidate_top_k,
            mode=retrieval_mode,
            source_types=normalized_source_types,
        )
        timing = _timing_event(
            "检索阶段总耗时",
            "混合流程",
            retrieval_start,
            "包含查询向量化、FAISS 检索、BM25 检索、混合合并和来源过滤。",
        )
        timing_events.append(timing)
        yield RagStreamEvent(progress=timing)

        retrieval_timings = retrieval.debug.get("retrieval_timings", [])
        if isinstance(retrieval_timings, list):
            for timing in retrieval_timings:
                if isinstance(timing, dict):
                    timing_events.append(timing)
                    yield RagStreamEvent(progress=timing)

        if use_rerank:
            yield RagStreamEvent(
                progress=_running_event(
                    "本地 Rerank 阶段",
                    "本地模型推理",
                    "正在加载或复用本地 rerank 模型，并对候选证据重新打分排序。",
                )
            )
        rerank_start = perf_counter()
        rerank_result = self._rerank_results(
            question=normalized_question,
            results=retrieval.results,
            top_k=top_k,
            use_rerank=use_rerank,
        )
        if use_rerank:
            timing = _timing_event(
                "本地 Rerank 阶段",
                "本地模型推理",
                rerank_start,
                "包含本地模型加载、query+chunk 相关性打分和最终 top-k 截取。",
            )
            timing_events.append(timing)
            yield RagStreamEvent(progress=timing)
        timing_events.extend(rerank_result.timings)
        for timing in rerank_result.timings:
            yield RagStreamEvent(progress=timing)

        evidence_start = perf_counter()
        evidences = build_evidences(rerank_result.results)
        timing = _timing_event(
            "证据对象整理",
            "本地计算",
            evidence_start,
            "把底层检索结果转换成 prompt 和前端都能使用的 Evidence 结构。",
        )
        timing_events.append(timing)
        yield RagStreamEvent(progress=timing)

        prompt_start = perf_counter()
        prompt = build_rag_prompt(
            question=normalized_question,
            evidences=evidences,
            max_context_chars=self.max_context_chars,
        )
        timing = _timing_event(
            "Prompt 构造",
            "本地计算",
            prompt_start,
            "把用户问题、召回证据和回答要求组装成最终发送给 LLM 的文本。",
        )
        timing_events.append(timing)
        yield RagStreamEvent(progress=timing)

        debug_base = {
            "pipeline": [
                f"1. 使用 {retrieval.debug.get('retrieval_mode_label')} 召回证据",
                "2. 使用本地 rerank 模型对候选证据重排序",
                "3. 对重排结果整理为证据列表",
                "4. 把用户问题、召回证据和回答要求组装成 prompt",
                "5. 流式调用 DashScope LLM 生成基于证据的答案",
            ],
            "embedding_model": EMBEDDING_MODEL,
            "embedding_used": retrieval_mode in {"faiss", "hybrid"},
            "llm_max_tokens": LLM_MAX_TOKENS,
            "llm_enable_thinking": LLM_ENABLE_THINKING,
            "llm_stream": LLM_STREAM,
            "use_rerank": use_rerank,
            "top_k": top_k,
            "source_types": normalized_source_types or ["video", "wechat_qa"],
            "retrieved_count": len(evidences),
            "evidence_summary": build_prompt_debug_info(evidences),
            "final_prompt": prompt,
            "final_prompt_chars": len(prompt),
            "rerank_candidate_top_k": candidate_top_k,
            **rerank_result.debug,
            **retrieval.debug,
        }

        yield RagStreamEvent(
            prepared=PreparedRagRequest(
                question=normalized_question,
                prompt=prompt,
                evidences=evidences,
                debug_base=debug_base,
                timing_events=timing_events,
                pipeline_start=pipeline_start,
            )
        )

    def _prepare(
        self,
        *,
        question: str,
        top_k: int,
        source_types: list[str] | None,
        retrieval_mode: RetrievalMode,
        use_rerank: bool,
    ) -> PreparedRagRequest:
        """执行 LLM 之前的准备阶段：参数清洗、检索、证据整理和 prompt 构造。"""

        pipeline_start = perf_counter()
        timing_events: list[dict[str, Any]] = []

        input_start = perf_counter()
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("问题不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        normalized_source_types = self._normalize_source_types(source_types)
        timing_events.append(
            _timing_event(
                "参数校验与来源清洗",
                "本地计算",
                input_start,
                "清洗用户问题、检查 top-k，并把前端知识库范围转换成内部 source_type。",
            )
        )

        candidate_top_k = self._get_candidate_top_k(
            final_top_k=top_k,
            use_rerank=use_rerank,
        )
        retrieval_start = perf_counter()
        retrieval = self.retriever.retrieve(
            question=normalized_question,
            top_k=candidate_top_k,
            mode=retrieval_mode,
            source_types=normalized_source_types,
        )
        timing_events.append(
            _timing_event(
                "检索阶段总耗时",
                "混合流程",
                retrieval_start,
                "包含查询向量化、FAISS 检索、BM25 检索、混合合并和来源过滤。",
            )
        )
        timing_events.extend(retrieval.debug.get("retrieval_timings", []))  # type: ignore

        rerank_result = self._rerank_results(
            question=normalized_question,
            results=retrieval.results,
            top_k=top_k,
            use_rerank=use_rerank,
        )
        timing_events.extend(rerank_result.timings)

        evidence_start = perf_counter()
        evidences = build_evidences(rerank_result.results)
        timing_events.append(
            _timing_event(
                "证据对象整理",
                "本地计算",
                evidence_start,
                "把底层检索结果转换成 prompt 和前端都能使用的 Evidence 结构。",
            )
        )

        prompt_start = perf_counter()
        prompt = build_rag_prompt(
            question=normalized_question,
            evidences=evidences,
            max_context_chars=self.max_context_chars,
        )
        timing_events.append(
            _timing_event(
                "Prompt 构造",
                "本地计算",
                prompt_start,
                "把用户问题、召回证据和回答要求组装成最终发送给 LLM 的文本。",
            )
        )

        debug_base = {
            "pipeline": [
                f"1. 使用 {retrieval.debug.get('retrieval_mode_label')} 召回证据",
                "2. 使用本地 rerank 模型对候选证据重排序",
                "3. 对重排结果整理为证据列表",
                "4. 把用户问题、召回证据和回答要求组装成 prompt",
                "5. 流式调用 DashScope LLM 生成基于证据的答案",
            ],
            "embedding_model": EMBEDDING_MODEL,
            "embedding_used": retrieval_mode in {"faiss", "hybrid"},
            "llm_max_tokens": LLM_MAX_TOKENS,
            "llm_enable_thinking": LLM_ENABLE_THINKING,
            "llm_stream": LLM_STREAM,
            "use_rerank": use_rerank,
            "top_k": top_k,
            "source_types": normalized_source_types or ["video", "wechat_qa"],
            "retrieved_count": len(evidences),
            "evidence_summary": build_prompt_debug_info(evidences),
            "final_prompt": prompt,
            "final_prompt_chars": len(prompt),
            "rerank_candidate_top_k": candidate_top_k,
            **rerank_result.debug,
            **retrieval.debug,
        }

        return PreparedRagRequest(
            question=normalized_question,
            prompt=prompt,
            evidences=evidences,
            debug_base=debug_base,
            timing_events=timing_events,
            pipeline_start=pipeline_start,
        )

    def _build_answer(
        self,
        *,
        prepared: PreparedRagRequest,
        generation: GenerationResult,
        generation_start: float,
    ) -> RagAnswer:
        """把准备阶段和 LLM 生成阶段合并成最终 RagAnswer。"""

        timing_events = list(prepared.timing_events)
        generation_timings = getattr(generation, "timings", [])
        if generation_timings:
            timing_events.extend(generation_timings)
        else:
            timing_events.append(
                _timing_event(
                    "LLM 答案生成",
                    "DashScope LLM 调用",
                    generation_start,
                    "调用 qwen3.6-flash 阅读召回证据并生成最终答案。",
                )
            )
        timing_events.append(
            _timing_event(
                "RAG 全流程总耗时",
                "端到端",
                prepared.pipeline_start,
                "从接收用户问题到拿到最终答案的总耗时。",
            )
        )

        debug = {
            **prepared.debug_base,
            "llm_model": generation.model,
            "llm_usage": generation.usage,
            "timings": timing_events,
        }

        return RagAnswer(
            question=prepared.question,
            answer=generation.answer,
            evidences=prepared.evidences,
            debug=debug,
        )

    @staticmethod
    def _get_candidate_top_k(final_top_k: int, *, use_rerank: bool) -> int:
        """计算进入 rerank 前的初步召回候选数量。

        rerank 的价值在于“先多召回，再精排”。
        如果只召回最终 top-k 条，rerank 没有足够候选可以调整顺序。
        """

        if not use_rerank or not RERANK_ENABLED:
            return final_top_k
        max_candidates = max(RERANK_MAX_CANDIDATES, final_top_k)
        return min(
            max(final_top_k, final_top_k * RERANK_CANDIDATE_MULTIPLIER),
            max_candidates,
        )

    def _rerank_results(
        self,
        *,
        question: str,
        results: list[Any],
        top_k: int,
        use_rerank: bool,
    ) -> RerankResult:
        """执行本地 rerank，并统一返回结果、耗时和调试信息。"""

        if not use_rerank:
            skip_start = perf_counter()
            return RerankResult(
                results=results[:top_k],
                timings=[
                    _timing_event(
                        "本地 Rerank 跳过",
                        "前端配置",
                        skip_start,
                        "前端关闭“使用 Rerank 重排”，直接使用初步检索排序结果。",
                    )
                ],
                debug={
                    "rerank_enabled": False,
                    "rerank_model_id": RERANK_MODEL_ID,
                    "rerank_model_dir": str(RERANK_MODEL_DIR),
                    "rerank_device": RERANK_DEVICE,
                    "rerank_skipped_reason": "前端关闭",
                    "rerank_input_count": len(results),
                    "rerank_output_count": min(len(results), top_k),
                },
            )

        return self.reranker.rerank(
            question=question,
            results=results,
            top_k=top_k,
        )

    @staticmethod
    def _normalize_source_types(source_types: list[str] | None) -> list[str] | None:
        """清洗前端传入的知识库来源，空列表表示不限制来源。"""

        if not source_types:
            return None

        allowed_source_types = {"video", "wechat_qa"}
        normalized = [
            source_type
            for source_type in source_types
            if source_type in allowed_source_types
        ]
        if set(normalized) == allowed_source_types:
            return None
        return normalized or None


def _timing_event(
    node: str,
    kind: str,
    start: float,
    description: str,
) -> dict[str, Any]:
    """生成一个流程耗时节点，帮助前端解释每一步花了多久。"""

    return {
        "node": node,
        "kind": kind,
        "elapsed_ms": round((perf_counter() - start) * 1000, 2),
        "description": description,
        "status": "done",
    }


def _running_event(
    node: str,
    kind: str,
    description: str,
) -> dict[str, Any]:
    """生成一个进行中的流程节点，让前端在长耗时步骤开始时就能给出反馈。"""

    return {
        "node": node,
        "kind": kind,
        "elapsed_ms": None,
        "description": description,
        "status": "running",
    }
