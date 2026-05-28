"""本地 Rerank 重排序模块。

第 7 阶段只引入本地 rerank，不做 query 改写，也不做父页面检索。

Rerank 的位置在“初步召回”和“Prompt 构造”之间：

1. FAISS / BM25 / 混合检索先快速召回一批候选。
2. 本地 rerank 模型逐条判断“用户问题 + 候选 chunk”是否匹配。
3. 按 rerank 分数重新排序，再截取最终 top-k 交给 LLM。

这样做的意义是：召回阶段负责“尽量别漏”，rerank 阶段负责“更精细地排前后”。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from src.vector_store import SearchResult


@dataclass
class RerankResult:
    """rerank 后的结果和可解释调试信息。"""

    results: list[SearchResult]
    timings: list[dict[str, Any]]
    debug: dict[str, Any]


class LocalReranker:
    """基于本地 cross-encoder rerank 模型的重排序器。

    当前推荐 `BAAI/bge-reranker-base`，原因是它比 large 更轻，
    对中文和英文检索都比较稳，且可以直接用 transformers 在本地推理。
    """

    def __init__(
        self,
        *,
        enabled: bool,
        model_id: str,
        model_dir: Path,
        batch_size: int,
        max_length: int,
        device: str,
    ) -> None:
        self.enabled = enabled
        self.model_id = model_id
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.device_config = device
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None

    def rerank(
        self,
        *,
        question: str,
        results: list[SearchResult],
        top_k: int,
    ) -> RerankResult:
        """对初步召回结果做本地重排序。"""

        timings: list[dict[str, Any]] = []
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not results:
            return RerankResult(
                results=[],
                timings=[
                    _timing_event(
                        "本地 Rerank 跳过",
                        "本地计算",
                        perf_counter(),
                        "没有候选证据需要重排序。",
                    )
                ],
                debug=self._build_debug(enabled=False, skipped_reason="没有候选证据"),
            )
        if not self.enabled:
            return RerankResult(
                results=results[:top_k],
                timings=[
                    _timing_event(
                        "本地 Rerank 跳过",
                        "本地配置",
                        perf_counter(),
                        "配置关闭 RERANK_ENABLED，直接使用初步召回排序。",
                    )
                ],
                debug=self._build_debug(enabled=False, skipped_reason="配置关闭"),
            )
        if not self.model_dir.exists():
            return RerankResult(
                results=results[:top_k],
                timings=[
                    _timing_event(
                        "本地 Rerank 跳过",
                        "本地模型缺失",
                        perf_counter(),
                        f"未找到本地 rerank 模型目录：{self.model_dir}。请先运行下载脚本。",
                    )
                ],
                debug=self._build_debug(enabled=False, skipped_reason="本地模型缺失"),
            )

        load_start = perf_counter()
        self._ensure_model_loaded()
        timings.append(
            _timing_event(
                "本地 Rerank 模型加载",
                "本地模型",
                load_start,
                "首次使用时加载 tokenizer 和 sequence classification 模型；后续会复用内存中的模型。",
            )
        )

        score_start = perf_counter()
        scores = self._score_pairs(question=question, results=results)
        timings.append(
            _timing_event(
                "本地 Rerank 重排序",
                "本地模型推理",
                score_start,
                f"对 {len(results)} 条候选逐条计算 query+chunk 相关性分数。",
            )
        )

        select_start = perf_counter()
        reranked_results = self._attach_scores_and_sort(
            results=results,
            scores=scores,
        )[:top_k]
        timings.append(
            _timing_event(
                "Rerank 截取 top-k",
                "本地计算",
                select_start,
                "按 rerank 分数重新排序后，截取最终交给 Prompt 的证据数量。",
            )
        )

        return RerankResult(
            results=reranked_results,
            timings=timings,
            debug={
                **self._build_debug(enabled=True, skipped_reason=""),
                "rerank_input_count": len(results),
                "rerank_output_count": len(reranked_results),
            },
        )

    def _ensure_model_loaded(self) -> None:
        """懒加载本地模型，避免 Streamlit 启动时立刻占用内存。"""

        if self._tokenizer is not None and self._model is not None:
            return

        # 依赖放在函数内部导入，让没有安装 torch/transformers 的环境也能做语法检查。
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = self._resolve_device(torch)
        # 这里必须强制只读本地文件。
        # 有些模型配置里会保留训练时的 `_name_or_path`，例如 xlm-roberta-base。
        # 如果不加 local_files_only，transformers 可能会误以为需要联网补下载底座模型，
        # 导致用户点击“开始提问”时又出现下载日志或网络报错。
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        )
        model.to(device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def _resolve_device(self, torch: Any) -> str:
        """根据配置和本机情况选择推理设备。"""

        if self.device_config != "auto":
            return self.device_config
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _score_pairs(self, *, question: str, results: list[SearchResult]) -> list[float]:
        """批量计算 query 与每条候选文本的相关性分数。"""

        import torch

        if self._tokenizer is None or self._model is None or self._device is None:
            raise RuntimeError("rerank 模型尚未加载")

        scores: list[float] = []
        pairs = [
            (question, _build_rerank_candidate_text(result.metadata))
            for result in results
        ]

        with torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[start : start + self.batch_size]
                inputs = self._tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {
                    key: value.to(self._device)
                    for key, value in inputs.items()
                }
                logits = self._model(**inputs).logits
                batch_scores = logits.view(-1).detach().cpu().tolist()
                scores.extend(float(score) for score in batch_scores)

        return scores

    @staticmethod
    def _attach_scores_and_sort(
        *,
        results: list[SearchResult],
        scores: list[float],
    ) -> list[SearchResult]:
        """把 rerank 分写回 score_details，并按新分数排序。"""

        reranked: list[SearchResult] = []
        for pre_rank, (result, rerank_score) in enumerate(
            zip(results, scores),
            start=1,
        ):
            score_details = dict(result.score_details)
            score_details["pre_rerank_score"] = result.score
            score_details["pre_rerank_rank"] = pre_rank
            score_details["rerank_score"] = rerank_score
            reranked.append(
                SearchResult(
                    score=rerank_score,
                    metadata=result.metadata,
                    score_details=score_details,
                )
            )

        reranked.sort(key=lambda result: result.score, reverse=True)
        for rerank_rank, result in enumerate(reranked, start=1):
            result.score_details["rerank_rank"] = rerank_rank
        return reranked

    def _build_debug(self, *, enabled: bool, skipped_reason: str) -> dict[str, Any]:
        """生成前端可展示的 rerank 调试信息。"""

        return {
            "rerank_enabled": enabled,
            "rerank_model_id": self.model_id,
            "rerank_model_dir": str(self.model_dir),
            "rerank_device": self._device or self.device_config,
            "rerank_skipped_reason": skipped_reason,
        }


def _build_rerank_candidate_text(metadata: dict[str, Any]) -> str:
    """构造 rerank 看到的候选文本。

    这里保留问题别名和正文：问题别名帮助模型理解候选 chunk 的问法入口，
    正文才是最终回答依据。两者一起给 rerank，有助于判断候选是否真的匹配用户问题。
    """

    questions = metadata.get("questions") or []
    if not isinstance(questions, list):
        questions = [str(questions)]
    question_text = "；".join(str(question) for question in questions[:5])
    content = str(metadata.get("content", "")).strip()
    if question_text and content:
        return f"问题别名：{question_text}\n正文：{content}"
    return content or question_text


def _timing_event(
    node: str,
    kind: str,
    start: float,
    description: str,
) -> dict[str, Any]:
    """生成 rerank 阶段的耗时节点。"""

    return {
        "node": node,
        "kind": kind,
        "elapsed_ms": round((perf_counter() - start) * 1000, 2),
        "description": description,
    }
