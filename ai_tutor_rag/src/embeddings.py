"""DashScope 文本向量化封装。

本模块位于 RAG 流程的“检索准备阶段”：

1. 建库时，把每条知识记录的 `retrieval_text` 转成文档向量。
2. 查询时，把用户问题转成查询向量。
3. 后续 FAISS 只负责比较向量相似度，不再理解原始中文文本。

为什么要单独封装：
- DashScope 是外部服务，调用方式、错误信息和批处理限制都应该集中处理。
- 其他模块只需要调用 `embed_documents()` 或 `embed_query()`，不需要关心 SDK 细节。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any

import dashscope

from src.query_cache import QueryVectorCache


@dataclass
class EmbeddingResult:
    """一次 embedding 调用的结果。

    `vectors` 是真正要写入 FAISS 的稠密向量。
    `total_tokens` 用来帮助学习者理解：embedding 也是模型调用，也会产生 token 消耗。
    """

    vectors: list[list[float]]
    total_tokens: int = 0


class DashScopeEmbeddingClient:
    """面向本项目的 DashScope embedding 客户端。

    DashScope 文本向量接口支持区分 `document` 和 `query`：
    - document：用于知识库内容建索引。
    - query：用于用户问题检索。

    这样做属于“非对称检索”的常见设计，模型可以分别优化文档表示和查询表示。
    如果使用 `tongyi-embedding-vision-*` 多模态模型，则改走 `MultiModalEmbedding`
    接口；当前文本知识库会以 `{"text": "..."} ` 的形式送入模型。
    """

    def __init__(
        self,
        *,
        model: str,
        dimension: int,
        batch_size: int,
        base_url: str,
        query_cache_path: Path | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if batch_size > 10:
            raise ValueError("DashScope 文本向量接口一次最多提交 10 条文本")

        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.base_url = base_url
        self.query_cache = (
            QueryVectorCache(
                path=query_cache_path,
                model=model,
                dimension=dimension,
            )
            if query_cache_path is not None
            else None
        )
        self.last_query_cache_hit: bool | None = None
        dashscope.base_http_api_url = self.base_url

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        """把知识库文本批量转换成文档向量。"""

        return self._embed_texts(texts=texts, text_type="document")

    def embed_query(self, query: str) -> list[float]:
        """把用户问题转换成查询向量。

        查询阶段会优先读本地缓存。缓存命中时不再调用 DashScope，
        这样 Streamlit 里重复调试同一个问题会明显更快。
        """

        if self.query_cache is not None:
            cached_vector = self.query_cache.get(query)
            if cached_vector is not None:
                self.last_query_cache_hit = True
                return cached_vector

        self.last_query_cache_hit = False
        result = self._embed_texts(texts=[query], text_type="query")
        vector = result.vectors[0]
        if self.query_cache is not None:
            self.query_cache.set(query, vector)
        return vector

    def _embed_texts(self, *, texts: list[str], text_type: str) -> EmbeddingResult:
        """按批次调用 DashScope，并保持返回向量顺序和输入文本顺序一致。"""

        if not texts:
            return EmbeddingResult(vectors=[], total_tokens=0)

        self._check_api_key()

        all_vectors: list[list[float]] = []
        total_tokens = 0

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            if self._is_multimodal_model():
                response = dashscope.MultiModalEmbedding.call(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    model=self.model,
                    input=[{"text": text} for text in batch],
                    dimension=self.dimension,
                    auto_truncation=True,
                )
            else:
                response = dashscope.TextEmbedding.call(
                    model=self.model,
                    input=batch,
                    dimension=self.dimension,
                    output_type="dense",
                    text_type=text_type,
                )

            status_code = self._response_get(response, "status_code")
            if status_code != HTTPStatus.OK:
                code = self._response_get(response, "code", "")
                message = self._response_get(response, "message", "")
                raise RuntimeError(
                    f"DashScope embedding 调用失败: status={status_code}, "
                    f"code={code}, message={message}"
                )

            embeddings = self._response_get(response, "output", {}).get(
                "embeddings", []
            )
            ordered_embeddings = sorted(
                embeddings,
                key=lambda item: item.get("text_index", 0),
            )

            for item in ordered_embeddings:
                vector = item.get("embedding")
                if not isinstance(vector, list):
                    raise RuntimeError("DashScope 返回结果中缺少 embedding 向量")
                if len(vector) != self.dimension:
                    raise RuntimeError(
                        f"向量维度不匹配: 期望 {self.dimension}, 实际 {len(vector)}"
                    )
                all_vectors.append(vector)

            usage = self._response_get(response, "usage", {}) or {}
            total_tokens += int(usage.get("total_tokens", 0))

        return EmbeddingResult(vectors=all_vectors, total_tokens=total_tokens)

    def _is_multimodal_model(self) -> bool:
        """判断当前模型是否需要走 DashScope 多模态 embedding 接口。"""

        multimodal_prefixes = (
            "tongyi-embedding-vision-",
            "qwen3-vl-embedding",
            "qwen2.5-vl-embedding",
            "multimodal-embedding-",
        )
        return self.model.startswith(multimodal_prefixes)

    @staticmethod
    def _check_api_key() -> None:
        """确认环境变量中存在 API Key，避免发起一定会失败的网络请求。"""

        if not os.getenv("DASHSCOPE_API_KEY"):
            raise RuntimeError(
                "未检测到 DASHSCOPE_API_KEY 环境变量。请先配置 DashScope API Key。"
            )

    @staticmethod
    def _response_get(response: Any, key: str, default: Any = None) -> Any:
        """兼容 DashScope SDK 返回对象的属性访问和字典访问。"""

        if isinstance(response, dict):
            return response.get(key, default)
        return getattr(response, key, default)
