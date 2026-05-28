"""查询向量本地缓存。

查询向量缓存解决的是在线问答里的一个常见性能问题：
同一个用户问题在调试时可能会反复提问，如果每次都重新调用 DashScope embedding，
页面会一直等待远程 API。把 query vector 按“模型 + 维度 + 规范化问题”缓存下来后，
重复问题可以直接从本地文件读取，FAISS 检索仍然使用同一套 768 维向量。
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class QueryVectorCache:
    """保存和读取查询向量的 JSONL 缓存。

    这里使用 JSONL 而不是数据库，是为了让初学者能直接打开文件观察缓存内容。
    cache key 会包含 embedding 模型和维度，避免以后更换模型后误用旧向量。
    """

    def __init__(self, *, path: Path, model: str, dimension: int) -> None:
        self.path = path
        self.model = model
        self.dimension = dimension
        self._lock = threading.Lock()
        self._vectors: dict[str, list[float]] = {}
        self._load()

    def get(self, query: str) -> list[float] | None:
        """读取某个问题的缓存向量，未命中时返回 None。"""

        key = self._build_key(query)
        vector = self._vectors.get(key)
        if vector is None:
            return None
        return list(vector)

    def set(self, query: str, vector: list[float]) -> None:
        """写入某个问题的查询向量。

        如果缓存文件写入失败，不应该影响主流程问答；但维度错误必须暴露出来，
        因为错误维度进入 FAISS 会导致检索结果不可信。
        """

        if len(vector) != self.dimension:
            raise ValueError(
                f"缓存向量维度不匹配: 期望 {self.dimension}, 实际 {len(vector)}"
            )

        key = self._build_key(query)
        normalized_query = _normalize_query(query)

        with self._lock:
            if key in self._vectors:
                return

            self.path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "key": key,
                "query": normalized_query,
                "model": self.model,
                "dimension": self.dimension,
                "vector": vector,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._vectors[key] = list(vector)

    def _load(self) -> None:
        """启动时把已有缓存读入内存，加快后续查询。"""

        if not self.path.exists():
            return

        with self.path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    continue

                if not self._is_current_model_record(record):
                    continue

                key = str(record.get("key", ""))
                vector = record.get("vector")
                if key and _is_valid_vector(vector, self.dimension):
                    self._vectors[key] = [float(value) for value in vector]

    def _build_key(self, query: str) -> str:
        """根据模型、维度和问题文本生成稳定 cache key。"""

        payload = {
            "model": self.model,
            "dimension": self.dimension,
            "query": _normalize_query(query),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _is_current_model_record(self, record: dict[str, Any]) -> bool:
        """只加载当前 embedding 配置可复用的缓存记录。"""

        try:
            record_dimension = int(record.get("dimension", -1))
        except (TypeError, ValueError):
            return False

        return record.get("model") == self.model and record_dimension == self.dimension


def _normalize_query(query: str) -> str:
    """规范化问题文本，让多余空白不影响缓存命中。"""

    return " ".join(query.strip().split())


def _is_valid_vector(value: Any, dimension: int) -> bool:
    """检查缓存文件里的向量是否仍然可用。"""

    return isinstance(value, list) and len(value) == dimension
