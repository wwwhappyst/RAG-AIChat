"""下载第 7 阶段本地 rerank 模型。

本脚本使用 ModelScope 下载 `BAAI/bge-reranker-base` 到项目本地 models 目录。
下载完成后，在线问答链路会直接从本地目录加载模型，不会调用 DashScope rerank API。

运行方式：

    python download_reranker_model.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from config import RERANK_MODEL_DIR, RERANK_MODEL_ID


def main() -> int:
    """下载 rerank 模型，并把最终路径固定到 config.py 配置的位置。"""

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "未安装 modelscope。请先执行：pip install -r requirements.txt"
        ) from exc

    RERANK_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    local_cache_dir = RERANK_MODEL_DIR.parent / ".modelscope_cache"
    local_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MODELSCOPE_CACHE", str(local_cache_dir))

    print(f"开始下载 rerank 模型: {RERANK_MODEL_ID}")
    print(f"目标目录: {RERANK_MODEL_DIR}")
    print(f"ModelScope 缓存目录: {local_cache_dir}")

    try:
        downloaded_path = snapshot_download(
            model_id=RERANK_MODEL_ID,
            local_dir=str(RERANK_MODEL_DIR),
            cache_dir=str(local_cache_dir),
        )
    except TypeError:
        # 兼容旧版本 modelscope：旧接口可能没有 local_dir 参数。
        downloaded_path = snapshot_download(
            model_id=RERANK_MODEL_ID,
            cache_dir=str(RERANK_MODEL_DIR.parent),
        )
        _copy_downloaded_model(Path(downloaded_path), RERANK_MODEL_DIR)

    print(f"下载完成: {downloaded_path}")
    print("后续 Streamlit 提问时会从本地目录加载 rerank 模型。")
    return 0


def _copy_downloaded_model(source_dir: Path, target_dir: Path) -> None:
    """把旧版 ModelScope cache 目录复制到项目固定模型目录。"""

    if source_dir.resolve() == target_dir.resolve():
        return
    if target_dir.exists():
        return
    shutil.copytree(source_dir, target_dir)


if __name__ == "__main__":
    raise SystemExit(main())
