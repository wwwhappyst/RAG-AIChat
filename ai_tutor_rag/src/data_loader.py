"""读取原始知识库文件。

第 1 阶段只负责安全地读取源 Excel 文件，并返回普通 Python 字典列表。
后续阶段不应该再直接依赖 Excel，而应该使用本阶段产出的标准 JSONL。
这样可以让“原始数据格式”和“RAG 内部数据格式”解耦。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_excel_rows(excel_path: Path) -> list[dict[str, Any]]:
    """读取一个 Excel 文件，并把每一行转换成字典。

    为什么要单独写这个函数：
    - Excel 是原始数据格式，不适合作为 RAG 系统内部长期流转的格式。
    - 我们先读一次 Excel，再统一清洗并写成 JSONL，后续模块都读 JSONL。
    - 把 Excel 读取逻辑隔离出来，后面测试向量库、检索器时会更简单。
    """

    if not excel_path.exists():
        raise FileNotFoundError(f"找不到 Excel 文件: {excel_path}")

    dataframe = pd.read_excel(excel_path)

    # `where(pd.notnull(...), None)` 会把 pandas 里的 NaN 转成普通 Python None。
    # 这样 cleaner.py 里只需要处理 None，不需要理解 pandas 的缺失值细节。
    dataframe = dataframe.where(pd.notnull(dataframe), None)
    raw_rows = dataframe.to_dict(orient="records")

    # Excel 的列名在 pandas 类型系统里可以是任意 Hashable。
    # 但进入本项目后，我们统一把字段名当作字符串，例如 questions、content、image。
    return [
        {str(column_name): value for column_name, value in row.items()}
        for row in raw_rows
    ]
