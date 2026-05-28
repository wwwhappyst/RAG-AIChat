"""测试 DashScope API 和指定 embedding 模型是否可用。

在 ai_tutor_rag 目录下运行：

    python test_dashscope_api.py

这个脚本只做一件事：用一小段文本调用 DashScope 多模态 embedding 模型。
它不会打印 API Key，只会告诉你：

- 环境变量是否存在；
- DashScope SDK 是否能导入；
- 模型是否返回 200；
- 返回的向量维度是否符合预期。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http import HTTPStatus
from typing import Any

from config import EMBEDDING_DIMENSION, EMBEDDING_MODEL


def parse_args() -> argparse.Namespace:
    """解析测试参数，方便后续临时切换模型或测试文本。"""

    parser = argparse.ArgumentParser(description="测试 DashScope embedding API")
    parser.add_argument(
        "--model",
        default=EMBEDDING_MODEL,
        help="要测试的 DashScope embedding 模型名。",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=EMBEDDING_DIMENSION,
        help="期望返回的向量维度。",
    )
    parser.add_argument(
        "--text",
        default="coze平台怎么注册并开始实践？",
        help="用于生成测试向量的一小段文本。",
    )
    return parser.parse_args()


def response_get(response: Any, key: str, default: Any = None) -> Any:
    """兼容 DashScope SDK 返回对象的属性访问和字典访问。"""

    if isinstance(response, dict):
        return response.get(key, default)
    return getattr(response, key, default)


def find_quota_fields(data: Any, prefix: str = "") -> dict[str, Any]:
    """递归查找响应里可能和额度相关的字段。

    说明：
    - DashScope 模型调用通常会返回 `usage`，表示本次请求用了多少 token。
    - 免费额度的“已用量 / 剩余量”属于账号计费侧信息，当前模型调用响应不一定返回。
    - 这里做一次保守扫描：如果未来 SDK 返回 quota、remaining 等字段，脚本可以直接展示。
    """

    keywords = ("quota", "remain", "remaining", "balance", "free")
    found: dict[str, Any] = {}

    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if any(keyword in str(key).lower() for keyword in keywords):
                found[path] = value
            found.update(find_quota_fields(value, path))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            found.update(find_quota_fields(value, f"{prefix}[{index}]"))

    return found


def print_usage_info(usage: dict[str, Any] | None) -> None:
    """打印本次 API 调用消耗，用于区分“模型可用”和“本次用了多少”。"""

    if not usage:
        print("本次请求用量: 响应中未返回 usage 字段")
        return

    details = usage.get("input_tokens_details", {}) or {}
    print("本次请求用量:")
    print(f"  input_tokens: {usage.get('input_tokens', 0)}")
    print(f"  text_tokens: {details.get('text_tokens', 0)}")
    print(f"  image_tokens: {details.get('image_tokens', 0)}")
    print(f"  output_tokens: {usage.get('output_tokens', 0)}")
    print(f"  total_tokens: {usage.get('total_tokens', 0)}")


def main() -> int:
    """发起一次最小 embedding 请求，并输出清晰的诊断信息。"""

    args = parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    print("DashScope API 连通性测试")
    print(f"模型: {args.model}")
    print(f"期望维度: {args.dimension}")
    print(f"API Key 环境变量: {'已检测到' if api_key else '未检测到'}")

    if not api_key:
        print("失败: 请先配置 DASHSCOPE_API_KEY 环境变量。")
        return 1

    try:
        import dashscope
    except ImportError:
        print("失败: 当前 Python 环境没有安装 dashscope。")
        print("建议执行: pip install -r requirements.txt")
        return 1

    response = dashscope.MultiModalEmbedding.call(
        api_key=api_key,
        model=args.model,
        input=[{"text": args.text}], # type: ignore
        dimension=args.dimension,
        auto_truncation=True,
    )

    status_code = response_get(response, "status_code")
    code = response_get(response, "code", "")
    message = response_get(response, "message", "")

    print(f"HTTP 状态: {status_code}")
    if status_code != HTTPStatus.OK:
        print("失败: DashScope 返回非 200 状态。")
        print(json.dumps({"code": code, "message": message}, ensure_ascii=False))
        return 1

    usage = response_get(response, "usage", {}) or {}
    print_usage_info(usage)

    output = response_get(response, "output", {}) or {}
    embeddings = output.get("embeddings", [])
    if not embeddings:
        print("失败: 返回结果中没有 embeddings。")
        print(json.dumps(output, ensure_ascii=False)[:1000])
        return 1

    vector = embeddings[0].get("embedding")
    if not isinstance(vector, list):
        print("失败: 返回结果中没有可用的 embedding 向量。")
        return 1

    print("成功: DashScope API 和 embedding 模型可用。")
    print(f"实际向量维度: {len(vector)}")
    print(f"向量前 5 项: {[round(value, 6) for value in vector[:5]]}")

    quota_fields = find_quota_fields(dict(response))
    if quota_fields:
        print("检测到可能的额度字段:")
        print(json.dumps(quota_fields, ensure_ascii=False, indent=2))
    else:
        print("免费额度信息: 本次模型调用响应未返回已用免费额度或剩余额度。")
        print("说明: 当前脚本可统计本次请求 token 用量；账号级免费额度请在百炼控制台的计费/额度页面查看。")

    if len(vector) != args.dimension:
        print("警告: 实际向量维度和期望维度不一致，请检查模型参数。")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
