"""DashScope LLM 答案生成封装。

本模块位于 RAG 流程的最后一步：它不负责检索，也不直接读取知识库。
它只接收已经构造好的 prompt，然后调用 DashScope 大语言模型生成答案。

这样拆分的好处是：后续如果要替换模型、调整温度或增加流式输出，不会影响 FAISS 检索逻辑。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator

from openai import OpenAI


@dataclass
class GenerationResult:
    """一次 LLM 生成的结构化结果。"""

    answer: str
    model: str
    usage: dict[str, Any]
    timings: list[dict[str, Any]]


@dataclass
class GenerationStreamEvent:
    """一次流式生成事件。

    `delta` 是本次新增的答案片段；`result` 只会在流式结束时出现，
    里面包含完整答案、usage 和耗时信息。
    """

    delta: str = ""
    result: GenerationResult | None = None


class DashScopeGenerator:
    """面向本项目的 DashScope LLM 客户端。"""

    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        base_url: str,
        max_tokens: int,
        enable_thinking: bool,
        stream: bool,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.stream = stream

        api_key = os.getenv("DASHSCOPE_API_KEY")
        self._check_api_key(api_key)
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )

    def generate(self, prompt: str) -> GenerationResult:
        """调用 DashScope LLM，根据 RAG prompt 生成最终答案。

        配置开启流式时，命令行等同步调用方仍然可以使用这个方法：
        它会内部消费流式片段，并在最后返回完整 `GenerationResult`。
        """

        if self.stream:
            final_result: GenerationResult | None = None
            for event in self.generate_stream(prompt):
                if event.result is not None:
                    final_result = event.result
            if final_result is None:
                raise RuntimeError("DashScope LLM 流式返回结束，但没有生成最终结果")
            return final_result

        return self._generate_once(prompt)

    def generate_stream(self, prompt: str) -> Iterator[GenerationStreamEvent]:
        """流式调用 DashScope LLM，边接收边产出答案片段。

        流式输出主要优化“用户开始看到答案”的时间。
        最终仍会在流结束后汇总完整答案、token usage 和耗时信息，
        供前端调试区继续展示。
        """

        total_start = perf_counter()
        timings = self._build_initial_timings(prompt, stream=True)
        request_start = perf_counter()

        answer_parts: list[str] = []
        usage: dict[str, Any] = {}
        first_chunk_received = False
        stream_receive_start = perf_counter()

        for chunk in self._iter_stream_chunks(prompt):
            if not first_chunk_received:
                first_chunk_received = True
                timings.append(
                    _timing_event(
                        "DashScope 流式首块等待",
                        "网络/API/模型生成",
                        request_start,
                        "等待 DashScope 返回第一个流式响应块；这个时间更接近用户体感的首字等待。",
                    )
                )

            chunk_usage = self._extract_usage(chunk)
            if chunk_usage:
                usage = chunk_usage

            delta = self._extract_stream_delta(chunk)
            if delta:
                answer_parts.append(delta)
                yield GenerationStreamEvent(delta=delta)

        timings.append(
            _timing_event(
                "DashScope 流式响应接收",
                "网络/API/模型生成",
                stream_receive_start,
                "持续接收 DashScope 返回的增量文本，直到模型完成本次回答。",
            )
        )

        answer = self._join_stream_answer(answer_parts)
        timings.extend(self._build_finish_timings(answer, usage, total_start, stream=True))
        yield GenerationStreamEvent(
            result=GenerationResult(
                answer=answer,
                model=self.model,
                usage=usage,
                timings=timings,
            )
        )

    def _generate_once(self, prompt: str) -> GenerationResult:
        """同步调用 DashScope LLM，等待完整答案后一次性返回。"""

        total_start = perf_counter()
        timings = self._build_initial_timings(prompt, stream=False)

        request_start = perf_counter()
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"enable_thinking": self.enable_thinking},
        )
        timings.append(
            _timing_event(
                "DashScope Chat Completions 请求等待",
                "网络/API/模型生成",
                request_start,
                "同步等待 DashScope 返回完整响应；这里包含网络传输、服务端调度、模型阅读证据和生成文本。",
            )
        )

        answer = self._extract_answer(completion)
        if not answer:
            raise RuntimeError("DashScope LLM 返回成功，但没有解析到答案文本")
        usage = self._extract_usage(completion)
        timings.extend(self._build_finish_timings(answer, usage, total_start, stream=False))

        return GenerationResult(
            answer=answer,
            model=self.model,
            usage=usage,
            timings=timings,
        )

    def _iter_stream_chunks(self, prompt: str) -> Iterator[Any]:
        """迭代流式响应，并兼容不支持 stream_options 的 OpenAI 兼容服务。"""

        request_payload = {
            "model": self.model,
            "messages": self._build_messages(prompt),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": {"enable_thinking": self.enable_thinking},
        }
        yielded_any_chunk = False
        try:
            for chunk in self.client.chat.completions.create(**request_payload):
                yielded_any_chunk = True
                yield chunk
        except Exception as exc:  # noqa: BLE001
            if yielded_any_chunk or not _is_stream_options_error(exc):
                raise

            # 有些 OpenAI 兼容服务支持流式文本，但不支持在最后一个 chunk 返回 usage。
            # 这种情况下优先保证答案能流式展示，usage 会退化为空字典。
            request_payload.pop("stream_options", None)
            for chunk in self.client.chat.completions.create(**request_payload):
                yield chunk

    def _build_initial_timings(self, prompt: str, *, stream: bool) -> list[dict[str, Any]]:
        """记录发起 LLM 请求前的本地准备耗时。"""

        timings: list[dict[str, Any]] = []

        client_start = perf_counter()
        timings.append(
            _timing_event(
                "OpenAI 兼容客户端复用",
                "本地计算",
                client_start,
                "复用 Pipeline 初始化时创建的 DashScope OpenAI 兼容客户端，避免每次提问重新初始化。",
            )
        )

        payload_start = perf_counter()
        mode_text = "stream=True" if stream else "stream=False"
        self._build_messages(prompt)
        timings.append(
            _timing_event(
                "LLM 请求参数组装",
                "本地计算",
                payload_start,
                f"组装 system/user messages；prompt 字符数约 {len(prompt)}，max_tokens={self.max_tokens}，enable_thinking={self.enable_thinking}，{mode_text}。",
            )
        )
        return timings

    def _build_finish_timings(
        self,
        answer: str,
        usage: dict[str, Any],
        total_start: float,
        *,
        stream: bool,
    ) -> list[dict[str, Any]]:
        """记录答案解析、usage 解析和 LLM 模块总耗时。"""

        answer_start = perf_counter()
        answer_node = "LLM 回答文本拼接" if stream else "LLM 回答文本解析"
        answer_description = (
            f"把流式 delta 片段拼接成完整答案；答案字符数约 {len(answer)}。"
            if stream
            else f"从响应 choices[0].message.content 中取出答案；答案字符数约 {len(answer)}。"
        )
        timings = [
            _timing_event(
                answer_node,
                "本地计算",
                answer_start,
                answer_description,
            )
        ]

        usage_start = perf_counter()
        timings.append(
            _timing_event(
                "LLM usage 解析",
                "本地计算",
                usage_start,
                _build_usage_description(usage),
            )
        )

        total_kind = "DashScope LLM 流式调用" if stream else "DashScope LLM 调用"
        total_description = (
            "从复用客户端、组装请求、接收流式片段到拼接完整答案的总耗时。"
            if stream
            else "从复用客户端、组装请求到解析完答案和 token 用量的总耗时。"
        )
        timings.append(
            _timing_event(
                "LLM 模块总耗时",
                total_kind,
                total_start,
                total_description,
            )
        )
        return timings

    @staticmethod
    def _build_messages(prompt: str) -> list[dict[str, str]]:
        """构造 Chat Completions messages，避免同步和流式路径重复。"""

        return [
            {
                "role": "system",
                "content": "你是一个严谨的中文 AI 助教，只基于用户提供的证据回答。",
            },
            {"role": "user", "content": prompt},
        ]

    @staticmethod
    def _join_stream_answer(answer_parts: list[str]) -> str:
        """把流式片段拼成最终答案，并检查空响应。"""

        answer = "".join(answer_parts).strip()
        if not answer:
            raise RuntimeError("DashScope LLM 流式返回成功，但没有解析到答案文本")
        return answer

    @staticmethod
    def _check_api_key(api_key: str | None) -> None:
        """确认环境变量中存在 API Key，避免发起一定会失败的生成请求。"""

        if not api_key:
            raise RuntimeError(
                "未检测到 DASHSCOPE_API_KEY 环境变量。请先配置 DashScope API Key。"
            )

    @staticmethod
    def _extract_answer(completion: Any) -> str:
        """从 OpenAI 兼容响应中取出 assistant 的回答文本。"""

        choices = getattr(completion, "choices", []) or []
        if not choices:
            return ""

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "") if message else ""
        return content.strip() if isinstance(content, str) else ""

    @staticmethod
    def _extract_stream_delta(chunk: Any) -> str:
        """从流式 chunk 中取出本次新增文本。"""

        choices = getattr(chunk, "choices", []) or []
        if not choices:
            return ""

        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", "") if delta else ""
        return content if isinstance(content, str) else ""

    @staticmethod
    def _extract_usage(completion: Any) -> dict[str, Any]:
        """把 OpenAI SDK 的 usage 对象转换成普通字典，方便前端展示。"""

        usage = getattr(completion, "usage", None)
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if isinstance(usage, dict):
            return usage
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }


def _timing_event(
    node: str,
    kind: str,
    start: float,
    description: str,
) -> dict[str, Any]:
    """生成 LLM 生成模块内部的耗时节点。"""

    return {
        "node": node,
        "kind": kind,
        "elapsed_ms": round((perf_counter() - start) * 1000, 2),
        "description": description,
    }


def _is_stream_options_error(exc: Exception) -> bool:
    """判断异常是否来自流式 usage 选项不兼容。"""

    message = str(exc).lower()
    return "stream_options" in message or "include_usage" in message


def _build_usage_description(usage: dict[str, Any]) -> str:
    """把 token 用量整理成耗时表可读说明。"""

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    completion_details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = completion_details.get("reasoning_tokens")
    if reasoning_tokens is None:
        return (
            f"解析 token 用量：prompt={prompt_tokens}，"
            f"completion={completion_tokens}，total={total_tokens}。"
        )
    return (
        f"解析 token 用量：prompt={prompt_tokens}，completion={completion_tokens}，"
        f"reasoning={reasoning_tokens}，total={total_tokens}。"
    )
