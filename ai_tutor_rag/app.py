"""第 4 阶段 Streamlit 前端。

这个页面不是单纯的聊天框，而是 RAG 调试台：

- 左侧负责输入问题和调整检索参数。
- 右侧展示答案、召回证据和流程日志。
- 每次提问都能看到系统先检索了哪些内容，再如何把证据交给 LLM。

这样设计是为了帮助初学者理解 RAG 的真实工作方式：答案质量不仅取决于大模型，
也取决于前面的检索结果和 prompt 构造。
"""

from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from config import DEFAULT_RETRIEVAL_TOP_K, EMBEDDING_MODEL, LLM_MODEL
from src.pipeline import BasicRagPipeline, RagAnswer
from src.retriever import RETRIEVAL_MODE_LABELS, RetrievalMode


SOURCE_TYPE_OPTIONS = {
    "课程视频": "video",
    "微信群问答": "wechat_qa",
}


@st.cache_resource(show_spinner=False)
def load_pipeline(embedding_model: str, llm_model: str) -> BasicRagPipeline:
    """加载并缓存 RAG Pipeline。

    Streamlit 每次交互都会重新执行脚本。FAISS 索引不需要每次都重新加载，
    所以这里使用缓存，减少页面操作时的等待时间。
    把模型名作为缓存参数，是为了在配置切换模型后自动重建 Pipeline，
    避免页面标题已更新但后台仍沿用旧模型客户端。
    """

    return BasicRagPipeline.from_config()


def render_sidebar() -> tuple[str, int, list[str], RetrievalMode, bool, bool, bool]:
    """渲染左侧参数面板，并返回用户输入。"""

    with st.sidebar:
        st.header("提问参数")
        question = st.text_area(
            "用户问题",
            value="coze平台怎么注册并开始实践？",
            height=120,
            placeholder="请输入一个和课程或答疑知识库相关的问题",
        )
        top_k = st.slider(
            "召回证据数量 top-k",
            min_value=1,
            max_value=10,
            value=DEFAULT_RETRIEVAL_TOP_K,
            help="top-k 越大，给模型看的候选证据越多，但也会增加 prompt 长度。",
        )
        st.markdown("知识库范围")
        st.caption("选择本次检索要使用的数据来源。两个都勾选时表示全库检索。")
        selected_labels: list[str] = []
        source_columns = st.columns(2)
        for column, label in zip(source_columns, SOURCE_TYPE_OPTIONS):
            with column:
                selected = st.checkbox(
                    label,
                    value=True,
                    key=f"source_type_{SOURCE_TYPE_OPTIONS[label]}",
                    help="取消勾选后，本次检索不会使用这个来源。",
                )
                if selected:
                    selected_labels.append(label)
        mode_options = list(RETRIEVAL_MODE_LABELS.keys())
        selected_mode = st.selectbox(
            "检索模式",
            options=mode_options,
            index=mode_options.index("hybrid"),
            format_func=lambda mode: RETRIEVAL_MODE_LABELS[mode],
            help="FAISS 偏语义相似，BM25 偏关键词精确命中，混合检索会把两路结果合并去重。",
        )
        use_rerank = st.checkbox(
            "使用 Rerank 重排",
            value=True,
            help="开启后会用本地 rerank 模型重排候选证据，排序更细，但 CPU 推理会增加等待时间。",
        )
        show_debug = st.checkbox(
            "显示调试信息",
            value=True,
            help="展示模型、流程日志、token 用量等信息，适合学习 RAG 链路。",
        )
        submitted = st.button("开始提问", type="primary", width="stretch")

    source_types = [SOURCE_TYPE_OPTIONS[label] for label in selected_labels]
    return question, top_k, source_types, selected_mode, use_rerank, show_debug, submitted


def run_question(
    *,
    question: str,
    top_k: int,
    source_types: list[str],
    retrieval_mode: RetrievalMode,
    use_rerank: bool,
) -> RagAnswer:
    """执行一次页面提问。

    页面层只负责收集参数和展示结果，真正的 RAG 逻辑仍然放在 pipeline 中。
    这样后续即使把 Streamlit 换成 FastAPI 或其他前端，也不用重写核心问答链路。
    """

    pipeline = load_pipeline(
        embedding_model=EMBEDDING_MODEL,
        llm_model=LLM_MODEL,
    )
    return pipeline.ask(
        question=question,
        top_k=top_k,
        source_types=source_types,
        retrieval_mode=retrieval_mode,
        use_rerank=use_rerank,
    )


def run_question_stream(
    *,
    question: str,
    top_k: int,
    source_types: list[str],
    retrieval_mode: RetrievalMode,
    use_rerank: bool,
):
    """执行一次页面流式提问。

    前端使用这个生成器边接收 LLM delta，边把答案写到页面上。
    流结束后，最后一个事件会带回完整 RagAnswer，供证据和调试区复用。
    """

    pipeline = load_pipeline(
        embedding_model=EMBEDDING_MODEL,
        llm_model=LLM_MODEL,
    )
    return pipeline.ask_stream(
        question=question,
        top_k=top_k,
        source_types=source_types,
        retrieval_mode=retrieval_mode,
        use_rerank=use_rerank,
    )


def render_answer(result: RagAnswer) -> None:
    """展示最终答案。"""

    st.subheader("最终答案")
    st.markdown(result.answer)


def render_progress_panel(progress_events: list[dict[str, Any]]) -> str:
    """渲染 RAG 实时流程面板。

    每完成一个节点就新增一行，帮助初学者看到系统不是“黑箱等待”，
    而是在按参数清洗、检索、证据整理、Prompt 构造、LLM 生成这些步骤推进。
    """

    if not progress_events:
        rows = '<div class="rag-progress-empty">等待开始执行 RAG 流程...</div>'
    else:
        rows = "\n".join(
            _format_progress_row(index=index, event=event)
            for index, event in enumerate(progress_events, start=1)
        )

    return f"""
<div class="rag-progress-panel">
  <div class="rag-progress-title">RAG 执行流程</div>
  <div class="rag-progress-list">
    {rows}
  </div>
</div>
"""


def _format_progress_row(index: int, event: dict[str, Any]) -> str:
    """把一个流程节点格式化成可读的一行 HTML。"""

    node = escape(str(event.get("node", "")))
    kind = escape(str(event.get("kind", "")))
    elapsed = event.get("elapsed_ms", 0)
    description = escape(str(event.get("description", "")))
    status = str(event.get("status") or "done")
    check_text = "..." if status == "running" else "✓"
    elapsed_text = "进行中" if elapsed is None else f"{elapsed} ms"
    return f"""
<div class="rag-progress-row rag-progress-row-{escape(status)}">
  <span class="rag-progress-check">{check_text}</span>
  <span class="rag-progress-index">{index}</span>
  <span class="rag-progress-node">{node}</span>
  <span class="rag-progress-kind">{kind}</span>
  <span class="rag-progress-time">{elapsed_text}</span>
  <span class="rag-progress-desc">{description}</span>
</div>
"""


def update_progress_events(
    progress_events: list[dict[str, Any]],
    event: dict[str, Any],
) -> None:
    """更新实时流程列表。

    同一个节点先出现“进行中”，完成后用带耗时的版本替换原行。
    这样页面不会重复刷出两行，也能让用户看到长耗时节点正在执行。
    """

    node = event.get("node")
    for index, old_event in enumerate(progress_events):
        if old_event.get("node") == node:
            progress_events[index] = event
            return
    progress_events.append(event)


def render_evidences(result: RagAnswer) -> None:
    """展示召回证据摘要和每条证据正文。"""

    st.subheader("召回证据")
    if not result.evidences:
        st.warning("当前筛选条件下没有召回到证据。可以放宽知识库范围或调大 top-k 再试。")
        return

    table_rows = [
        {
            "排名": evidence.rank,
            "排序分数": round(evidence.score, 4),
            "Rerank分": format_table_score(evidence.score_details.get("rerank_score")),
            "重排前分": format_table_score(evidence.score_details.get("pre_rerank_score")),
            "重排前排名": format_table_score(evidence.score_details.get("pre_rerank_rank")),
            "FAISS原始分": format_table_score(evidence.score_details.get("faiss_raw")),
            "FAISS归一分": format_table_score(evidence.score_details.get("faiss_norm")),
            "BM25原始分": format_table_score(evidence.score_details.get("bm25_raw")),
            "BM25归一分": format_table_score(evidence.score_details.get("bm25_norm")),
            "来源": source_type_label(evidence.source_type),
            "chunk_id": evidence.chunk_id,
            "标题": evidence.title,
            "正文预览": evidence.content[:80],
        }
        for evidence in result.evidences
    ]
    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)
    st.caption(
        "说明：当前 LLM 阶段使用流式输出，答案会先在“最终答案”页逐步显示；"
        "证据表格仍然在完整结果返回后统一展示。"
    )

    for evidence in result.evidences:
        with st.expander(
            f"证据 {evidence.rank} | {source_type_label(evidence.source_type)} | {evidence.title}",
            expanded=evidence.rank == 1,
        ):
            st.caption(
                f"chunk_id: {evidence.chunk_id} | 排序分数: {evidence.score:.4f}"
            )
            if evidence.score_details:
                st.markdown("**混合检索分数拆解**")
                st.json(format_score_details(evidence.score_details), expanded=False)
            if evidence.questions:
                st.markdown("**问题别名**")
                st.write("；".join(evidence.questions[:5]))
            st.markdown("**正文内容**")
            st.write(evidence.content)


def render_debug(result: RagAnswer) -> None:
    """展示可解释流程日志和模型调用信息。"""

    st.subheader("检索过程")
    for step in result.debug.get("pipeline", []):
        st.write(step)

    usage = result.debug.get("llm_usage", {}) or {}
    st.caption(f"检索模式：{result.debug.get('retrieval_mode_label', '')}")
    model_columns = st.columns(2)
    with model_columns[0]:
        render_text_card("Embedding 模型", result.debug.get("embedding_model", ""))
    with model_columns[1]:
        render_text_card(
            "LLM 模型",
            f"{result.debug.get('llm_model', '')} | max_tokens={result.debug.get('llm_max_tokens', '')} | enable_thinking={result.debug.get('llm_enable_thinking', '')} | stream={result.debug.get('llm_stream', '')}",
        )
    render_text_card(
        "Rerank 模型",
        f"{result.debug.get('rerank_model_id', '')} | use_rerank={result.debug.get('use_rerank', '')} | enabled={result.debug.get('rerank_enabled', '')} | device={result.debug.get('rerank_device', '')}",
    )

    metric_columns = st.columns(2)
    metric_columns[0].metric("召回数量", result.debug.get("retrieved_count", 0))
    metric_columns[1].metric("LLM tokens", usage.get("total_tokens", 0))
    if result.debug.get("retrieval_mode") == "hybrid":
        st.info(
            "混合检索使用本次候选池内 min-max 归一化："
            f"最终分 = {result.debug.get('hybrid_faiss_weight')} × FAISS归一分 "
            f"+ {result.debug.get('hybrid_bm25_weight')} × BM25归一分。"
        )

    with st.expander("原始调试数据"):
        st.json(result.debug)

    render_final_prompt(result.debug)
    render_timing_table(result.debug.get("timings", []))


def render_text_card(label: str, value: object) -> None:
    """展示适合长文本的调试字段。

    `st.metric` 适合展示短数字，但模型名通常很长。
    如果把模型名放进 metric，会被放大并截断，所以这里用可换行的小字号文本卡片。
    """

    st.markdown(
        f"""
<div class="debug-text-card">
  <div class="debug-text-card-label">{escape(label)}</div>
  <div class="debug-text-card-value">{escape(str(value))}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_final_prompt(debug: dict[str, object]) -> None:
    """展示最终发送给 LLM 的完整 prompt。

    对学习 RAG 来说，这一步很关键：召回证据并不是直接等于答案，
    系统会先把用户问题、证据和回答规则整理成 prompt，再交给 LLM。
    """

    st.subheader("最终发送给 LLM 的 Prompt")
    prompt = debug.get("final_prompt", "")
    if not isinstance(prompt, str) or not prompt:
        st.info("本次结果里没有记录最终 prompt。请重新提问一次生成新的调试数据。")
        return

    st.caption(
        f"字符数：{debug.get('final_prompt_chars', len(prompt))}。"
        "这里展示的是整理后的模型输入，不包含 DashScope API Key。"
    )
    with st.expander("查看完整 Prompt", expanded=False):
        st.code(prompt, language="markdown")


def render_timing_table(timings: object) -> None:
    """展示 RAG 各流程节点的执行耗时。

    这个表格用于定位“慢在哪里”：是外部模型调用慢，还是本地检索、分数融合、
    prompt 构造这些工程步骤慢。它展示的是工程链路耗时，不包含模型内部不可见过程。
    """

    st.subheader("流程执行时间")
    if not isinstance(timings, list) or not timings:
        st.info("本次结果里没有记录流程耗时。请重新提问一次生成新的调试数据。")
        return

    table_rows = []
    for index, item in enumerate(timings, start=1):
        if not isinstance(item, dict):
            continue
        table_rows.append(
            {
                "序号": index,
                "流程节点": item.get("node", ""),
                "类型": item.get("kind", ""),
                "耗时(ms)": item.get("elapsed_ms", 0),
                "说明": item.get("description", ""),
            }
        )

    if not table_rows:
        st.info("本次结果里没有可展示的流程耗时。")
        return

    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)


def render_empty_state() -> None:
    """展示未提问时的页面状态。"""

    st.info("在左侧输入问题并点击“开始提问”，这里会展示答案、召回证据和流程日志。")
    st.markdown(
        """
当前阶段已经支持：

- 使用 FAISS、BM25 或混合检索召回课程视频和微信群问答知识；
- 把召回证据放入 prompt；
- 调用 DashScope LLM 生成基于证据的答案。
"""
    )


def source_type_label(source_type: str) -> str:
    """把内部来源类型转换成页面上更容易理解的中文标签。"""

    labels = {
        "video": "课程视频",
        "wechat_qa": "微信群问答",
    }
    return labels.get(source_type, source_type)


def format_score_detail(value: object) -> object:
    """把分数明细格式化成展开详情里更易读的数字或占位符。"""

    if value is None:
        return "-"
    if isinstance(value, float):
        return round(value, 4)
    return value


def format_table_score(value: object) -> float | int | None:
    """把分数字段整理成 Streamlit 表格可稳定序列化的数值列。

    Streamlit 的 dataframe 底层会把 pandas 数据转成 Arrow 表。
    同一列如果同时出现数字和 "-" 字符串，Arrow 会尝试把 "-" 转成数字并报错。
    所以表格里用 None 表示缺失分数；展开详情再用 "-" 做人类可读占位符。
    """

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 4)
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def format_score_details(score_details: dict[str, object]) -> dict[str, object]:
    """格式化混合检索分数拆解，供前端展开查看。"""

    return {
        "FAISS 原始分": format_score_detail(score_details.get("faiss_raw")),
        "FAISS 归一分": format_score_detail(score_details.get("faiss_norm")),
        "FAISS 原始排名": format_score_detail(score_details.get("faiss_rank")),
        "BM25 原始分": format_score_detail(score_details.get("bm25_raw")),
        "BM25 归一分": format_score_detail(score_details.get("bm25_norm")),
        "BM25 原始排名": format_score_detail(score_details.get("bm25_rank")),
        "FAISS 权重": format_score_detail(score_details.get("faiss_weight")),
        "BM25 权重": format_score_detail(score_details.get("bm25_weight")),
        "最终混合分": format_score_detail(score_details.get("final_score")),
        "Rerank 分": format_score_detail(score_details.get("rerank_score")),
        "Rerank 排名": format_score_detail(score_details.get("rerank_rank")),
        "重排前分数": format_score_detail(score_details.get("pre_rerank_score")),
        "重排前排名": format_score_detail(score_details.get("pre_rerank_rank")),
    }


def main() -> None:
    """渲染 AI 助教 RAG 调试前端。"""

    st.set_page_config(
        page_title="AI 助教 RAG 调试台",
        layout="wide",
    )
    st.markdown(
        """
<style>
.block-container {
  padding-top: 3.25rem;
}
.compact-header {
  margin: 0 0 0.65rem 0;
  padding: 0;
}
.compact-header-title {
  font-size: 1.85rem;
  line-height: 1.15;
  font-weight: 760;
  letter-spacing: 0;
  margin: 0;
}
.compact-header-caption {
  margin-top: 0.35rem;
  color: rgba(250, 250, 250, 0.58);
  font-size: 0.82rem;
  line-height: 1.35;
}
.debug-text-card {
  border: 1px solid rgba(250, 250, 250, 0.14);
  border-radius: 8px;
  padding: 0.75rem 0.9rem;
  margin-bottom: 0.75rem;
  background: rgba(250, 250, 250, 0.03);
}
.debug-text-card-label {
  color: rgba(250, 250, 250, 0.72);
  font-size: 0.82rem;
  margin-bottom: 0.35rem;
}
.debug-text-card-value {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 0.95rem;
  line-height: 1.35;
  overflow-wrap: anywhere;
  word-break: break-word;
}
.rag-progress-panel {
  border: 1px solid rgba(250, 250, 250, 0.12);
  border-radius: 8px;
  background: rgba(250, 250, 250, 0.025);
  padding: 0.75rem 0.85rem;
  margin-bottom: 1rem;
}
.rag-progress-title {
  font-weight: 700;
  font-size: 0.95rem;
  margin-bottom: 0.55rem;
}
.rag-progress-list {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}
.rag-progress-row {
  display: grid;
  grid-template-columns: 1.2rem 1.5rem minmax(9rem, 1.2fr) minmax(6.5rem, 0.8fr) 5.5rem minmax(12rem, 2fr);
  gap: 0.5rem;
  align-items: center;
  font-size: 0.82rem;
  line-height: 1.35;
}
.rag-progress-check {
  color: #36d399;
  font-weight: 800;
}
.rag-progress-row-running .rag-progress-check,
.rag-progress-row-running .rag-progress-time {
  color: #fbbf24;
}
.rag-progress-index,
.rag-progress-kind,
.rag-progress-time,
.rag-progress-desc,
.rag-progress-empty {
  color: rgba(250, 250, 250, 0.62);
}
.rag-progress-node {
  color: rgba(250, 250, 250, 0.92);
  font-weight: 650;
}
.rag-progress-time {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  color: #36d399;
  white-space: nowrap;
}
@media (max-width: 900px) {
  .rag-progress-row {
    grid-template-columns: 1.2rem 1.5rem 1fr;
  }
  .rag-progress-kind,
  .rag-progress-time,
  .rag-progress-desc {
    grid-column: 3;
  }
}
</style>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="compact-header">
  <div class="compact-header-title">AI 助教 RAG 调试台</div>
  <div class="compact-header-caption">
    当前链路：FAISS / BM25 / 混合检索 -> 本地 Rerank -> 证据 prompt -> DashScope LLM。
    Embedding: {escape(EMBEDDING_MODEL)}，LLM: {escape(LLM_MODEL)}。
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    (
        question,
        top_k,
        source_types,
        retrieval_mode,
        use_rerank,
        show_debug,
        submitted,
    ) = render_sidebar()

    if not submitted:
        render_empty_state()
        return

    if not question.strip():
        st.warning("请先输入问题。")
        return
    if not source_types:
        st.warning("请至少选择一个知识库范围。")
        return

    answer_tab, evidence_tab, debug_tab = st.tabs(
        ["最终答案", "召回证据", "流程日志"]
    )

    result_holder: dict[str, RagAnswer] = {}

    with answer_tab:
        progress_events: list[dict[str, Any]] = []
        progress_placeholder = st.empty()
        progress_placeholder.markdown(
            render_progress_panel(progress_events),
            unsafe_allow_html=True,
        )
        st.subheader("最终答案")

        def stream_answer():
            """把 Pipeline 的流式事件转换成 Streamlit 可渲染的文本流。"""

            for event in run_question_stream(
                question=question,
                top_k=top_k,
                source_types=source_types,
                retrieval_mode=retrieval_mode,
                use_rerank=use_rerank,
            ):
                if event.progress:
                    update_progress_events(progress_events, event.progress)
                    progress_placeholder.markdown(
                        render_progress_panel(progress_events),
                        unsafe_allow_html=True,
                    )
                if event.answer_delta:
                    yield event.answer_delta
                if event.result is not None:
                    result_holder["result"] = event.result

        try:
            with st.spinner("正在检索证据并流式生成答案..."):
                st.write_stream(stream_answer())
        except Exception as exc:  # noqa: BLE001
            st.error("本次问答失败，请检查 DashScope、FAISS 索引或输入参数。")
            st.exception(exc)
            return

    result = result_holder.get("result")
    if result is None:
        st.error("流式生成结束后没有拿到完整问答结果，请重试。")
        return

    with evidence_tab:
        render_evidences(result)
    with debug_tab:
        if show_debug:
            render_debug(result)
        else:
            st.info("左侧开启“显示调试信息”后，可以查看模型、token 用量和原始 debug 数据。")


if __name__ == "__main__":
    main()
