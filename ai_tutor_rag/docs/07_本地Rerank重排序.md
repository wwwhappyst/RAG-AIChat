# 07 本地 Rerank 重排序：先多召回，再精排

## 本阶段做什么

第 7 阶段只做一件事：

```text
在 FAISS / BM25 / 混合检索之后，加入本地 rerank 重排序。
```

本阶段暂时不做：

```text
query 改写
父页面检索
```

原因是 query 改写通常需要额外调用 DashScope 或其他 LLM，会增加等待时间和 API 成本。
父页面检索会引入新的文档层级和上下文扩展逻辑，当前阶段先保持链路清晰。

## 为什么要加 Rerank

FAISS 和 BM25 更像“初筛”：

- FAISS 擅长语义相似。
- BM25 擅长关键词命中。
- 混合检索把两路候选合并。

但是初筛分数不一定等于最终相关性。
Rerank 模型会把“用户问题”和“候选 chunk 文本”放在一起判断，输出一个更精细的相关性分数。

当前链路变成：

```text
用户问题
  -> FAISS / BM25 / 混合检索多召回候选
  -> 本地 rerank 模型重新打分排序
  -> 截取最终 top-k
  -> Prompt
  -> DashScope LLM 流式回答
```

## 当前使用的模型

当前推荐并配置：

```text
BAAI/bge-reranker-base
```

选择原因：

- 本地可部署，不产生 DashScope API 成本。
- 比 large 版本更轻，适合教学项目先跑通。
- 支持中英文相关性重排，适合当前课程视频和微信群问答混合数据。
- 可以直接用 `transformers` 加载 sequence classification 模型，工程复杂度低。

## 模型下载

模型下载使用 ModelScope。

先安装依赖：

```powershell
python -m pip install -r requirements.txt
```

再下载模型：

```powershell
python download_reranker_model.py
```

下载目标目录：

```text
models/bge-reranker-base
```

如果模型目录不存在，系统会跳过 rerank，并在流程日志中提示“本地模型缺失”。
这样前端不会因为还没下载模型而崩溃。

## Streamlit 日志里的 torchvision 报错

如果点击“开始提问”后，终端日志出现类似：

```text
streamlit/watcher/local_sources_watcher.py
transformers.models.xxx.image_processing_xxx.py
ModuleNotFoundError: No module named 'torchvision'
```

这不是 rerank 模型在重新下载，也不是文本 rerank 需要 `torchvision`。

真实原因是：Streamlit 的源码监听器会扫描已经导入的 Python 模块。
`transformers` 内部有大量懒加载模块，扫描时可能误触发视觉模型相关模块，
而这些视觉模块依赖 `torchvision`。

当前项目只使用文本 rerank，不需要视觉模型能力。
因此项目已在 `.streamlit/config.toml` 中关闭文件监听：

```toml
[server]
fileWatcherType = "none"
```

这样可以避免 Streamlit 扫描 `transformers` 时产生无关报错。
代价是：修改代码后需要手动重启 Streamlit 才能看到新代码生效。

## 耗时记录在哪里

第 7 阶段会新增这些流程节点：

```text
本地 Rerank 模型加载
本地 Rerank 重排序
Rerank 截取 top-k
```

它们会同时出现在：

- 最终答案上方的实时 RAG 执行流程。
- 流程日志里的“流程执行时间”表格。

第一次使用 rerank 时，模型加载可能比较慢。
后续 Streamlit 缓存 Pipeline 后，模型对象会复用，主要耗时会集中在“本地 Rerank 重排序”。

## 为什么第一次点击像是卡住

本地 rerank 第一次执行时，需要把模型权重从磁盘加载到内存。
终端里看到：

```text
Loading weights: 100%
```

表示模型权重已经读取完成，不是重新下载模型。
在 CPU 环境下，后面还会继续做候选证据推理，所以第一次点击会比后续慢。

为了降低等待时间，当前配置把进入 rerank 的候选数控制得更保守：

```text
RERANK_CANDIDATE_MULTIPLIER = 1
RERANK_MAX_CANDIDATES = 8
RERANK_MAX_LENGTH = 256
```

也就是说，默认 top-k=5 时，通常先召回 5 条候选给 rerank，而不是 10 条或更多。
这样能保留“本地模型重排”的教学效果，同时减少本地 CPU 推理等待。
如果 top-k 大于 `RERANK_MAX_CANDIDATES`，系统仍会保证候选数至少等于最终 top-k，避免少召回。

前端实时流程面板也会在长耗时节点开始时先显示“进行中”，
完成后再替换为绿色勾和实际耗时。

## 前端是否启用 Rerank

Streamlit 左侧新增：

```text
使用 Rerank 重排
```

开启时：

```text
初步检索结果 -> 本地 rerank 重排序 -> Prompt
```

关闭时：

```text
初步检索结果 -> 直接截取 top-k -> Prompt
```

关闭 rerank 不会加载本地 rerank 模型，因此速度更快。
这适合用来对比“检索原始排序”和“rerank 精排排序”的差异。
流程日志中会记录“本地 Rerank 跳过”，方便确认本次是否真的关闭了 rerank。

## 本阶段相关文件

```text
config.py
download_reranker_model.py
src/reranker.py
src/pipeline.py
app.py
docs/07_本地Rerank重排序.md
```
