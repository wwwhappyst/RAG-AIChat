"""项目级配置。

这个文件集中管理路径和模型名称。对 RAG 项目来说，这样做很重要：
数据清洗、索引构建、检索流程、Streamlit 前端都会反复使用同一批路径。
把配置集中起来，后续修改路径时不需要到处找代码。
"""

from pathlib import Path


# 当前 Python 工程目录，也就是 `ai_tutor_rag/`。
# 后续所有项目内路径都从这里拼出来，避免脚本在不同工作目录下运行时找错文件。
PROJECT_ROOT = Path(__file__).resolve().parent

# 工作区根目录，里面同时放着 `ai_tutor_rag/` 和两份原始知识库。
# 第 1 阶段读取 Excel 时，需要从工程目录回到工作区根目录再定位原始数据。
WORKSPACE_ROOT = PROJECT_ROOT.parent

# 课程视频知识库的 Excel 汇总文件。
# 这类数据主要来自课程内容，source_type 会被标准化为 `video`。
VIDEO_EXCEL_PATH = (
    WORKSPACE_ROOT / "课程视频知识库" / "视频_json_excel" / "video-all.xlsx"
)

# 微信群问答知识库的 Excel 汇总文件。
# 这类数据主要来自答疑记录，source_type 会被标准化为 `wechat_qa`。
WECHAT_QA_EXCEL_PATH = (
    WORKSPACE_ROOT
    / "微信群问答知识库"
    / "问题列表转excel"
    / "AI解决方案专家-答疑汇总-all.xlsx"
)

# 第 1 阶段清洗后的标准数据目录。
# 原始 Excel 不直接喂给检索系统，先统一成 JSONL 和统计文件。
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# 标准知识库 JSONL 文件，一行表示一个知识 chunk。
# 后续 embedding、FAISS、BM25、rerank 都应该优先读取这个文件。
KNOWLEDGE_BASE_JSONL = PROCESSED_DATA_DIR / "knowledge_base.jsonl"

# 标准化后的统计信息，例如记录数、来源分布、问题别名数量等。
# 它不参与检索，只用于检查第 1 阶段数据清洗是否符合预期。
KNOWLEDGE_BASE_STATS_JSON = PROCESSED_DATA_DIR / "knowledge_base_stats.json"

# 第 2 阶段开始会生成检索索引。索引文件和原始数据分开存放，
# 这样可以随时删除并重建索引，而不会影响已经清洗好的知识库 JSONL。
# 所有索引文件的根目录。
INDEXES_DIR = PROJECT_ROOT / "indexes"

# FAISS 二进制索引目录，只保存向量和 FAISS 内部编号。
FAISS_INDEX_DIR = INDEXES_DIR / "faiss"

# metadata 目录，保存向量编号对应的 chunk_id、title、content 等可读信息。
METADATA_INDEX_DIR = INDEXES_DIR / "metadata"

# FAISS 索引文件路径。
# 检索时先加载它，用 query 向量搜索最相似的知识向量。
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "knowledge_base.index"

# BM25 关键词索引目录。
# BM25 是本地关键词检索算法，不调用任何 AI API；它依赖分词后的词项匹配来召回内容。
BM25_INDEX_DIR = INDEXES_DIR / "bm25"

# BM25 索引文件路径。
# 这里保存分词后的语料和 metadata，在线提问时加载后即可进行关键词检索。
BM25_INDEX_PATH = BM25_INDEX_DIR / "knowledge_base_bm25.pkl"

# BM25 构建说明文件路径。
# 记录语料规模、分词方式等信息，方便确认当前索引是否由最新知识库构建而来。
BM25_MANIFEST_PATH = BM25_INDEX_DIR / "knowledge_base_bm25_manifest.json"

# 向量 metadata 文件路径。
# FAISS 返回的是向量位置，系统再用同位置 metadata 找回原始 chunk 内容。
VECTOR_METADATA_PATH = METADATA_INDEX_DIR / "knowledge_base_metadata.jsonl"

# 向量库构建说明文件路径。
# 记录模型名、维度、记录数、token 用量等信息，方便排查索引是否和配置匹配。
VECTOR_MANIFEST_PATH = METADATA_INDEX_DIR / "knowledge_base_manifest.json"

# 查询向量缓存文件路径。
# 在线提问时，同一个问题在相同 embedding 模型和维度下会生成相同 query vector。
# 缓存后，重复问题可以跳过 DashScope embedding 远程调用，直接进入 FAISS 检索。
QUERY_VECTOR_CACHE_PATH = INDEXES_DIR / "query_cache" / "query_vectors.jsonl"

# 本地模型目录。
# 第 7 阶段开始，rerank 使用本地可部署模型，不调用 DashScope 或其他云端 API。
MODELS_DIR = PROJECT_ROOT / "models"

# 是否启用本地 rerank。
# rerank 位于“初步召回”和“Prompt 构造”之间，用更精细的 query+chunk 相关性分数重排候选。
RERANK_ENABLED = True

# 本地 rerank 模型。
# 这里先采用稳定且相对轻量的 `BAAI/bge-reranker-base`。
# 它不是 API 模型，下载后在本地推理，因此不会产生 DashScope token 成本。
RERANK_MODEL_ID = "BAAI/bge-reranker-base"
RERANK_MODEL_DIR = MODELS_DIR / "bge-reranker-base"

# 初步召回候选放大倍数。
# rerank 需要有足够候选才能发挥作用，但本地 CPU 推理也会随候选数变慢。
# 教学调试台先采用更保守的候选规模：默认 top-k=5 时进入 rerank 的候选约为 5-8 条。
RERANK_CANDIDATE_MULTIPLIER = 1
RERANK_MAX_CANDIDATES = 6

# rerank 本地推理参数。
# batch_size 越大吞吐越好，但内存占用也更高；教学项目先用保守值。
RERANK_BATCH_SIZE = 8
RERANK_MAX_LENGTH = 256
RERANK_DEVICE = "auto"

# 当前按用户指定使用通义多模态 embedding flash 模型。
# 即使当前知识库主要是文本，该模型也可以先生成文本向量；
# 后续如果要把图片字段纳入检索，可以继续扩展为图文融合向量。
# embedding 模型名称。
# 该模型负责把知识库文本和用户问题转换成同一语义空间里的向量。
EMBEDDING_MODEL = "tongyi-embedding-vision-flash-2026-03-06"

# embedding 向量维度。
# FAISS 索引维度必须和这里保持一致，否则查询向量无法进入同一个索引。
EMBEDDING_DIMENSION = 768

# embedding 批处理大小。
# DashScope 多模态 embedding 当前按小批次请求更稳，也便于失败时定位是哪一批出错。
EMBEDDING_BATCH_SIZE = 10

# DashScope 原生 API 地址。
# 当前用于 MultiModalEmbedding.call，也就是第 2 阶段建库和查询向量化。
EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"

# Qwen3.6 系列在百炼文档中推荐使用 OpenAI 兼容的 Chat Completions 调用方式。
# embedding 仍然使用 DashScope 原生接口，所以 LLM 和 embedding 的 base_url 分开配置。
# LLM 生成接口地址。
# 第 3 阶段的答案生成走 OpenAI 兼容 Chat Completions，而不是 embedding 的原生接口。
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 第 3 阶段开始引入“答案生成模型”。
# embedding 模型负责把文本变成向量，LLM 模型负责阅读召回证据并组织回答。
# 两者职责不同，所以在配置中分开命名，避免初学者误以为一个模型同时完成所有事情。
# LLM 生成模型名称。
# 当前已按用户确认使用 `qwen3.6-flash`，用于阅读召回证据并生成最终中文回答。
LLM_MODEL = "qwen3.6-flash"

# LLM 生成温度。
# 数值越低，回答越稳定；RAG 问答需要尽量基于证据，所以这里使用较低温度。
LLM_TEMPERATURE = 0.2

# LLM 单次生成的最大 token 数。
# 这个限制用于控制回答长度和生成耗时：RAG 答案应该短而准，
# 如果不限制，模型可能生成较多 completion / reasoning tokens，导致等待时间明显变长。
LLM_MAX_TOKENS = 1000

# 是否开启 Qwen 的思考模式。
# 当前 AI 助教问答以“基于召回证据短答”为主，不是复杂数学或长链路推理任务。
# 关闭思考模式可以减少 reasoning tokens，通常能显著降低等待时间和 token 成本。
LLM_ENABLE_THINKING = False

# 是否使用流式 Chat Completions。
# 流式输出不会让模型少生成 token，但可以更早拿到首段答案，让前端边生成边展示，
# 用户体感会比同步等待完整响应更快。
LLM_STREAM = True

# 默认召回证据数量。
# 前端和命令行不单独指定 top-k 时，会默认把最相关的 5 条证据交给 prompt。
DEFAULT_RETRIEVAL_TOP_K = 5

# prompt 中召回证据的最大字符数。
# 这个限制可以防止一次塞入过多上下文，避免 token 成本失控，也让答案更聚焦。
MAX_CONTEXT_CHARS = 6000
