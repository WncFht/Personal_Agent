# RSS-RAG 智能问答系统

基于RSS数据源的检索增强生成(RAG)系统，通过Gradio提供用户界面，实现智能问答功能。

## 功能特点

- 支持添加、管理和更新RSS源
  - 单个添加RSS源
  - 批量导入OPML格式的RSS源
  - 分类管理
- 自动解析和存储RSS条目
- 基于向量检索的智能问答
  - 混合检索策略（BM25 + 向量检索）
  - 元数据过滤（按时间、来源等）
  - 支持多种LLM（OpenAI、HuggingFace模型、本地小型模型）
- 简洁易用的命令行工具
- 友好的Gradio用户界面（开发中）
- 支持流式输出回答

## 安装

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/rss-rag.git
cd rss-rag
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 下载必要的模型：

```bash
# 下载Embedding模型
git clone https://huggingface.co/BAAI/bge-base-zh-v1.5 models/bge-base-zh-v1.5

# 下载LLM模型（可选，如果使用本地模型）
git clone https://huggingface.co/Qwen/Qwen-1_8B-Chat models/Qwen-1_8B-Chat
```

## 使用方法

### 初始化数据库

```bash
python main.py --setup
```

或者：

```bash
python scripts/cli.py setup
```

### 命令行工具

RSS-RAG提供了一个功能丰富的命令行工具，用于管理RSS源和条目：

```bash
python scripts/cli.py [命令] [参数]
```

可用命令：

- `setup`: 初始化数据库
- `add`: 添加RSS源
- `import`: 导入OPML文件
- `list`: 列出所有RSS源
- `delete`: 删除RSS源
- `update`: 更新所有RSS源
- `entries`: 列出条目
- `view`: 查看条目详情
- `mark-read`: 标记条目为已读
- `export`: 导出数据
- `ask`: 提问问题（RAG功能）

#### 示例

添加RSS源：

```bash
python scripts/cli.py add https://example.com/feed.xml
```

导入OPML文件：

```bash
python scripts/cli.py import data/feeds.opml
```

导入OPML并获取条目：

```bash
python scripts/cli.py import data/feeds.opml --fetch
```

导入OPML并更新所有源：

```bash
python scripts/cli.py import data/feeds.opml --update
```

列出所有RSS源：

```bash
python scripts/cli.py list
```

查看条目详情：

```bash
python scripts/cli.py view 1 --content
```

标记所有条目为已读：

```bash
python scripts/cli.py mark-read --all
```

导出数据：

```bash
python scripts/cli.py export --output data/backup.json
```

提问问题：

```bash
python scripts/cli.py ask "Python有什么特点？"
```

按特定来源提问：

```bash
python scripts/cli.py ask "人工智能最新进展？" --feed 2
```

按时间范围提问：

```bash
python scripts/cli.py ask "数据科学入门" --days 7
```

## 项目结构

```
rss-rag/
├── data/                  # 数据存储目录
│   ├── rss.db             # SQLite数据库
│   └── vector_store/      # 向量数据库存储
├── models/                # 模型存储目录
│   ├── bge-base-zh-v1.5/  # BGE Embedding模型
│   └── Qwen-1_8B-Chat/    # Qwen LLM模型（可选）
├── src/                   # 源代码
│   ├── rss/               # RSS相关模块
│   │   ├── __init__.py
│   │   ├── parser.py      # RSS解析器
│   │   ├── opml_parser.py # OPML解析器
│   │   ├── models.py      # 数据模型
│   │   └── storage.py     # 数据存储
│   ├── rag/               # RAG相关模块
│   │   ├── __init__.py
│   │   ├── config.py      # 配置类
│   │   ├── rss_rag.py     # RAG主类
│   │   ├── embedding/     # 向量化模块
│   │   ├── retrieval/     # 检索模块
│   │   ├── llm/           # LLM模块
│   │   └── utils/         # 工具函数
│   ├── utils/             # 通用工具函数
│   └── ui/                # UI相关模块（开发中）
├── scripts/               # 脚本文件
│   ├── setup_db.py        # 数据库初始化
│   └── cli.py             # 命令行工具
├── tests/                 # 测试文件
├── config/                # 配置文件目录
├── requirements.txt       # 依赖项
├── requirements-dev.txt   # 开发依赖项
└── main.py                # 主入口
```

## 系统架构

RSS-RAG系统由以下主要组件构成：

### 1. RSS处理模块
- **RSSParser**: 负责解析RSS源和条目
- **OPMLParser**: 解析OPML文件，支持批量导入RSS源
- **RSSStorage**: 管理SQLite数据库，存储RSS源和条目

### 2. RAG核心模块
- **RSSRAG**: 系统主类，协调各组件工作
- **EmbeddingModel**: 文本向量化模型，基于BGE等模型
- **HybridRetriever**: 混合检索器，结合BM25和向量检索
- **TextSplitter**: 文本分块工具，优化长文本处理

### 3. LLM模块
- **BaseLLM**: LLM基类，定义通用接口
- **OpenAILLM**: OpenAI API集成
- **HuggingFaceLLM**: HuggingFace模型集成
- **TinyLLM**: 本地小型模型集成，支持多种模型格式

### 4. 用户界面
- 命令行工具: 提供完整的RSS管理和问答功能
- Gradio UI: 友好的Web界面（开发中）

## 高级功能

### 1. 混合检索策略

RSS-RAG采用混合检索策略，结合BM25和向量检索的优势：

```python
def search(self,
          query: str,
          top_k: int = 3,
          metadata_filters: Optional[Dict] = None,
          weights: Optional[Dict[str, float]] = None):
    if weights is None:
        weights = {'bm25': 0.3, 'vector': 0.7}
    # ...
```

### 2. 增强RAG提示词模板

系统支持两种RAG提示词模板：

1. **原始模板**：直接基于检索结果生成回答
2. **增强模板**：先生成初步回答，再基于检索结果修正，提高回答质量

```python
# 增强RAG提示词模板
ENHANCED_RAG_PROMPT_TEMPLATE = """参考信息：
{context}
---
我的问题或指令：
{question}
---
我的回答：
{answer}
---
请根据上述参考信息回答和我的问题或指令，修正我的回答。前面的参考信息和我的回答可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你修正的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复。
你修正的回答:"""
```

### 3. 流式输出

系统支持流式生成回答，提供更好的用户体验：

```python
def answer_stream(self, 
               query: str,
               feed_id: Optional[int] = None,
               date_range: Optional[Tuple[datetime, datetime]] = None,
               top_k: Optional[int] = None) -> Iterator[str]:
    # ...
```

### 4. 并行处理

支持并行处理RSS条目，提高数据处理效率：

```python
def _process_entries_parallel(self, entries: List[Entry]):
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(self.process_entry, entry): entry for entry in entries}
        for future in tqdm(as_completed(futures), total=len(entries)):
            try:
                future.result()
            except Exception as exc:
                logger.error(f"处理条目时出错: {exc}")
```

## 配置说明

系统通过`RAGConfig`类管理配置：

```python
class RAGConfig:
    def __init__(self,
                 vector_store_path: str = "data/vector_store",
                 embedding_model_path: str = "models/bge-base-zh-v1.5",
                 llm_type: str = "openai",
                 llm_model: str = "gpt-3.5-turbo",
                 llm_api_key: Optional[str] = None,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 top_k: int = 5,
                 use_enhanced_rag: bool = True,
                 use_parallel: bool = True):
    # ...
```

主要配置项：
- `vector_store_path`: 向量数据库存储路径
- `embedding_model_path`: Embedding模型路径
- `llm_type`: LLM类型（openai、huggingface、tiny）
- `llm_model`: 使用的模型名称
- `chunk_size`: 文本分块大小
- `use_enhanced_rag`: 是否使用增强RAG策略
- `use_parallel`: 是否使用并行处理

## 模型支持

### 1. Embedding模型
- 默认使用BGE模型：[BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5)
- 支持自定义Embedding模型

### 2. LLM模型
- **OpenAI**: 支持GPT-3.5、GPT-4等模型
- **HuggingFace**: 支持HuggingFace上的开源模型
- **TinyLLM**: 支持本地部署的小型模型，包括：
  - Qwen系列模型
  - ChatGLM系列模型
  - Llama/Mistral系列模型
  - Baichuan系列模型

## RAG系统改进方案

基于对TinyRAG系统的分析和借鉴，我们对RSS-RAG系统进行了以下改进：

### 1. 文本处理优化

#### 1.1 智能分句处理

当前实现：
```python
# 当前的TextSplitter实现
class TextSplitter:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = "\n",
        keep_separator: bool = False
    ):
        # ...
```

改进方案：
```python
# 引入TinyRAG的SentenceSplitter
class EnhancedTextSplitter:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        use_model: bool = False,
        model_path: str = "damo/nlp_bert_document-segmentation_chinese-base",
        device: str = "cuda"
    ):
        # ...
```

改进内容：
- 引入基于BERT的文档分割模型，提高中文文本的分句质量
- 增加语义感知的分块策略，保持语义完整性
- 优化长文本处理逻辑，避免语义割裂

### 2. 检索系统增强

#### 2.1 混合检索策略优化

当前实现：
```python
# 当前的HybridRetriever实现
def search(self,
          query: str,
          top_k: int = 3,
          metadata_filters: Optional[Dict] = None,
          weights: Optional[Dict[str, float]] = None):
    if weights is None:
        weights = {'bm25': 0.3, 'vector': 0.7}
    # ...
```

改进方案：
```python
# 引入TinyRAG的多阶段检索策略
def search(self,
          query: str,
          top_k: int = 3,
          metadata_filters: Optional[Dict] = None,
          weights: Optional[Dict[str, float]] = None):
    # 第一阶段：BM25和向量混合召回
    recall_results = self._hybrid_recall(query, top_k * 2)
    
    # 第二阶段：重排序
    if self.use_reranker:
        reranked_results = self._rerank(query, recall_results)
        return reranked_results[:top_k]
    # ...
```

改进内容：
- 引入多阶段检索策略，先召回后重排
- 添加基于BGE-M3的重排序模块，提高检索精度
- 优化BM25和向量检索的权重动态调整机制

#### 2.2 元数据过滤增强

当前实现：
```python
# 当前的元数据过滤实现
if metadata_filters:
    match = True
    for key, value in metadata_filters.items():
        if callable(value):
            if key not in doc.metadata or not value(doc.metadata[key]):
                match = False
                break
        # ...
```

改进方案：
```python
# 增强的元数据过滤
def _apply_filters(self, doc, metadata_filters):
    # 基础过滤
    if not self._basic_filter_match(doc, metadata_filters):
        return False
        
    # 时间范围过滤
    if not self._time_filter_match(doc, metadata_filters):
        return False
        
    # 来源过滤
    if not self._source_filter_match(doc, metadata_filters):
        return False
        
    return True
```

改进内容：
- 优化元数据过滤逻辑，提高过滤效率
- 增加时间范围的智能解析，支持相对时间表达
- 添加多级过滤策略，提高检索精度

### 3. RAG流程优化

#### 3.1 查询增强

当前实现：
```python
# 当前的查询处理
def answer(self, query: str, ...):
    # 直接使用原始查询
    search_results = self.search(query, ...)
    # ...
```

改进方案：
```python
# 引入TinyRAG的查询增强策略
def answer(self, query: str, ...):
    # 先用LLM生成初步回答
    initial_answer = self.llm.generate(query)
    
    # 增强查询
    enhanced_query = query + initial_answer + query
    
    # 使用增强查询检索
    search_results = self.search(enhanced_query, ...)
    # ...
```

改进内容：
- 引入查询增强策略，提高检索相关性
- 利用LLM初步回答扩展查询内容
- 优化查询-文档匹配度计算

#### 3.2 提示词模板优化

当前实现：
```python
# 当前的提示词模板
RAG_PROMPT_TEMPLATE = """参考信息：
{context}
---
问题：
{question}
---
请根据上述参考信息回答问题。如果参考信息不足以回答问题，请说明无法回答。回答要简洁、准确，并尽可能基于参考信息。
"""
```

改进方案：
```python
# 引入TinyRAG的提示词模板
RAG_PROMPT_TEMPLATE = """参考信息：
{context}
---
我的问题或指令：
{question}
---
我的回答：
{answer}
---
请根据上述参考信息回答和我的问题或指令，修正我的回答。前面的参考信息和我的回答可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你修正的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复。
你修正的回答:"""
```

改进内容：
- 引入"我的回答"部分，让LLM对初步回答进行修正
- 优化指令描述，提高LLM对参考信息的利用效率
- 增加语言匹配要求，提升多语言支持

### 4. 性能优化

#### 4.1 并行处理

当前实现：
```python
# 当前的处理方式
def process_entries(self, entries: List[Entry]):
    for entry in entries:
        self.process_entry(entry)
    # ...
```

改进方案：
```python
# 引入TinyRAG的并行处理
def process_entries(self, entries: List[Entry]):
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(self.process_entry, entry): entry for entry in entries}
        for future in tqdm(as_completed(futures), total=len(entries)):
            try:
                future.result()
            except Exception as exc:
                logger.error(f"处理条目时出错: {exc}")
    # ...
```

改进内容：
- 引入多线程并行处理，提高数据处理效率
- 优化资源利用，自动适应CPU核心数
- 增加错误处理和进度显示

#### 4.2 缓存优化

当前实现：
```python
# 当前的缓存实现
@lru_cache(maxsize=1000)
def _get_embedding(self, text: str) -> np.ndarray:
    return self.embedding_model.encode(text)
```

改进方案：
```python
# 增强的缓存策略
class EnhancedCache:
    def __init__(self, cache_size=1000, ttl=3600):
        self.cache = {}
        self.cache_size = cache_size
        self.ttl = ttl
        self.access_count = {}
        
    def get(self, key):
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
        
    def set(self, key, value):
        if len(self.cache) >= self.cache_size:
            self._evict_least_used()
        self.cache[key] = value
        self.access_count[key] = 1
```

改进内容：
- 引入更智能的缓存淘汰策略，提高缓存命中率
- 增加缓存过期时间设置，保证数据新鲜度
- 优化内存使用，避免缓存过大占用资源

## 开发计划

- [x] 基础RSS解析器实现
- [x] 数据存储功能
- [x] 命令行工具
- [x] RAG系统实现
  - [x] 文本分块
  - [x] 向量化处理
  - [x] 混合检索策略
  - [x] LLM集成
  - [x] 流式输出
- [ ] 实现改进方案
  - [ ] 智能分句处理
  - [ ] 多阶段检索策略
  - [ ] 查询增强
  - [ ] 缓存优化
- [ ] Gradio UI开发
  - [ ] RSS源管理界面
  - [ ] 条目浏览界面
  - [ ] 问答界面
- [ ] 性能优化
  - [ ] 减少内存占用
  - [ ] 提高检索速度
  - [ ] 优化LLM调用

## 贡献

欢迎提交问题和拉取请求！

## 许可证

MIT 