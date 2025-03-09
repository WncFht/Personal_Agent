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
  - 支持多种LLM（OpenAI、HuggingFace模型）
- 简洁易用的命令行工具
- 友好的Gradio用户界面（开发中）

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

## 使用方法

### 初始化数据库

```bash
python main.py --setup
```

或者：

```bash
python scripts/setup_db.py
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
│   └── rss.db             # SQLite数据库
├── src/                   # 源代码
│   ├── rss/               # RSS相关模块
│   │   ├── __init__.py
│   │   ├── parser.py      # RSS解析器
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
│   └── ui/                # UI相关模块（开发中）
│       └── __init__.py
├── scripts/               # 脚本文件
│   ├── setup_db.py        # 数据库初始化
│   └── cli.py             # 命令行工具
├── tests/                 # 测试文件
│   └── test_rag.py        # RAG测试
├── requirements.txt       # 依赖项
├── README.md              # 项目说明
└── main.py                # 主入口
```

## 开发计划

- [x] 基础RSS解析器实现
- [x] 数据存储功能
- [x] 命令行工具
- [x] RAG系统实现
  - [x] 文本分块
  - [x] 向量化处理
  - [x] 混合检索策略
  - [x] LLM集成
- [ ] Gradio UI开发
- [ ] 性能优化

## 贡献

欢迎提交问题和拉取请求！

## 许可证

MIT 

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

### 5. 实施计划

1. **阶段一：基础改进**
   - 实现智能分句处理
   - 优化混合检索策略
   - 引入查询增强机制

2. **阶段二：高级功能**
   - 添加重排序模块
   - 优化提示词模板
   - 实现并行处理

3. **阶段三：性能优化**
   - 增强缓存策略
   - 优化内存使用
   - 提高整体响应速度

### 6. 依赖更新

```txt
# 新增依赖
modelscope>=1.9.5
sentence-transformers>=2.2.2
```

以上改进方案基于TinyRAG的优秀实践，结合RSS-RAG的特点进行了定制化设计，将显著提升系统的检索精度和响应速度。 