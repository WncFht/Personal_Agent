# RSS-RAG 智能问答系统

基于RSS数据源的检索增强生成(RAG)系统，通过Gradio提供用户界面，实现智能问答功能。

![alt text](docs/image.png)

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
- 友好的Gradio用户界面
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

### 启动Web界面

```bash
python run_ui.py
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
│   └── ui/                # UI相关模块
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
- Gradio UI: 友好的Web界面

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

## 项目改进待办清单

### 核心架构优化

- [ ] **模块化重构**
  - [ ] 创建 `core` 模块，包含所有共享的基础功能
    - [ ] 实现 `ServiceManager` 类作为核心服务容器，管理所有服务实例
    - [ ] 设计 `EventBus` 系统，用于模块间解耦通信
    - [ ] 创建统一的异常处理机制和错误码系统
  - [ ] 重构 RSS 模块
    - [ ] 创建 `RSSManager` 类，封装所有 RSS 源管理功能
    - [ ] 将 `parser.py` 和 `opml_parser.py` 合并为 `parsers` 子模块
    - [ ] 优化 `storage.py`，减少重复代码，提高查询效率
  - [ ] 重构 RAG 模块
    - [ ] 优化 `RSSRAG` 类，减少职责，遵循单一职责原则
    - [ ] 将检索逻辑抽象为独立的 `RetrievalService`
    - [ ] 创建 `ContextManager` 处理上下文管理和会话状态

- [ ] **接口统一与解耦**
  - [ ] 设计统一的服务接口层
    - [ ] 创建 `RSSService` 接口，统一 RSS 相关操作
    - [ ] 创建 `RAGService` 接口，统一问答相关操作
    - [ ] 创建 `ConfigService` 接口，统一配置管理
  - [ ] 实现依赖注入模式
    - [ ] 使用工厂模式创建服务实例
    - [ ] 减少全局变量使用，通过依赖注入传递服务实例
    - [ ] 设计服务生命周期管理机制

- [ ] **数据层优化**
  - [ ] 设计统一的数据访问层
    - [ ] 创建 `Repository` 抽象类，定义通用数据操作接口
    - [ ] 实现 `RSSRepository` 和 `EntryRepository` 类
    - [ ] 添加数据验证和转换逻辑
  - [ ] 优化数据模型
    - [ ] 扩展 `Feed` 和 `Entry` 模型，支持更多元数据
    - [ ] 实现模型间关系映射
    - [ ] 添加数据模型验证机制

### UI 与 CLI 改进

- [ ] **UI 架构重构**
  - [ ] 创建 `UIManager` 类，管理 Gradio 界面生命周期
    - [ ] 将 `create_ui()` 拆分为多个组件创建函数
    - [ ] 实现 UI 组件注册和管理机制
    - [ ] 设计 UI 状态管理系统
  - [ ] 分离业务逻辑和界面代码
    - [ ] 创建 `ViewModel` 类处理 UI 数据转换和验证
    - [ ] 实现 UI 事件处理器，连接界面和服务层
    - [ ] 使用装饰器统一处理 UI 异常和错误提示

- [ ] **CLI 架构重构**
  - [ ] 创建 `CLIManager` 类，统一管理命令行接口
    - [ ] 实现命令注册和发现机制
    - [ ] 设计插件式命令扩展系统
    - [ ] 添加命令帮助生成器
  - [ ] 优化命令执行流程
    - [ ] 创建统一的参数解析和验证机制
    - [ ] 实现命令执行上下文，传递共享状态
    - [ ] 添加命令执行钩子，支持前置和后置处理

- [ ] **共享组件提取**
  - [ ] 创建 `common` 模块，包含 UI 和 CLI 共享的组件
    - [ ] 实现 `ProgressReporter` 接口，统一进度报告
    - [ ] 创建 `OutputFormatter` 类，支持多种输出格式
    - [ ] 设计 `CommandRegistry` 系统，实现命令复用
  - [ ] 统一配置管理
    - [ ] 扩展 `ConfigManager`，支持 UI 和 CLI 共享配置
    - [ ] 实现配置验证和迁移机制
    - [ ] 添加配置变更通知系统

### 性能与可维护性优化

- [ ] **异步处理框架**
  - [ ] 设计统一的异步任务系统
    - [ ] 创建 `TaskManager` 类，管理后台任务
    - [ ] 实现任务队列和优先级机制
    - [ ] 添加任务状态监控和取消功能
  - [ ] 重构 IO 密集型操作
    - [ ] 使用 `asyncio` 重写 RSS 解析和更新逻辑
    - [ ] 实现异步数据库访问
    - [ ] 优化文件操作，减少阻塞

- [ ] **缓存与性能优化**
  - [ ] 实现统一的缓存系统
    - [ ] 创建 `CacheManager` 类，管理多级缓存
    - [ ] 设计缓存键生成和失效策略
    - [ ] 添加缓存统计和监控功能
  - [ ] 优化数据处理流程
    - [ ] 实现数据批处理机制
    - [ ] 添加增量处理支持
    - [ ] 优化内存使用，减少大对象创建

- [ ] **测试与文档**
  - [ ] 建立测试框架
    - [ ] 创建单元测试基类和工具函数
    - [ ] 实现测试数据生成器
    - [ ] 添加关键组件的测试用例
  - [ ] 完善文档系统
    - [ ] 添加核心类和方法的文档字符串
    - [ ] 创建架构和设计文档
    - [ ] 实现自动文档生成

### 功能增强（基于新架构）

- [ ] **RSS 功能增强**
  - [ ] 实现 RSS 源健康监控
  - [ ] 添加内容去重机制
  - [ ] 优化全文获取功能

- [ ] **RAG 策略优化**
  - [ ] 实现基础查询改写功能
  - [ ] 添加上下文压缩机制
  - [ ] 优化检索相关性评分

- [ ] **用户体验改进**
  - [ ] 实现基础的会话历史管理
  - [ ] 添加简单的数据可视化功能
  - [ ] 优化长时间操作的进度反馈

### 实施路线图

**第一阶段：核心架构重构**
1. 创建 `ServiceManager` 和核心服务接口
2. 实现基础的事件系统
3. 重构 RSS 和 RAG 核心模块
4. 建立统一的数据访问层

**第二阶段：UI 和 CLI 改进**
1. 创建 `UIManager` 和 `CLIManager`
2. 分离业务逻辑和界面代码
3. 提取共享组件到 `common` 模块
4. 统一配置管理机制

**第三阶段：性能优化**
1. 实现异步任务系统
2. 重构 IO 密集型操作
3. 添加缓存机制
4. 优化数据处理流程

**第四阶段：测试与文档**
1. 建立测试框架
2. 添加核心组件测试
3. 完善文档系统
4. 实现自动文档生成

**第五阶段：功能增强**
1. 基于新架构实现 RSS 功能增强
2. 优化 RAG 策略
3. 改进用户体验

## 多模态 Agent 架构设计

基于现有的 RSS-RAG 系统，我们将扩展为一个功能更加强大的多模态 Agent 架构，集成 RAG 检索、RSS 订阅、多模态识别、记忆功能和工具调用能力。

### 核心架构设计

- [ ] **Agent 核心框架**
  - [ ] 创建 `AgentCore` 模块，作为整个系统的中枢
    - [ ] 实现 `Agent` 基类，定义通用的 Agent 接口和生命周期
    - [ ] 设计 `AgentMemory` 系统，支持短期和长期记忆
    - [ ] 创建 `AgentPlanner` 组件，负责任务规划和分解
  - [ ] 实现 `MultiModalAgent` 类，支持文本、图像、音频等多种输入
    - [ ] 设计模态处理管道，统一处理不同类型的输入
    - [ ] 实现模态融合机制，整合多模态信息
  - [ ] 创建 `ToolAgent` 类，专注于工具调用和执行
    - [ ] 设计工具注册和发现机制
    - [ ] 实现工具调用安全沙箱
    - [ ] 支持工具执行结果的处理和反馈

- [ ] **工具系统设计**
  - [ ] 创建统一的工具接口层
    - [ ] 设计 `Tool` 基类，定义工具的通用接口
    - [ ] 实现 `ToolRegistry` 类，管理所有可用工具
    - [ ] 创建工具验证和安全检查机制
  - [ ] 实现核心工具集
    - [ ] `RSSToolkit`: RSS 源管理和内容获取工具
    - [ ] `RAGToolkit`: 知识检索和问答工具
    - [ ] `FileToolkit`: 文件操作和处理工具
    - [ ] `ImageToolkit`: 图像识别和处理工具
    - [ ] `WebToolkit`: 网页访问和内容提取工具
    - [ ] `APIToolkit`: 外部 API 调用工具

- [ ] **多模态处理系统**
  - [ ] 创建 `ModalityProcessor` 框架
    - [ ] 实现 `TextProcessor` 处理文本输入
    - [ ] 实现 `ImageProcessor` 处理图像输入
    - [ ] 实现 `AudioProcessor` 处理音频输入
    - [ ] 实现 `FileProcessor` 处理文件输入
  - [ ] 设计模态转换机制
    - [ ] 图像到文本的描述转换
    - [ ] 音频到文本的转录
    - [ ] 文件内容的结构化提取
  - [ ] 实现多模态融合策略
    - [ ] 基于注意力机制的模态融合
    - [ ] 跨模态上下文理解

- [ ] **记忆系统**
  - [ ] 设计分层记忆架构
    - [ ] 实现 `WorkingMemory` 管理当前会话状态
    - [ ] 实现 `EpisodicMemory` 存储历史交互
    - [ ] 实现 `SemanticMemory` 存储知识和概念
  - [ ] 创建记忆检索机制
    - [ ] 基于相似度的记忆检索
    - [ ] 基于时间的记忆衰减
    - [ ] 基于重要性的记忆强化
  - [ ] 实现记忆整合与总结
    - [ ] 定期记忆压缩和抽象
    - [ ] 记忆冲突解决策略
    - [ ] 记忆到知识的转化

### 集成与扩展

- [ ] **RSS-RAG 集成**
  - [ ] 将现有 RSS 模块重构为工具形式
    - [ ] 创建 `RSSFeedTool` 管理 RSS 源
    - [ ] 实现 `RSSContentTool` 获取和处理 RSS 内容
    - [ ] 设计 `RSSMonitorTool` 监控 RSS 更新
  - [ ] 将 RAG 系统集成为知识工具
    - [ ] 创建 `RAGQueryTool` 执行知识检索
    - [ ] 实现 `KnowledgeBaseTool` 管理知识库
    - [ ] 设计 `ContextEnhancementTool` 增强查询上下文

- [ ] **LLM 集成**
  - [ ] 设计模型抽象层
    - [ ] 创建 `ModelInterface` 统一不同 LLM 的接口
    - [ ] 实现模型切换和回退机制
    - [ ] 支持模型并行调用和结果聚合
  - [ ] 支持多种 LLM 提供商
    - [ ] 本地模型（如 Llama、Mistral）
    - [ ] API 模型（如 OpenAI、Claude）
    - [ ] 自定义模型部署

- [ ] **UI 与交互**
  - [ ] 创建统一的交互接口
    - [ ] 设计 `AgentUI` 基类，定义通用交互接口
    - [ ] 实现 `WebUI` 提供 Web 界面
    - [ ] 实现 `CLIUI` 提供命令行界面
    - [ ] 实现 `APIUI` 提供 API 接口
  - [ ] 设计多模态输入处理
    - [ ] 支持文本、图像、音频、文件上传
    - [ ] 实现拖放和复制粘贴功能
    - [ ] 支持语音输入和输出
  - [ ] 创建交互式可视化
    - [ ] 实现 Agent 思考过程可视化
    - [ ] 设计工具调用流程展示
    - [ ] 创建记忆网络可视化

### 实施路线图

**第一阶段：基础架构**
1. 实现 `AgentCore` 和基本的 Agent 类
2. 设计并实现工具系统
3. 创建简单的记忆系统
4. 集成现有的 RSS-RAG 功能

**第二阶段：多模态支持**
1. 实现各种模态处理器
2. 设计模态融合机制
3. 增强记忆系统，支持多模态记忆
4. 添加基本的多模态工具

**第三阶段：高级功能**
1. 实现复杂的工具调用和编排
2. 增强记忆检索和利用
3. 添加自主规划和学习能力
4. 实现高级的多模态理解

**第四阶段：UI 和部署**
1. 开发 Web 和命令行界面
2. 实现 API 服务
3. 优化性能和资源使用
4. 添加监控和日志系统

### 技术栈选择

- **核心框架**：Python 3.10+
- **LLM 集成**：支持 OpenAI、Anthropic、HuggingFace、本地模型
- **多模态处理**：
  - 图像：使用 CLIP、Pix2Struct 等模型
  - 音频：使用 Whisper 等模型
  - 文件：支持 PDF、Word、Excel 等格式解析
- **存储系统**：
  - 向量数据库：FAISS、Chroma、Milvus
  - 关系数据库：SQLite、PostgreSQL
  - 文件存储：本地文件系统、S3 兼容存储
- **Web 界面**：Gradio、Streamlit 或 FastAPI + React
- **部署选项**：Docker 容器、Python 包、Web 服务

### 示例用例

1. **智能信息助手**：
   - 订阅多个 RSS 源，自动整理和总结内容
   - 回答用户关于订阅内容的问题
   - 支持上传文档和图片进行分析和问答

2. **研究辅助工具**：
   - 收集和整理研究论文和资料
   - 分析图表和数据
   - 生成研究摘要和报告

3. **内容创作助手**：
   - 收集相关领域的最新信息
   - 协助内容规划和创作
   - 提供参考资料和建议

4. **个人知识管理**：
   - 整理和索引个人知识库
   - 连接不同来源的信息
   - 提供智能检索和推荐

## 贡献

欢迎提交问题和拉取请求！

## 许可证

MIT 

## UI 代码评审结果

### app.py 评审

#### 优点：
1. **模块化设计**：代码结构清晰，功能按模块划分，便于维护和扩展
2. **完善的错误处理**：大部分函数都有适当的错误处理机制
3. **良好的用户体验**：提供了流式输出、参考信息显示等增强用户体验的功能
4. **数据可视化**：使用 matplotlib 提供了数据统计和可视化功能
5. **配置管理**：使用 ConfigManager 进行配置管理，支持配置热重载

#### 改进建议：
1. **代码冗长**：部分函数过长，如 `create_ui()` 和 `get_feed_stats()`，应拆分为更小的函数
2. **异常处理不一致**：有些地方使用 try-except，有些地方直接返回错误信息，应统一处理方式
3. **全局变量使用**：过度依赖全局变量（rag_system, storage 等），应考虑使用依赖注入或类封装
4. **UI 组件耦合**：UI 组件与业务逻辑耦合度高，难以单独测试
5. **缺少注释**：部分复杂逻辑缺少详细注释，增加维护难度
6. **硬编码问题**：存在硬编码的配置项和路径，应移至配置文件
7. **性能优化空间**：数据加载和处理未充分利用异步和并行处理
8. **缺少进度反馈**：长时间操作（如更新所有 RSS 源）缺少进度反馈
9. **UI 响应性**：部分操作会阻塞 UI 线程，影响用户体验
10. **国际化支持**：界面文本硬编码，不利于国际化

### cli.py 评审

#### 优点：
1. **功能完整**：提供了丰富的命令行功能，覆盖了系统的主要操作
2. **参数设计合理**：命令行参数设计符合常规习惯，易于使用
3. **帮助信息清晰**：每个命令都有简洁明了的帮助信息
4. **模块化结构**：每个命令对应独立的函数，结构清晰

#### 改进建议：
1. **代码重复**：与 app.py 存在功能重复，应提取共享逻辑到公共模块
2. **错误处理简单**：错误处理较为简单，可增加更详细的错误信息和日志
3. **缺少进度显示**：长时间操作缺少进度显示，用户体验不佳
4. **配置管理不一致**：与 app.py 的配置管理方式不一致，应统一
5. **测试覆盖不足**：缺少单元测试和集成测试
6. **文档不足**：缺少详细的使用文档和示例
7. **输出格式单一**：输出格式固定，不支持 JSON/CSV 等结构化输出
8. **交互性有限**：缺少交互式命令模式，用户体验有限

### 通用改进建议

1. **代码复用**：提取 app.py 和 cli.py 的共享逻辑到公共模块
2. **依赖注入**：减少全局变量使用，采用依赖注入模式
3. **异步处理**：使用异步处理提高 UI 响应性和命令行性能
4. **测试覆盖**：增加单元测试和集成测试
5. **文档完善**：完善代码注释和用户文档
6. **国际化支持**：提取界面文本，支持多语言
7. **配置统一**：统一配置管理方式
8. **UI 组件化**：将 UI 组件与业务逻辑分离，提高可测试性和可维护性
9. **性能优化**：优化数据加载和处理性能
10. **用户体验提升**：增加进度反馈、交互性和可访问性

## 架构设计详细说明

### 核心架构设计

#### 服务层设计

新的架构将采用服务导向设计，主要包含以下核心服务：

```
core/
├── services/
│   ├── service_manager.py     # 服务管理器
│   ├── base_service.py        # 服务基类
│   ├── rss_service.py         # RSS服务接口
│   ├── rag_service.py         # RAG服务接口
│   ├── config_service.py      # 配置服务接口
│   └── event_service.py       # 事件服务接口
├── events/
│   ├── event_bus.py           # 事件总线
│   ├── event.py               # 事件基类
│   └── handlers/              # 事件处理器
├── exceptions/
│   ├── base_exception.py      # 异常基类
│   ├── error_codes.py         # 错误码定义
│   └── handlers.py            # 异常处理器
└── utils/
    ├── logging.py             # 日志工具
    ├── validation.py          # 数据验证工具
    └── helpers.py             # 通用辅助函数
```

**ServiceManager** 是整个系统的核心，负责管理所有服务的生命周期：

```python
class ServiceManager:
    """服务管理器，负责管理所有服务的生命周期"""
    
    def __init__(self):
        self._services = {}
        self._initialized = False
    
    def register(self, service_type, service_instance):
        """注册服务"""
        self._services[service_type] = service_instance
        return self
    
    def get(self, service_type):
        """获取服务实例"""
        if service_type not in self._services:
            raise ServiceNotFoundError(f"Service {service_type} not registered")
        return self._services[service_type]
    
    def initialize(self):
        """初始化所有服务"""
        if self._initialized:
            return
            
        # 按依赖顺序初始化服务
        for service in self._services.values():
            service.initialize(self)
            
        self._initialized = True
    
    def shutdown(self):
        """关闭所有服务"""
        # 按依赖顺序的逆序关闭服务
        for service in reversed(list(self._services.values())):
            service.shutdown()
            
        self._initialized = False
```

#### 数据层设计

数据层将采用仓储模式（Repository Pattern），将数据访问逻辑与业务逻辑分离：

```
data/
├── repositories/
│   ├── base_repository.py     # 仓储基类
│   ├── rss_repository.py      # RSS仓储
│   └── entry_repository.py    # 条目仓储
├── models/
│   ├── base_model.py          # 模型基类
│   ├── feed.py                # Feed模型
│   └── entry.py               # Entry模型
└── db/
    ├── connection.py          # 数据库连接管理
    ├── migrations/            # 数据库迁移脚本
    └── queries/               # SQL查询定义
```

**Repository** 基类定义了通用的数据操作接口：

```python
class Repository:
    """仓储基类，定义通用数据操作接口"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def get_by_id(self, id):
        """根据ID获取实体"""
        raise NotImplementedError
    
    def get_all(self, filters=None, order_by=None, limit=None, offset=None):
        """获取所有实体"""
        raise NotImplementedError
    
    def add(self, entity):
        """添加实体"""
        raise NotImplementedError
    
    def update(self, entity):
        """更新实体"""
        raise NotImplementedError
    
    def delete(self, id):
        """删除实体"""
        raise NotImplementedError
```

### UI 与 CLI 架构设计

#### UI 架构

UI 层将采用 MVVM（Model-View-ViewModel）模式，将界面与业务逻辑分离：

```
ui/
├── manager.py                 # UI管理器
├── components/                # UI组件
│   ├── base_component.py      # 组件基类
│   ├── question_panel.py      # 问答面板
│   ├── feed_manager.py        # RSS源管理面板
│   └── settings_panel.py      # 设置面板
├── viewmodels/                # 视图模型
│   ├── base_viewmodel.py      # 视图模型基类
│   ├── question_viewmodel.py  # 问答视图模型
│   └── feed_viewmodel.py      # RSS源视图模型
├── events/                    # UI事件
│   ├── ui_event.py            # UI事件基类
│   └── handlers.py            # UI事件处理器
└── decorators/                # UI装饰器
    ├── error_handler.py       # 错误处理装饰器
    └── progress.py            # 进度报告装饰器
```

**UIManager** 负责管理 Gradio 界面的生命周期：

```python
class UIManager:
    """UI管理器，负责管理Gradio界面的生命周期"""
    
    def __init__(self, service_manager):
        self.service_manager = service_manager
        self.components = {}
        self.app = None
    
    def register_component(self, name, component):
        """注册UI组件"""
        self.components[name] = component
        return self
    
    def create_ui(self):
        """创建Gradio界面"""
        with gr.Blocks(title="RSS-RAG 智能问答系统", theme=gr.themes.Soft()) as app:
            gr.Markdown("# RSS-RAG 智能问答系统")
            
            with gr.Tabs() as tabs:
                # 创建各个组件的UI
                for component in self.components.values():
                    component.create(app, tabs)
            
            self.app = app
        return app
    
    def launch(self, host="127.0.0.1", port=7860, share=False):
        """启动UI服务"""
        if not self.app:
            self.create_ui()
        
        self.app.launch(server_name=host, server_port=port, share=share)
```

#### CLI 架构

CLI 层将采用命令模式，将命令与执行逻辑分离：

```
cli/
├── manager.py                 # CLI管理器
├── commands/                  # 命令定义
│   ├── base_command.py        # 命令基类
│   ├── rss_commands.py        # RSS相关命令
│   └── rag_commands.py        # RAG相关命令
├── context.py                 # 命令执行上下文
├── formatters/                # 输出格式化器
│   ├── base_formatter.py      # 格式化器基类
│   ├── text_formatter.py      # 文本格式化器
│   └── json_formatter.py      # JSON格式化器
└── progress/                  # 进度报告
    ├── reporter.py            # 进度报告器
    └── bar.py                 # 进度条实现
```

**CLIManager** 负责管理命令行接口：

```python
class CLIManager:
    """CLI管理器，负责管理命令行接口"""
    
    def __init__(self, service_manager):
        self.service_manager = service_manager
        self.commands = {}
        self.parser = None
    
    def register_command(self, name, command_class):
        """注册命令"""
        self.commands[name] = command_class
        return self
    
    def setup_parser(self):
        """设置命令行解析器"""
        parser = argparse.ArgumentParser(description='RSS-RAG 命令行工具')
        parser.add_argument('--config', help='配置文件路径')
        
        subparsers = parser.add_subparsers(dest='command', help='子命令')
        
        # 为每个命令创建子解析器
        for name, command_class in self.commands.items():
            command = command_class(self.service_manager)
            command.setup_parser(subparsers)
        
        self.parser = parser
        return parser
    
    def run(self, args=None):
        """运行CLI"""
        if not self.parser:
            self.setup_parser()
            
        args = self.parser.parse_args(args)
        
        if not args.command:
            self.parser.print_help()
            return
        
        # 创建命令执行上下文
        context = CommandContext(self.service_manager, args)
        
        # 执行命令
        command = self.commands[args.command](self.service_manager)
        return command.execute(context)
```

### 共享组件设计

共享组件将放在 `common` 模块中，供 UI 和 CLI 共同使用：

```
common/
├── progress/
│   ├── reporter.py            # 进度报告接口
│   ├── console_reporter.py    # 控制台进度报告
│   └── ui_reporter.py         # UI进度报告
├── output/
│   ├── formatter.py           # 输出格式化接口
│   ├── text_formatter.py      # 文本格式化
│   └── json_formatter.py      # JSON格式化
└── commands/
    ├── registry.py            # 命令注册表
    └── executor.py            # 命令执行器
```

**ProgressReporter** 接口用于统一进度报告：

```python
class ProgressReporter:
    """进度报告接口"""
    
    def start(self, total, description=None):
        """开始进度报告"""
        raise NotImplementedError
    
    def update(self, current, description=None):
        """更新进度"""
        raise NotImplementedError
    
    def finish(self, description=None):
        """完成进度报告"""
        raise NotImplementedError
```

### 异步处理框架

异步处理框架将使用 `asyncio` 实现，主要包含以下组件：

```
async/
├── task_manager.py            # 任务管理器
├── task.py                    # 任务基类
├── queue.py                   # 任务队列
└── workers/                   # 工作线程
    ├── worker.py              # 工作线程基类
    └── pool.py                # 工作线程池
```

**TaskManager** 负责管理异步任务：

```python
class TaskManager:
    """任务管理器，负责管理异步任务"""
    
    def __init__(self, max_workers=None):
        self.queue = TaskQueue()
        self.workers = WorkerPool(max_workers)
        self._running = False
    
    async def start(self):
        """启动任务管理器"""
        if self._running:
            return
            
        self._running = True
        await self.workers.start(self.queue)
    
    async def stop(self):
        """停止任务管理器"""
        if not self._running:
            return
            
        self._running = False
        await self.workers.stop()
    
    async def submit(self, task, priority=0):
        """提交任务"""
        await self.queue.put(task, priority)
        return task
    
    async def wait_for(self, task):
        """等待任务完成"""
        return await task.wait()
```

### 实施建议

1. **从核心架构开始**：首先实现 `ServiceManager` 和核心服务接口，这是整个系统的基础。

2. **渐进式重构**：不要一次性重写所有代码，而是逐步替换现有功能，确保系统始终可用。

3. **保持向后兼容**：在重构过程中，保持 API 的向后兼容性，避免破坏现有功能。

4. **编写测试**：为每个新组件编写测试，确保重构不会引入新的问题。

5. **文档先行**：在实现之前，先编写详细的设计文档，明确每个组件的职责和接口。

6. **定期集成**：频繁合并代码，避免长时间的分支开发导致的集成问题。

7. **用户反馈**：在重构过程中，收集用户反馈，确保新架构满足实际需求。

