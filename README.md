# AgentRSS: 智能RSS阅读与问答系统

![AgentRSS Logo](docs/logo.png)

AgentRSS是一个集成了RSS订阅、检索增强生成(RAG)和智能Agent的系统，为用户提供个性化的信息获取、整理和问答服务。

## 🌟 核心特性

- **智能RSS管理**：订阅、分类和更新各类信息源
- **强大的RAG引擎**：基于向量检索的智能问答系统
- **多模态Agent**：支持文本、图像、音频等多种输入形式
- **工具集成**：内置丰富工具集，支持扩展自定义工具
- **记忆系统**：短期和长期记忆管理，提供个性化体验
- **多模型支持**：兼容各类LLM，从轻量本地模型到云端大模型
- **友好界面**：简洁直观的Web界面和命令行工具

## 🏗️ 系统架构

AgentRSS采用模块化、可扩展的架构设计，主要由以下核心组件构成：

```
agentRSS/
├── core/                  # 核心框架
│   ├── agent/             # Agent系统
│   │   ├── base.py        # Agent基类
│   │   ├── rss_agent.py   # RSS专用Agent
│   │   ├── rag_agent.py   # RAG专用Agent
│   │   └── multi_agent.py # 多Agent协作系统
│   ├── memory/            # 记忆系统
│   │   ├── working.py     # 工作记忆
│   │   ├── episodic.py    # 情景记忆
│   │   └── semantic.py    # 语义记忆
│   ├── tools/             # 工具系统
│   │   ├── registry.py    # 工具注册中心
│   │   ├── rss_tools.py   # RSS相关工具
│   │   ├── search_tools.py # 搜索工具
│   │   └── file_tools.py  # 文件处理工具
│   └── llm/               # 语言模型接口
│       ├── base.py        # 模型基类
│       ├── openai.py      # OpenAI模型
│       ├── local.py       # 本地模型
│       └── adapter.py     # 模型适配器
├── rss/                   # RSS模块
│   ├── parser.py          # RSS解析器
│   ├── manager.py         # RSS源管理
│   ├── fetcher.py         # RSS内容获取
│   └── storage.py         # RSS数据存储
├── rag/                   # RAG模块
│   ├── engine.py          # RAG引擎
│   ├── embedding/         # 向量化模块
│   │   ├── model.py       # 嵌入模型
│   │   └── processor.py   # 文本处理器
│   ├── retrieval/         # 检索模块
│   │   ├── vector_db.py   # 向量数据库
│   │   ├── hybrid.py      # 混合检索
│   │   └── reranker.py    # 重排序器
│   └── generation/        # 生成模块
│       ├── prompt.py      # 提示模板
│       └── generator.py   # 回答生成器
├── multimodal/            # 多模态处理
│   ├── text.py            # 文本处理
│   ├── image.py           # 图像处理
│   ├── audio.py           # 音频处理
│   └── fusion.py          # 模态融合
├── ui/                    # 用户界面
│   ├── web/               # Web界面
│   │   ├── app.py         # Web应用
│   │   ├── components/    # UI组件
│   │   └── static/        # 静态资源
│   └── cli/               # 命令行界面
│       ├── commands/      # 命令定义
│       └── formatters/    # 输出格式化
└── utils/                 # 通用工具
    ├── config.py          # 配置管理
    ├── logging.py         # 日志系统
    ├── async_utils.py     # 异步工具
    └── security.py        # 安全工具
```

## 🔄 数据流与工作流程

AgentRSS的工作流程如下：

1. **信息获取**：RSS模块定期从订阅源获取最新内容
2. **内容处理**：将RSS内容解析、清洗并存储
3. **知识索引**：RAG模块对内容进行向量化并建立索引
4. **用户交互**：用户通过UI提交查询或指令
5. **Agent处理**：Agent系统分析用户意图，规划执行步骤
6. **工具调用**：根据需要调用相关工具（搜索、RSS查询等）
7. **回答生成**：基于检索结果和工具输出生成回答
8. **记忆更新**：更新系统记忆，优化未来交互

## 🧠 Agent系统设计

AgentRSS采用多层Agent架构：

### 1. 核心Agent框架

- **BaseAgent**：所有Agent的基类，定义通用接口
- **ToolAgent**：专注于工具调用的Agent
- **MemoryAgent**：具备记忆管理能力的Agent
- **PlannerAgent**：负责任务规划和分解的Agent

### 2. 专用Agent

- **RSSAgent**：管理RSS源和内容的专用Agent
- **RAGAgent**：处理知识检索和问答的专用Agent
- **MultimodalAgent**：处理多模态输入的Agent

### 3. 协作机制

- **AgentRegistry**：Agent注册和发现机制
- **AgentRouter**：根据任务类型路由到合适的Agent
- **AgentOrchestrator**：协调多个Agent协作完成复杂任务

## 📊 记忆系统

AgentRSS的记忆系统分为三层：

1. **工作记忆(WorkingMemory)**：
   - 存储当前会话状态和上下文
   - 短期存储，会话结束后清除

2. **情景记忆(EpisodicMemory)**：
   - 存储用户历史交互记录
   - 中期存储，支持检索历史对话

3. **语义记忆(SemanticMemory)**：
   - 存储从交互中提取的知识和概念
   - 长期存储，构建用户知识图谱

## 🛠️ 工具系统

AgentRSS的工具系统采用插件式设计：

1. **核心工具**：
   - **RSSTools**：RSS源管理、内容获取工具
   - **RAGTools**：知识检索和问答工具
   - **FileTools**：文件操作和处理工具

2. **扩展工具**：
   - **WebTools**：网页访问和内容提取工具
   - **APITools**：外部API调用工具
   - **AnalysisTools**：数据分析和可视化工具

3. **工具注册机制**：
   - 统一的工具接口
   - 动态工具注册和发现
   - 工具权限和安全管理

## 🔍 RAG引擎

AgentRSS的RAG引擎采用先进的检索增强生成技术：

1. **文本处理**：
   - 智能分块策略
   - 多级别文本表示

2. **检索策略**：
   - 混合检索（向量检索+关键词检索）
   - 多阶段检索和重排序
   - 上下文感知检索

3. **生成优化**：
   - 动态提示工程
   - 引用追踪和验证
   - 多文档综合

## 🌐 多模态支持

AgentRSS支持多种输入模态：

1. **文本处理**：理解和生成自然语言文本
2. **图像处理**：识别和分析图像内容
3. **音频处理**：语音识别和处理
4. **模态融合**：整合多模态信息，提供统一理解

## 🖥️ 用户界面

AgentRSS提供多种交互方式：

1. **Web界面**：
   - 响应式设计，支持移动设备
   - 实时更新和流式输出
   - 丰富的可视化组件

2. **命令行界面**：
   - 功能完整的CLI工具
   - 批处理支持
   - 脚本集成能力

3. **API接口**：
   - RESTful API
   - WebSocket实时通信
   - 第三方集成支持

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/agent-rss.git
cd agent-rss

# 安装依赖
pip install -r requirements.txt

# 下载必要的模型
python scripts/download_models.py
```

### 配置

创建配置文件：

```bash
cp config/config.example.yaml config/config.yaml
```

编辑配置文件，添加必要的API密钥和设置。

### 运行

启动Web界面：

```bash
python run_ui.py
```

使用命令行工具：

```bash
python -m agentrss.cli [命令] [参数]
```

## 📚 使用示例

### 1. RSS源管理

```bash
# 添加RSS源
python -m agentrss.cli rss add --url https://example.com/feed.xml --category tech

# 导入OPML文件
python -m agentrss.cli rss import --file feeds.opml

# 更新所有RSS源
python -m agentrss.cli rss update
```

### 2. 智能问答

```bash
# 基于RSS内容提问
python -m agentrss.cli ask "最近有哪些关于人工智能的重要新闻？"

# 指定时间范围和来源
python -m agentrss.cli ask "上周有哪些科技新闻？" --from "7d" --sources "tech"
```

### 3. Agent交互

```bash
# 启动交互式Agent会话
python -m agentrss.cli agent

# 执行复杂任务
python -m agentrss.cli agent "分析最近一周的科技新闻，找出最重要的三个趋势，并生成摘要报告"
```

## 🧩 扩展开发

AgentRSS设计为高度可扩展的系统，您可以：

1. **添加新工具**：实现Tool接口，注册到工具注册表
2. **自定义Agent**：继承BaseAgent，实现特定功能
3. **扩展UI**：添加新的UI组件或视图
4. **集成新模型**：实现ModelAdapter接口，支持新的LLM

详细开发指南请参考[开发文档](docs/development.md)。

## 📊 性能与资源需求

AgentRSS支持不同规模的部署：

1. **轻量级部署**：
   - 本地小型模型
   - SQLite存储
   - 最小内存需求：4GB

2. **标准部署**：
   - 中型本地模型或云API
   - PostgreSQL数据库
   - 推荐内存：8GB+

3. **高性能部署**：
   - 大型模型或多模型组合
   - 分布式向量数据库
   - 推荐内存：16GB+，GPU加速

## 🤝 贡献指南

我们欢迎各种形式的贡献！请查看[贡献指南](CONTRIBUTING.md)了解详情。

## 📜 许可证

本项目采用MIT许可证 - 详情请参阅[LICENSE](LICENSE)文件。

## 🙏 致谢

AgentRSS基于多个开源项目构建，特别感谢：
- [smolagents](https://github.com/huggingface/smolagents)：提供Agent框架支持
- [TinyRAG](https://github.com/example/tinyrag)：提供轻量级RAG引擎
- [OpenManus](https://github.com/mannaandpoem/OpenManus)：提供Agent设计灵感

