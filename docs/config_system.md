# RSS-RAG 配置系统

RSS-RAG 系统采用了一个灵活的分层配置系统，支持多种配置方式和热重载功能。本文档将详细介绍配置系统的使用方法和特性。

## 配置系统特性

1. **分层配置**：配置按照优先级从低到高分为四层
   - 默认配置：系统内置的基础配置
   - 应用配置：`config/app_config.json` 文件中的配置
   - 用户配置：`config/user_config.json` 文件中的配置
   - 环境变量：以 `RSS_RAG_` 为前缀的环境变量

2. **配置验证**：使用 Pydantic 进行配置验证，确保配置参数的有效性

3. **热重载配置**：支持在运行时自动检测配置文件变化并重新加载配置，无需重启应用

4. **配置变更通知**：支持配置变更时的回调函数，方便系统组件响应配置变化

## 配置文件

系统使用以下配置文件：

1. **应用配置文件**：`config/app_config.json`
   - 包含应用级别的配置，通常由应用开发者设置
   - 适用于整个应用的默认配置

2. **用户配置文件**：`config/user_config.json`
   - 包含用户级别的配置，通常由用户通过UI设置
   - 只需包含与默认配置不同的配置项
   - 优先级高于应用配置

## 环境变量配置

系统支持通过环境变量覆盖配置，环境变量需要以 `RSS_RAG_` 为前缀，后跟配置项名称的大写形式。例如：

```bash
# 设置设备为 CPU
export RSS_RAG_DEVICE=cpu

# 设置 OpenAI API 密钥
export RSS_RAG_OPENAI_API_KEY=your_api_key_here

# 设置检索结果数量
export RSS_RAG_TOP_K=5
```

可以创建一个 `.env` 文件，将环境变量配置放在其中，系统会自动加载。参考 `.env.example` 文件了解支持的环境变量。

## 配置项说明

以下是主要配置项的说明：

### 基础配置

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| base_dir | 字符串 | "data/rag_db" | 基础数据目录 |
| device | 字符串 | "cpu" | 设备类型，可选值: cpu, cuda |

### 模型配置

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| llm_type | 字符串 | "tiny" | LLM类型，可选值: tiny, openai, huggingface, deepseek |
| llm_model_id | 字符串 | "models/tiny_llm_sft_92m" | LLM模型ID |
| embedding_model_id | 字符串 | "models/bge-base-zh-v1.5" | Embedding模型ID |
| reranker_model_id | 字符串 | "models/bge-reranker-base" | 重排序模型ID |
| system_prompt | 字符串 | "你是一个有用的AI助手。" | 系统提示词 |

### API配置

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| openai_api_key | 字符串 | "" | OpenAI API密钥 |
| openai_model | 字符串 | "gpt-3.5-turbo" | OpenAI模型名称 |
| deepseek_api_key | 字符串 | "" | DeepSeek API密钥 |
| deepseek_model | 字符串 | "deepseek-chat" | DeepSeek模型名称 |
| deepseek_base_url | 字符串 | "https://api.deepseek.com" | DeepSeek API基础URL |

### 文本分割配置

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| chunk_size | 整数 | 500 | 文本分块大小 |
| chunk_overlap | 整数 | 50 | 文本分块重叠大小 |
| use_model_for_splitting | 布尔值 | false | 是否使用模型进行分句 |
| sentence_splitter_model | 字符串 | "damo/nlp_bert_document-segmentation_chinese-base" | 分句模型 |

### 检索配置

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| top_k | 整数 | 3 | 检索结果数量 |
| search_weights | 对象 | null | 混合检索权重 |
| use_reranker | 布尔值 | true | 是否使用重排序 |
| use_query_enhancement | 布尔值 | false | 是否使用查询增强 |

### 缓存配置

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| use_cache | 布尔值 | true | 是否使用缓存 |
| cache_dir | 字符串 | "data/cache" | 缓存目录 |
| cache_size | 整数 | 1000 | 缓存大小 |
| cache_ttl | 整数 | 3600 | 缓存过期时间（秒） |

### RSS特定配置

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| max_history_days | 整数 | 30 | RSS条目最大保留天数 |
| update_interval | 整数 | 3600 | RSS更新间隔（秒） |

### 并行处理配置

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| use_parallel_processing | 布尔值 | true | 是否使用并行处理 |
| max_workers | 整数 | null | 最大工作线程数，null表示使用CPU核心数 |

## 在代码中使用配置系统

### 基本使用

```python
from src.rag.core.config_manager import ConfigManager
from src.rag.rss_rag import RSSRAG

# 创建配置管理器
config_manager = ConfigManager(
    config_path="config/app_config.json",
    user_config_path="config/user_config.json",
    auto_reload=True
)

# 创建RSS-RAG系统
rag_system = RSSRAG(config_manager=config_manager)

# 获取当前配置
current_config = config_manager.get_config()
print(f"当前设备: {current_config['device']}")

# 更新配置
config_manager.update_config({
    "top_k": 5,
    "use_reranker": True
})
```

### 注册配置变更回调

```python
def on_config_changed(old_config, new_config):
    print("配置已变更:")
    for key in new_config:
        if key in old_config and old_config[key] != new_config[key]:
            print(f"  {key}: {old_config[key]} -> {new_config[key]}")

# 注册回调
config_manager.register_change_callback(on_config_changed)
```

### 重置配置

```python
# 重置所有配置到默认值
config_manager.reset_to_default()

# 只重置特定配置项
config_manager.reset_to_default(keys=["chunk_size", "chunk_overlap"])
```

## 通过UI管理配置

系统提供了友好的Web界面用于管理配置。在Gradio界面中，选择"系统配置"标签页，可以查看和修改所有配置项。

修改配置后，点击"保存配置"按钮应用更改。系统会自动验证配置的有效性，并将有效的配置保存到用户配置文件中。

如果需要恢复默认配置，可以点击"重置为默认配置"按钮。 