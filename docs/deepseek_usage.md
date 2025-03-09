# DeepSeek模型使用指南

本文档介绍如何在RSS-RAG系统中使用DeepSeek模型。

## 简介

DeepSeek是一家领先的AI公司，提供高性能的大语言模型API服务。RSS-RAG系统现已支持DeepSeek的模型，包括：

- **DeepSeek-V3** (`deepseek-chat`): 通用大语言模型，适合各种对话场景
- **DeepSeek-R1** (`deepseek-reasoner`): 推理增强型大语言模型，适合需要复杂推理的场景

## 准备工作

1. 获取DeepSeek API密钥
   - 访问 [DeepSeek官网](https://www.deepseek.com/) 注册账号
   - 在控制台创建API密钥

2. 安装依赖
   ```bash
   pip install requests
   ```

## 使用方法

### 1. 通过命令行使用

```bash
# 使用DeepSeek-V3模型
python scripts/cli.py ask "人工智能的最新进展是什么？" --llm-type deepseek --deepseek-api-key YOUR_API_KEY

# 使用DeepSeek-R1模型
python scripts/cli.py ask "分析量子计算对密码学的影响" --llm-type deepseek --deepseek-model deepseek-reasoner --deepseek-api-key YOUR_API_KEY

# 使用环境变量设置API密钥
export DEEPSEEK_API_KEY=YOUR_API_KEY
python scripts/cli.py ask "解释大语言模型的工作原理" --llm-type deepseek
```

### 2. 通过配置文件使用

创建配置文件 `config/my_deepseek_config.json`:

```json
{
    "base_dir": "data/rag_db",
    "device": "cuda",
    
    "llm_type": "deepseek",
    "deepseek_api_key": "YOUR_API_KEY",
    "deepseek_model": "deepseek-chat",
    "deepseek_base_url": "https://api.deepseek.com",
    
    "embedding_model_id": "models/bge-base-zh-v1.5",
    "reranker_model_id": "models/bge-reranker-base",
    
    "system_prompt": "你是一个有用的AI助手，擅长回答关于科技和人工智能的问题。",
    
    "chunk_size": 800,
    "chunk_overlap": 100,
    "use_model_for_splitting": false,
    
    "top_k": 5,
    "use_reranker": true,
    "use_query_enhancement": true
}
```

然后使用配置文件:

```bash
python scripts/cli.py ask "解释量子计算的基本原理" --config config/my_deepseek_config.json
```

### 3. 在代码中使用

```python
from src.rag.config import RAGConfig
from src.rag.rss_rag import RSSRAG

# 创建配置
config = RAGConfig(
    llm_type="deepseek",
    deepseek_api_key="YOUR_API_KEY",
    deepseek_model="deepseek-chat",
    system_prompt="你是一个有用的AI助手。"
)

# 初始化RAG系统
rag = RSSRAG(config)

# 加载状态
rag.load_state()

# 使用DeepSeek模型回答问题
answer = rag.answer("人工智能的未来发展趋势是什么？")
print(answer)
```

## 模型参数

DeepSeek模型支持以下参数:

- `model`: 模型名称，可选值:
  - `deepseek-chat`: DeepSeek-V3模型，通用对话模型
  - `deepseek-reasoner`: DeepSeek-R1模型，推理增强型模型
- `temperature`: 温度参数，控制生成的随机性，默认0.7
- `top_p`: 控制生成文本的多样性，默认0.9
- `max_tokens`: 最大生成token数，默认2048

## 注意事项

1. API密钥安全
   - 不要在代码中硬编码API密钥
   - 优先使用环境变量或配置文件存储API密钥
   - 不要将包含API密钥的配置文件提交到版本控制系统

2. 流量控制
   - DeepSeek API可能有调用频率限制
   - 在高频率调用场景下，考虑添加重试和延迟机制

3. 错误处理
   - 代码中已实现基本的错误处理
   - 在生产环境中，可能需要更完善的错误处理和重试机制

## 故障排除

1. API调用失败
   - 检查API密钥是否正确
   - 检查网络连接
   - 查看日志中的详细错误信息

2. 响应格式异常
   - 可能是API版本变更导致
   - 检查最新的DeepSeek API文档

## 参考资料

- [DeepSeek API文档](https://api-docs.deepseek.com/zh-cn/)
- [DeepSeek模型介绍](https://www.deepseek.com/products) 