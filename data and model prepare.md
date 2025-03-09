# 数据和模型准备说明

## 1. 模型下载

### 1.1 Embedding模型
- BGE模型：[BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5)
  ```bash
  # 下载模型到models目录
  git clone https://huggingface.co/BAAI/bge-base-zh-v1.5 models/bge-base-zh-v1.5
  ```

### 1.2 LLM模型（待实现）
- Qwen模型：[Qwen/Qwen-1_8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat)
  ```bash
  # 下载模型到models目录
  git clone https://huggingface.co/Qwen/Qwen-1_8B-Chat models/Qwen-1_8B-Chat
  ```

## 2. 数据存储
- 向量数据库文件存储在 `data/vector_store` 目录下
- RSS数据存储在 `data/rss.db` SQLite数据库中 