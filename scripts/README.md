# RSS-RAG 调试与改进工具

本目录包含用于调试和改进RSS-RAG系统的工具脚本。这些工具可以帮助您分析系统性能，找出问题所在，并实现改进方案。

## 工具列表

1. **debug_embedding.py**: 全面分析embedding和检索系统
2. **debug_retrieval.py**: 针对特定查询分析检索结果
3. **implement_query_enhancement.py**: 实现查询增强策略和其他改进

## 使用方法

### 1. 调试Embedding和检索系统

全面分析系统的各个方面，包括文本分块、embedding向量、检索结果和数据库内容：

```bash
# 运行所有分析
python scripts/debug_embedding.py

# 只分析文本分块
python scripts/debug_embedding.py --chunks

# 只分析Embedding向量
python scripts/debug_embedding.py --embeddings

# 只分析检索结果
python scripts/debug_embedding.py --retrieval

# 只分析数据库内容
python scripts/debug_embedding.py --database

# 只提供改进建议
python scripts/debug_embedding.py --recommendations

# 使用自定义查询测试检索
python scripts/debug_embedding.py --retrieval --query "人工智能最新进展"

# 限制分析的条目数量
python scripts/debug_embedding.py --limit 50
```

### 2. 调试特定查询的检索结果

针对特定查询分析检索过程和结果，显示详细的embedding数据和检索组件贡献：

```bash
# 使用默认查询
python scripts/debug_retrieval.py

# 使用自定义查询
python scripts/debug_retrieval.py "DeepSeek的技术特点"

# 显示更多检索结果
python scripts/debug_retrieval.py --top-k 10

# 显示详细信息（包括完整内容和向量详情）
python scripts/debug_retrieval.py -v
```

### 3. 实现查询增强策略和其他改进

自动修改代码，实现查询增强策略和其他改进：

```bash
# 只实现查询增强策略（默认行为）
python scripts/implement_query_enhancement.py

# 实现所有改进
python scripts/implement_query_enhancement.py --all

# 只实现分块大小过滤
python scripts/implement_query_enhancement.py --chunk-filter

# 只更新分块大小配置
python scripts/implement_query_enhancement.py --update-chunk-size

# 只更新检索权重
python scripts/implement_query_enhancement.py --update-weights

# 自定义参数
python scripts/implement_query_enhancement.py --all --min-length 150 --chunk-size 1000 --chunk-overlap 150 --bm25-weight 0.5 --vector-weight 0.5

# 不备份原始文件
python scripts/implement_query_enhancement.py --all --no-backup
```

## 改进建议

根据调试结果，以下是常见的改进建议：

### 1. 增加分块大小

如果检索结果的相关性不高，或者内容片段太小，可以增加分块大小：

```bash
python scripts/implement_query_enhancement.py --update-chunk-size --chunk-size 800 --chunk-overlap 100
```

### 2. 过滤过短的内容片段

如果存在很多短小的内容片段，可以添加最小长度过滤：

```bash
python scripts/implement_query_enhancement.py --chunk-filter --min-length 100
```

### 3. 实现查询增强策略

通过先生成初步回答，再基于初步回答扩展查询，提高检索相关性：

```bash
python scripts/implement_query_enhancement.py --query-enhancement
```

### 4. 调整检索权重

根据检索结果的质量，调整BM25和向量检索的权重：

```bash
# 增加BM25权重，提高多样性
python scripts/implement_query_enhancement.py --update-weights --bm25-weight 0.5 --vector-weight 0.5

# 增加向量检索权重，提高语义相关性
python scripts/implement_query_enhancement.py --update-weights --bm25-weight 0.2 --vector-weight 0.8
```

## 注意事项

1. 这些工具会修改源代码文件，默认会创建备份（.bak文件）
2. 修改后需要重新启动系统才能生效
3. 如果修改导致系统出现问题，可以使用备份文件恢复
4. 建议在测试环境中使用这些工具，而不是直接在生产环境中使用 