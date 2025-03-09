#!/usr/bin/env python
"""
RSS-RAG Embedding和检索调试工具
用于分析embedding质量和检索结果，并提供改进建议
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
import json
import numpy as np
from pprint import pprint
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rss.storage import RSSStorage
from src.rag.config import RAGConfig
from src.rag.rss_rag import RSSRAG
from src.rag.utils.text_splitter import TextSplitter

def analyze_chunks(args):
    """分析文本分块情况"""
    print("\n===== 文本分块分析 =====")
    
    # 初始化存储
    storage = RSSStorage(args.db_path)
    
    # 获取条目
    entries = storage.get_entries(limit=args.limit, offset=0)
    
    # 初始化文本分割器
    text_splitter = TextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # 分析分块情况
    chunk_lengths = []
    total_chunks = 0
    empty_chunks = 0
    short_chunks = 0  # 少于50个字符的块
    
    for entry in tqdm(entries, desc="分析文本分块"):
        # 合并标题和内容
        text = f"{entry.title}\n\n{entry.content}" if entry.content else entry.title
        
        # 分块
        chunks = text_splitter.split_text(text)
        
        # 统计
        total_chunks += len(chunks)
        for chunk in chunks:
            chunk_len = len(chunk)
            chunk_lengths.append(chunk_len)
            if chunk_len == 0:
                empty_chunks += 1
            elif chunk_len < 50:
                short_chunks += 1
    
    # 输出统计结果
    if chunk_lengths:
        print(f"总条目数: {len(entries)}")
        print(f"总分块数: {total_chunks}")
        print(f"平均每条目分块数: {total_chunks / len(entries):.2f}")
        print(f"空块数量: {empty_chunks}")
        print(f"短块数量 (<50字符): {short_chunks}")
        print(f"分块长度统计:")
        print(f"  - 最小长度: {min(chunk_lengths)}")
        print(f"  - 最大长度: {max(chunk_lengths)}")
        print(f"  - 平均长度: {sum(chunk_lengths) / len(chunk_lengths):.2f}")
        print(f"  - 中位数长度: {sorted(chunk_lengths)[len(chunk_lengths)//2]}")
        
        # 长度分布
        print("\n分块长度分布:")
        ranges = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 1000), (1000, float('inf'))]
        for start, end in ranges:
            count = sum(1 for l in chunk_lengths if start <= l < end)
            percentage = count / len(chunk_lengths) * 100
            print(f"  - {start}-{end if end != float('inf') else '∞'}: {count} ({percentage:.2f}%)")
    else:
        print("没有找到任何分块")

def analyze_embeddings(args):
    """分析Embedding向量"""
    print("\n===== Embedding向量分析 =====")
    
    # 初始化RAG配置
    config = RAGConfig(
        vector_store_path=args.vector_store,
        embedding_model_path=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        llm_type="tiny",
        llm_model=args.llm_model
    )
    
    # 初始化RAG系统
    rag = RSSRAG(config)
    
    # 加载状态
    rag.load_state()
    
    # 获取向量数据
    if hasattr(rag.retriever, 'vector_store') and hasattr(rag.retriever.vector_store, 'embeddings'):
        embeddings = rag.retriever.vector_store.embeddings
        docs = rag.retriever.vector_store.docs
        
        print(f"向量数据库中的文档数: {len(docs)}")
        print(f"向量维度: {embeddings.shape[1] if embeddings is not None else 'N/A'}")
        
        # 分析向量质量
        if embeddings is not None and len(embeddings) > 1:
            # 计算向量之间的相似度
            from sklearn.metrics.pairwise import cosine_similarity
            
            # 随机抽样100个向量计算相似度（如果向量数量大于100）
            sample_size = min(100, len(embeddings))
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[indices]
            
            # 计算余弦相似度
            similarities = cosine_similarity(sample_embeddings)
            
            # 移除对角线（自身与自身的相似度）
            np.fill_diagonal(similarities, 0)
            
            # 统计相似度分布
            avg_similarity = np.mean(similarities)
            max_similarity = np.max(similarities)
            
            print(f"向量相似度分析 (随机抽样 {sample_size} 个向量):")
            print(f"  - 平均相似度: {avg_similarity:.4f}")
            print(f"  - 最大相似度: {max_similarity:.4f}")
            
            # 相似度分布
            ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
            similarities_flat = similarities.flatten()
            similarities_flat = similarities_flat[similarities_flat > 0]  # 移除对角线的0
            
            print("\n相似度分布:")
            for start, end in ranges:
                count = np.sum((similarities_flat >= start) & (similarities_flat < end))
                percentage = count / len(similarities_flat) * 100
                print(f"  - {start:.1f}-{end:.1f}: {count} ({percentage:.2f}%)")
            
            # 高相似度文档分析
            if max_similarity > 0.9:
                print("\n高相似度文档分析 (相似度 > 0.9):")
                high_sim_pairs = []
                for i in range(len(similarities)):
                    for j in range(i+1, len(similarities)):
                        if similarities[i, j] > 0.9:
                            doc_i = docs[indices[i]]
                            doc_j = docs[indices[j]]
                            high_sim_pairs.append((doc_i, doc_j, similarities[i, j]))
                
                for i, (doc_i, doc_j, sim) in enumerate(high_sim_pairs[:5]):  # 只显示前5对
                    print(f"\n高相似度对 {i+1} (相似度: {sim:.4f}):")
                    print(f"文档1: {doc_i.metadata.get('title', 'N/A')}")
                    print(f"文档2: {doc_j.metadata.get('title', 'N/A')}")
        else:
            print("没有足够的向量数据进行分析")
    else:
        print("无法访问向量数据")

def analyze_retrieval(args):
    """分析检索结果"""
    print("\n===== 检索结果分析 =====")
    
    # 初始化RAG配置
    config = RAGConfig(
        vector_store_path=args.vector_store,
        embedding_model_path=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        llm_type="tiny",
        llm_model=args.llm_model
    )
    
    # 初始化RAG系统
    rag = RSSRAG(config)
    
    # 加载状态
    rag.load_state()
    
    # 测试查询
    queries = args.queries if args.queries else [
        "人工智能最新进展",
        "Python编程技巧",
        "数据科学工具",
        "深度学习框架比较",
        "大语言模型应用"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        
        # 执行检索
        results = rag.search(query, top_k=args.top_k)
        
        # 分析结果
        if results:
            print(f"找到 {len(results)} 条相关结果")
            
            # 输出结果详情
            for i, result in enumerate(results):
                print(f"\n结果 {i+1}:")
                print(f"  - 标题: {result.metadata.get('title', 'N/A')}")
                print(f"  - 来源: {result.metadata.get('feed_title', 'N/A')}")
                print(f"  - 日期: {result.metadata.get('published_date', 'N/A')}")
                print(f"  - 相关度得分: {result.score:.4f}")
                print(f"  - 内容长度: {len(result.page_content)} 字符")
                print(f"  - 内容预览: {result.page_content[:100]}...")
            
            # 分析相关度分布
            scores = [r.score for r in results]
            print(f"\n相关度得分统计:")
            print(f"  - 最高得分: {max(scores):.4f}")
            print(f"  - 最低得分: {min(scores):.4f}")
            print(f"  - 平均得分: {sum(scores)/len(scores):.4f}")
            print(f"  - 得分差异 (最高-最低): {max(scores)-min(scores):.4f}")
            
            # 内容长度分析
            content_lengths = [len(r.page_content) for r in results]
            print(f"\n内容长度统计:")
            print(f"  - 最长: {max(content_lengths)} 字符")
            print(f"  - 最短: {min(content_lengths)} 字符")
            print(f"  - 平均: {sum(content_lengths)/len(content_lengths):.2f} 字符")
        else:
            print("没有找到相关结果")

def analyze_database(args):
    """分析数据库内容"""
    print("\n===== 数据库内容分析 =====")
    
    # 初始化存储
    storage = RSSStorage(args.db_path)
    
    # 获取源和条目统计
    feed_count = storage.get_feed_count()
    entry_count = storage.get_entry_count()
    
    print(f"RSS源数量: {feed_count}")
    print(f"条目总数: {entry_count}")
    
    # 获取所有源
    feeds = storage.get_feeds()
    
    if feeds:
        print("\nRSS源统计:")
        for feed in feeds:
            entry_count = storage.get_entry_count(feed_id=feed.id)
            print(f"  - {feed.title}: {entry_count} 条目")
        
        # 分析条目内容长度
        entries = storage.get_entries(limit=args.limit, offset=0)
        
        if entries:
            content_lengths = []
            title_lengths = []
            empty_content = 0
            
            for entry in entries:
                title_lengths.append(len(entry.title) if entry.title else 0)
                if entry.content:
                    content_lengths.append(len(entry.content))
                else:
                    empty_content += 1
            
            print(f"\n条目内容分析 (抽样 {len(entries)} 条):")
            print(f"  - 空内容条目数: {empty_content} ({empty_content/len(entries)*100:.2f}%)")
            
            if content_lengths:
                print(f"  - 内容长度统计:")
                print(f"    - 最短: {min(content_lengths)} 字符")
                print(f"    - 最长: {max(content_lengths)} 字符")
                print(f"    - 平均: {sum(content_lengths)/len(content_lengths):.2f} 字符")
                print(f"    - 中位数: {sorted(content_lengths)[len(content_lengths)//2]} 字符")
                
                # 长度分布
                print("\n  - 内容长度分布:")
                ranges = [(0, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10000), (10000, float('inf'))]
                for start, end in ranges:
                    count = sum(1 for l in content_lengths if start <= l < end)
                    percentage = count / len(content_lengths) * 100
                    print(f"    - {start}-{end if end != float('inf') else '∞'}: {count} ({percentage:.2f}%)")
            
            print(f"\n  - 标题长度统计:")
            print(f"    - 最短: {min(title_lengths)} 字符")
            print(f"    - 最长: {max(title_lengths)} 字符")
            print(f"    - 平均: {sum(title_lengths)/len(title_lengths):.2f} 字符")
    else:
        print("数据库中没有RSS源")

def provide_recommendations(args):
    """根据分析结果提供改进建议"""
    print("\n===== 改进建议 =====")
    
    # 初始化存储
    storage = RSSStorage(args.db_path)
    
    # 获取条目
    entries = storage.get_entries(limit=args.limit, offset=0)
    
    # 初始化文本分割器
    text_splitter = TextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # 分析分块情况
    chunk_lengths = []
    content_lengths = []
    empty_content = 0
    
    for entry in entries:
        # 统计内容长度
        if entry.content:
            content_lengths.append(len(entry.content))
        else:
            empty_content += 1
            
        # 合并标题和内容
        text = f"{entry.title}\n\n{entry.content}" if entry.content else entry.title
        
        # 分块
        chunks = text_splitter.split_text(text)
        
        # 统计
        for chunk in chunks:
            chunk_lengths.append(len(chunk))
    
    # 根据分析结果提供建议
    recommendations = []
    
    # 1. 分块大小建议
    if chunk_lengths:
        avg_chunk_len = sum(chunk_lengths) / len(chunk_lengths)
        if avg_chunk_len < 100:
            recommendations.append(f"当前平均分块长度较短 ({avg_chunk_len:.2f} 字符)，建议增加chunk_size参数 (当前: {args.chunk_size})，可以尝试设置为 {min(1000, args.chunk_size * 2)}。")
        elif avg_chunk_len > 400:
            recommendations.append(f"当前平均分块长度较长 ({avg_chunk_len:.2f} 字符)，可能会影响检索精度，建议减小chunk_size参数 (当前: {args.chunk_size})，可以尝试设置为 {max(200, args.chunk_size // 2)}。")
        else:
            recommendations.append(f"当前平均分块长度适中 ({avg_chunk_len:.2f} 字符)，chunk_size参数 (当前: {args.chunk_size}) 设置合理。")
    
    # 2. 分块重叠建议
    if args.chunk_overlap < args.chunk_size * 0.1:
        recommendations.append(f"当前分块重叠较小 ({args.chunk_overlap} 字符)，可能导致语义割裂，建议增加chunk_overlap参数，可以尝试设置为chunk_size的10%-20% ({int(args.chunk_size * 0.1)}-{int(args.chunk_size * 0.2)})。")
    elif args.chunk_overlap > args.chunk_size * 0.5:
        recommendations.append(f"当前分块重叠较大 ({args.chunk_overlap} 字符)，会导致存储冗余，建议减小chunk_overlap参数，可以尝试设置为chunk_size的10%-20% ({int(args.chunk_size * 0.1)}-{int(args.chunk_size * 0.2)})。")
    
    # 3. 内容质量建议
    if empty_content / len(entries) > 0.3:
        recommendations.append(f"数据中存在较多空内容条目 ({empty_content/len(entries)*100:.2f}%)，建议检查RSS解析过程，确保正确提取内容。")
    
    if content_lengths:
        avg_content_len = sum(content_lengths) / len(content_lengths)
        if avg_content_len < 200:
            recommendations.append(f"条目平均内容长度较短 ({avg_content_len:.2f} 字符)，可能导致检索信息不足，建议添加更多高质量的RSS源或调整内容提取策略。")
    
    # 4. 检索策略建议
    recommendations.append("可以尝试调整混合检索的权重，当前默认BM25权重为0.3，向量检索权重为0.7，可以根据实际效果进行调整。")
    recommendations.append("考虑实现查询增强策略，先用LLM生成初步回答，再基于初步回答扩展查询，提高检索相关性。")
    
    # 5. 模型建议
    recommendations.append(f"当前使用的Embedding模型为 {args.embedding_model}，如果检索效果不佳，可以尝试更换为其他高性能中文Embedding模型，如BAAI/bge-large-zh-v1.5。")
    
    # 输出建议
    for i, rec in enumerate(recommendations):
        print(f"{i+1}. {rec}")
    
    # 输出改进代码示例
    print("\n===== 改进代码示例 =====")
    
    # 1. 调整分块参数示例
    print("\n1. 调整文本分块参数:")
    print("""```python
# 在src/rag/config.py中修改默认参数
class RAGConfig:
    def __init__(self,
                 # ...其他参数...
                 chunk_size: int = 800,  # 增加分块大小
                 chunk_overlap: int = 100,  # 设置适当的重叠
                 # ...其他参数...
                ):
        # ...
```""")
    
    # 2. 实现查询增强示例
    print("\n2. 实现查询增强策略:")
    print("""```python
# 在src/rag/rss_rag.py的answer方法中添加查询增强
def answer(self, query: str, ...):
    # 先用LLM生成初步回答
    initial_answer = self.llm.generate(query)
    
    # 增强查询
    enhanced_query = query + " " + initial_answer
    
    # 使用增强查询检索
    search_results = self.search(enhanced_query, ...)
    # ...
```""")
    
    # 3. 调整混合检索权重示例
    print("\n3. 调整混合检索权重:")
    print("""```python
# 在src/rag/retrieval/hybrid_retriever.py中修改默认权重
def search(self,
          query: str,
          top_k: int = 3,
          metadata_filters: Optional[Dict] = None,
          weights: Optional[Dict[str, float]] = None):
    if weights is None:
        weights = {'bm25': 0.5, 'vector': 0.5}  # 调整权重
    # ...
```""")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RSS-RAG Embedding和检索调试工具')
    parser.add_argument('--db', dest='db_path', default='data/rss.db',
                        help='数据库文件路径 (默认: data/rss.db)')
    parser.add_argument('--vector-store', dest='vector_store', default='data/vector_store',
                        help='向量存储路径 (默认: data/vector_store)')
    parser.add_argument('--embedding-model', dest='embedding_model', default='models/bge-base-zh-v1.5',
                        help='Embedding模型路径 (默认: models/bge-base-zh-v1.5)')
    parser.add_argument('--llm-model', dest='llm_model', default='models/Qwen-1_8B-Chat',
                        help='LLM模型路径 (默认: models/Qwen-1_8B-Chat)')
    parser.add_argument('--chunk-size', dest='chunk_size', type=int, default=500,
                        help='分块大小 (默认: 500)')
    parser.add_argument('--chunk-overlap', dest='chunk_overlap', type=int, default=50,
                        help='分块重叠 (默认: 50)')
    parser.add_argument('--top-k', dest='top_k', type=int, default=5,
                        help='检索结果数量 (默认: 5)')
    parser.add_argument('--limit', dest='limit', type=int, default=100,
                        help='分析的条目数量限制 (默认: 100)')
    parser.add_argument('--query', dest='queries', action='append',
                        help='测试查询 (可多次指定)')
    parser.add_argument('--all', action='store_true',
                        help='运行所有分析')
    parser.add_argument('--chunks', action='store_true',
                        help='分析文本分块')
    parser.add_argument('--embeddings', action='store_true',
                        help='分析Embedding向量')
    parser.add_argument('--retrieval', action='store_true',
                        help='分析检索结果')
    parser.add_argument('--database', action='store_true',
                        help='分析数据库内容')
    parser.add_argument('--recommendations', action='store_true',
                        help='提供改进建议')
    
    args = parser.parse_args()
    
    # 如果没有指定具体分析，则运行所有分析
    if not (args.chunks or args.embeddings or args.retrieval or args.database or args.recommendations or args.all):
        args.all = True
    
    print("RSS-RAG Embedding和检索调试工具")
    print(f"数据库路径: {args.db_path}")
    print(f"向量存储路径: {args.vector_store}")
    print(f"Embedding模型: {args.embedding_model}")
    print(f"分块参数: 大小={args.chunk_size}, 重叠={args.chunk_overlap}")
    
    # 运行选定的分析
    if args.all or args.database:
        analyze_database(args)
    
    if args.all or args.chunks:
        analyze_chunks(args)
    
    if args.all or args.embeddings:
        analyze_embeddings(args)
    
    if args.all or args.retrieval:
        analyze_retrieval(args)
    
    if args.all or args.recommendations:
        provide_recommendations(args)

if __name__ == '__main__':
    main() 