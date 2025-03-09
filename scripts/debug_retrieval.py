#!/usr/bin/env python
"""
RSS-RAG 检索调试工具
用于分析检索结果和embedding数据，帮助改进RAG系统
"""
import os
import sys
import argparse
import numpy as np
from pprint import pprint
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rss.storage import RSSStorage
from src.rag.config import RAGConfig
from src.rag.rss_rag import RSSRAG
from src.rag.utils.text_splitter import TextSplitter

def debug_retrieval(query, config_path=None, top_k=5, verbose=False):
    """调试检索过程"""
    print(f"\n===== 检索调试: '{query}' =====")
    
    # 加载配置
    if config_path and os.path.exists(config_path):
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = RAGConfig(**config_dict)
    else:
        config = RAGConfig(
            vector_store_path="data/vector_store",
            embedding_model_path="models/bge-base-zh-v1.5",
            chunk_size=800,  # 使用更大的分块大小
            chunk_overlap=100,  # 增加重叠
            llm_type="tiny",
            llm_model="models/tiny_llm_sft_92m"  # 使用实际的模型路径
        )
    
    # 初始化RAG系统
    print("初始化RAG系统...")
    rag = RSSRAG(config)
    
    # 加载状态
    print("加载向量存储...")
    rag.load_state()
    
    # 获取查询的embedding
    print("生成查询embedding...")
    query_embedding = rag.retriever.embedding_model.encode(query)
    
    print(f"查询embedding维度: {query_embedding.shape}")
    print(f"查询embedding前10个值: {query_embedding[:10]}")
    print(f"查询embedding L2范数: {np.linalg.norm(query_embedding):.4f}")
    
    # 执行检索
    print("\n执行检索...")
    results = rag.search(query, top_k=top_k)
    
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
            
            # 获取结果的embedding
            if hasattr(rag.retriever, 'vector_store') and hasattr(rag.retriever.vector_store, 'embeddings'):
                # 找到对应的embedding
                doc_id = None
                for j, doc in enumerate(rag.retriever.vector_store.docs):
                    if doc.page_content == result.page_content:
                        doc_id = j
                        break
                
                if doc_id is not None:
                    doc_embedding = rag.retriever.vector_store.embeddings[doc_id]
                    
                    # 计算与查询的余弦相似度
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                    
                    print(f"  - 向量相似度: {similarity:.4f}")
                    print(f"  - 向量L2范数: {np.linalg.norm(doc_embedding):.4f}")
                    
                    if verbose:
                        print(f"  - 向量前10个值: {doc_embedding[:10]}")
            
            # 显示内容预览
            preview_len = min(200, len(result.page_content))
            print(f"  - 内容预览: {result.page_content[:preview_len]}...")
            
            if verbose:
                print(f"  - 完整内容: {result.page_content}")
    else:
        print("没有找到相关结果")
    
    # 分析BM25和向量检索的贡献
    print("\n分析检索组件贡献:")
    
    # 执行纯BM25检索
    bm25_results = rag.retriever._bm25_search(query, top_k=top_k)
    
    # 执行纯向量检索
    vector_results = rag.retriever._vector_search(query, top_k=top_k)
    
    print("\nBM25检索结果:")
    for i, result in enumerate(bm25_results[:3]):  # 只显示前3个
        print(f"  {i+1}. {result.metadata.get('title', 'N/A')} (得分: {result.score:.4f})")
    
    print("\n向量检索结果:")
    for i, result in enumerate(vector_results[:3]):  # 只显示前3个
        print(f"  {i+1}. {result.metadata.get('title', 'N/A')} (得分: {result.score:.4f})")
    
    # 分析混合检索结果与单独检索结果的重叠
    hybrid_titles = [r.metadata.get('title', '') for r in results]
    bm25_titles = [r.metadata.get('title', '') for r in bm25_results]
    vector_titles = [r.metadata.get('title', '') for r in vector_results]
    
    bm25_overlap = len(set(hybrid_titles) & set(bm25_titles))
    vector_overlap = len(set(hybrid_titles) & set(vector_titles))
    
    print(f"\n混合检索与BM25重叠: {bm25_overlap}/{len(results)} ({bm25_overlap/len(results)*100:.1f}%)")
    print(f"混合检索与向量检索重叠: {vector_overlap}/{len(results)} ({vector_overlap/len(results)*100:.1f}%)")
    
    # 提供改进建议
    print("\n===== 改进建议 =====")
    
    # 根据重叠情况提供建议
    if bm25_overlap > vector_overlap:
        print("1. 当前检索结果更偏向BM25，可以考虑增加向量检索的权重。")
        print("   建议修改: weights = {'bm25': 0.3, 'vector': 0.7} -> weights = {'bm25': 0.2, 'vector': 0.8}")
    elif vector_overlap > bm25_overlap:
        print("1. 当前检索结果更偏向向量检索，可以考虑增加BM25的权重以提高多样性。")
        print("   建议修改: weights = {'bm25': 0.3, 'vector': 0.7} -> weights = {'bm25': 0.4, 'vector': 0.6}")
    
    # 根据结果相关度提供建议
    scores = [r.score for r in results]
    if max(scores) < 0.5:
        print("2. 检索结果的最高相关度较低，建议:")
        print("   - 增加分块大小，当前为", config.chunk_size, "，建议增加到", min(1000, config.chunk_size * 2))
        print("   - 考虑使用更高性能的Embedding模型，如BAAI/bge-large-zh-v1.5")
        print("   - 实现查询增强策略，使用LLM扩展原始查询")
    
    # 根据内容长度提供建议
    content_lengths = [len(r.page_content) for r in results]
    if min(content_lengths) < 100:
        print("3. 检索结果中存在较短的内容片段，建议:")
        print("   - 增加最小分块长度限制，过滤掉过短的片段")
        print("   - 在src/rag/utils/text_splitter.py中添加过滤逻辑:")
        print("     ```python")
        print("     def split_text(self, text):")
        print("         chunks = # 现有分块逻辑")
        print("         # 过滤掉过短的片段")
        print("         chunks = [chunk for chunk in chunks if len(chunk) >= 100]")
        print("         return chunks")
        print("     ```")
    
    # 查询增强建议
    print("4. 实现查询增强策略:")
    print("   ```python")
    print("   # 在src/rag/rss_rag.py的answer方法中添加")
    print("   def answer(self, query: str, ...):")
    print("       # 先用LLM生成初步回答")
    print("       initial_answer = self.llm.generate(query)")
    print("       ")
    print("       # 增强查询")
    print("       enhanced_query = query + ' ' + initial_answer")
    print("       ")
    print("       # 使用增强查询检索")
    print("       search_results = self.search(enhanced_query, ...)")
    print("   ```")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RSS-RAG 检索调试工具')
    parser.add_argument('query', nargs='?', default="人工智能最新进展",
                        help='要测试的查询 (默认: "人工智能最新进展")')
    parser.add_argument('--config', dest='config_path', default=None,
                        help='配置文件路径 (JSON格式)')
    parser.add_argument('--top-k', dest='top_k', type=int, default=5,
                        help='检索结果数量 (默认: 5)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='显示详细信息')
    
    args = parser.parse_args()
    
    debug_retrieval(args.query, args.config_path, args.top_k, args.verbose)

if __name__ == '__main__':
    main() 