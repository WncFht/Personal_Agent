import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from loguru import logger
from functools import lru_cache
import os

from ..embedding.embedding_model import EmbeddingModel
from ..utils.text_splitter import TextSplitter
try:
    from .reranker import BGEM3Reranker
    RERANKER_AVAILABLE = True
except ImportError:
    logger.warning("未安装sentence-transformers，无法使用重排序功能")
    RERANKER_AVAILABLE = False

class Document:
    """文档类，用于存储文本及其元数据"""
    def __init__(self, 
                 text: str,
                 metadata: Optional[Dict] = None,
                 embedding: Optional[np.ndarray] = None):
        self.text = text
        self.metadata = metadata or {}
        self.embedding = embedding

class EnhancedCache:
    """增强的缓存类，支持LRU和TTL"""
    
    def __init__(self, cache_size: int = 1000, ttl: int = 3600):
        """
        初始化
        
        Args:
            cache_size: 缓存大小
            ttl: 缓存过期时间（秒）
        """
        self.cache = {}
        self.cache_size = cache_size
        self.ttl = ttl
        self.access_count = {}
        self.last_access_time = {}
        
    def get(self, key: str) -> Optional[np.ndarray]:
        """获取缓存"""
        if key in self.cache:
            current_time = datetime.now().timestamp()
            # 检查是否过期
            if self.ttl > 0 and current_time - self.last_access_time[key] > self.ttl:
                # 过期，删除缓存
                del self.cache[key]
                del self.access_count[key]
                del self.last_access_time[key]
                return None
                
            # 更新访问计数和时间
            self.access_count[key] += 1
            self.last_access_time[key] = current_time
            return self.cache[key]
        return None
        
    def set(self, key: str, value: np.ndarray):
        """设置缓存"""
        # 如果缓存已满，清理最少使用的项
        if len(self.cache) >= self.cache_size:
            self._evict_least_used()
            
        # 添加新项
        self.cache[key] = value
        self.access_count[key] = 1
        self.last_access_time[key] = datetime.now().timestamp()
        
    def _evict_least_used(self):
        """清理最少使用的缓存项"""
        if not self.cache:
            return
            
        # 找出访问次数最少的项
        min_key = min(self.access_count.items(), key=lambda x: x[1])[0]
        
        # 删除
        del self.cache[min_key]
        del self.access_count[min_key]
        del self.last_access_time[min_key]

class HybridRetriever:
    """混合检索器，结合BM25和向量检索"""
    
    def __init__(self,
                 embedding_model: EmbeddingModel,
                 base_dir: str = "data/rag_db",
                 cache_size: int = 1000,
                 cache_ttl: int = 3600,
                 use_reranker: bool = False,
                 reranker_model_path: str = "models/bge-reranker-base",
                 device: str = "cuda"):
        self.embedding_model = embedding_model
        self.base_dir = base_dir
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.device = device
        
        # 初始化文档存储
        self.documents: List[Document] = []
        self.document_embeddings: List[np.ndarray] = []
        
        # 初始化BM25参数
        self.initialize_bm25()
        
        # 初始化缓存
        self.embedding_cache = EnhancedCache(cache_size=cache_size, ttl=cache_ttl)
        
        # 初始化重排序器
        self.use_reranker = use_reranker and RERANKER_AVAILABLE
        if self.use_reranker:
            try:
                self.reranker = BGEM3Reranker(model_path=reranker_model_path, device=device)
                logger.info("重排序器初始化成功")
            except Exception as e:
                logger.error(f"重排序器初始化失败: {e}")
                self.use_reranker = False
        
    def initialize_bm25(self):
        """初始化BM25参数"""
        self.k1 = 1.5
        self.b = 0.75
        self.avgdl = 0
        self.doc_freqs = {}
        self.idf = {}
        self.doc_len = []
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的embedding向量（带缓存）"""
        # 先从缓存获取
        cached_embedding = self.embedding_cache.get(text)
        if cached_embedding is not None:
            return cached_embedding
            
        # 缓存未命中，计算embedding
        embedding = self.embedding_model.encode(text)
        
        # 存入缓存
        self.embedding_cache.set(text, embedding)
        
        return embedding
    
    def add_document(self, text: str, metadata: Optional[Dict] = None):
        """添加文档到检索器"""
        # 创建Document对象
        embedding = self._get_embedding(text)
        doc = Document(text=text, metadata=metadata, embedding=embedding)
        
        # 更新文档存储
        self.documents.append(doc)
        self.document_embeddings.append(embedding)
        
        # 更新BM25参数
        self._update_bm25_params(text)
        
    def _update_bm25_params(self, text: str):
        """更新BM25参数"""
        # 分词
        words = text.split()
        doc_len = len(words)
        self.doc_len.append(doc_len)
        
        # 更新文档频率
        for word in set(words):
            if word not in self.doc_freqs:
                self.doc_freqs[word] = 0
            self.doc_freqs[word] += 1
            
        # 更新平均文档长度
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        
        # 更新IDF
        N = len(self.documents)
        for word, freq in self.doc_freqs.items():
            self.idf[word] = np.log((N - freq + 0.5) / (freq + 0.5) + 1)
            
    def _bm25_score(self, query: str, doc_idx: int) -> float:
        """计算BM25得分"""
        score = 0.0
        doc = self.documents[doc_idx].text
        doc_len = self.doc_len[doc_idx]
        
        # 分词
        query_words = query.split()
        doc_words = doc.split()
        
        # 计算得分
        for word in query_words:
            if word not in self.idf:
                continue
                
            # 计算词频
            freq = doc_words.count(word)
            
            # BM25公式
            numerator = self.idf[word] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += numerator / denominator
            
        return score
        
    def _vector_similarity(self, query_embedding: np.ndarray, doc_idx: int) -> float:
        """计算向量相似度"""
        return np.dot(query_embedding, self.document_embeddings[doc_idx])
    
    def _hybrid_recall(self,
                      query: str,
                      top_k: int = 10,
                      metadata_filters: Optional[Dict] = None,
                      weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float, Dict]]:
        """第一阶段：混合召回"""
        if weights is None:
            weights = {'bm25': 0.3, 'vector': 0.7}
            
        logger.info(f"执行混合召回:")
        logger.info(f"- 查询文本: {query}")
        logger.info(f"- BM25权重: {weights['bm25']}")
        logger.info(f"- 向量检索权重: {weights['vector']}")
            
        # 获取查询向量
        query_embedding = self._get_embedding(query)
        logger.info("已生成查询文本的embedding向量")
        
        # 计算所有文档的得分
        scores = []
        logger.info(f"开始计算文档得分 (共 {len(self.documents)} 个文档)")
        
        for i in range(len(self.documents)):
            # 计算BM25得分
            bm25_score = self._bm25_score(query, i)
            
            # 计算向量相似度
            vector_score = self._vector_similarity(query_embedding, i)
            
            # 合并得分
            final_score = weights['bm25'] * bm25_score + weights['vector'] * vector_score
            
            # 添加到结果列表
            doc = self.documents[i]
            
            # 应用元数据过滤
            if not self._apply_filters(doc, metadata_filters):
                continue
                    
            logger.debug(f"文档 {i}:")
            logger.debug(f"  - BM25得分: {bm25_score:.4f}")
            logger.debug(f"  - 向量相似度: {vector_score:.4f}")
            logger.debug(f"  - 最终得分: {final_score:.4f}")
            
            scores.append((doc.text, final_score, doc.metadata))
            
        # 排序并返回top_k结果
        scores.sort(key=lambda x: x[1], reverse=True)
        results = scores[:top_k]
        
        logger.info(f"混合召回完成，返回前 {top_k} 个结果")
        return results
    
    def _apply_filters(self, doc: Document, metadata_filters: Optional[Dict]) -> bool:
        """应用元数据过滤"""
        if not metadata_filters:
            return True
            
        # 基础过滤
        for key, value in metadata_filters.items():
            if callable(value):
                if key not in doc.metadata or not value(doc.metadata[key]):
                    return False
            elif key not in doc.metadata or doc.metadata[key] != value:
                return False
                
        return True
    
    def _rerank(self, query: str, documents: List[Tuple[str, float, Dict]], top_k: int = 3) -> List[Tuple[str, float, Dict]]:
        """第二阶段：重排序"""
        if not self.use_reranker:
            return documents[:top_k]
            
        try:
            logger.info(f"执行重排序，共 {len(documents)} 个文档")
            reranked_results = self.reranker.rerank(query, documents, top_k)
            logger.info(f"重排序完成，返回前 {top_k} 个结果")
            return reranked_results
        except Exception as e:
            logger.error(f"重排序失败: {e}，返回原始结果")
            return documents[:top_k]
        
    def search(self,
              query: str,
              top_k: int = 3,
              metadata_filters: Optional[Dict] = None,
              weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float, Dict]]:
        """多阶段检索实现"""
        logger.info(f"开始多阶段检索: {query}")
        
        # 第一阶段：混合召回
        recall_results = self._hybrid_recall(
            query=query, 
            top_k=top_k * 2,  # 召回阶段获取更多结果
            metadata_filters=metadata_filters,
            weights=weights
        )
        
        # 如果没有找到结果，直接返回
        if not recall_results:
            logger.info("未找到匹配结果")
            return []
            
        # 第二阶段：重排序
        if self.use_reranker:
            final_results = self._rerank(query, recall_results, top_k)
        else:
            final_results = recall_results[:top_k]
            
        logger.info(f"检索完成，返回前 {top_k} 个结果")
        return final_results
        
    def save(self, file_path: str):
        """保存检索器状态"""
        state = {
            'documents': [(doc.text, doc.metadata) for doc in self.documents],
            'embeddings': [emb.tolist() for emb in self.document_embeddings],
            'bm25_params': {
                'k1': self.k1,
                'b': self.b,
                'avgdl': self.avgdl,
                'doc_freqs': self.doc_freqs,
                'idf': self.idf,
                'doc_len': self.doc_len
            }
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            
    def load(self, file_path: str):
        """加载检索器状态"""
        with open(file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
            
        # 重建文档列表
        self.documents = []
        self.document_embeddings = []
        for text, metadata in state['documents']:
            embedding = np.array(state['embeddings'][len(self.documents)])
            doc = Document(text=text, metadata=metadata, embedding=embedding)
            self.documents.append(doc)
            self.document_embeddings.append(embedding)
            
        # 恢复BM25参数
        bm25_params = state['bm25_params']
        self.k1 = bm25_params['k1']
        self.b = bm25_params['b']
        self.avgdl = bm25_params['avgdl']
        self.doc_freqs = bm25_params['doc_freqs']
        self.idf = bm25_params['idf']
        self.doc_len = bm25_params['doc_len'] 