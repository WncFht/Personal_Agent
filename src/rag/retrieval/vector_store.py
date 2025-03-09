"""
向量存储实现
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
import json
import os

class VectorStore:
    """向量存储类，使用FAISS作为后端"""
    
    def __init__(self, embedding_dim: int, index_type: str = "l2"):
        """
        初始化向量存储
        
        Args:
            embedding_dim: 向量维度
            index_type: 索引类型，支持'l2'和'ip'(内积)
        """
        if index_type == "l2":
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == "ip":
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")
            
        self.metadata: List[Dict[str, Any]] = []
        
    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        添加向量和元数据
        
        Args:
            vectors: 向量数组，shape为(n_vectors, embedding_dim)
            metadata: 元数据列表，长度为n_vectors
        """
        if len(vectors) != len(metadata):
            raise ValueError("向量数量与元数据数量不匹配")
            
        self.index.add(vectors)
        self.metadata.extend(metadata)
        
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        搜索最相似的向量
        
        Args:
            query_vector: 查询向量，shape为(embedding_dim,)或(1, embedding_dim)
            k: 返回的最相似向量数量
            
        Returns:
            (distances, indices, metadata)元组：
            - distances: 距离数组
            - indices: 索引数组
            - metadata: 元数据列表
        """
        if len(query_vector.shape) == 1:
            if query_vector.shape[0] != self.index.d:
                raise ValueError(f"查询向量维度不匹配：期望{self.index.d}，实际{query_vector.shape[0]}")
            query_vector = query_vector.reshape(1, -1)
        elif query_vector.shape[1] != self.index.d:
            raise ValueError(f"查询向量维度不匹配：期望{self.index.d}，实际{query_vector.shape[1]}")
            
        if self.index.ntotal == 0:  # 处理空存储的情况
            return np.array([]), np.array([]), []
            
        k = min(k, self.index.ntotal)
        if k == 0:  # 避免k=0的情况
            return np.array([]), np.array([]), []
            
        distances, indices = self.index.search(query_vector, k)
        
        results_metadata = [self.metadata[i] for i in indices[0]]
        return distances[0], indices[0], results_metadata
    
    def save(self, directory: str):
        """
        保存向量存储到目录
        
        Args:
            directory: 保存目录
        """
        os.makedirs(directory, exist_ok=True)
        
        # 保存FAISS索引
        index_path = os.path.join(directory, "index.faiss")
        faiss.write_index(self.index, index_path)
        
        # 保存元数据
        metadata_path = os.path.join(directory, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
    @classmethod
    def load(cls, directory: str, embedding_dim: int, index_type: str = "l2") -> "VectorStore":
        """
        从目录加载向量存储
        
        Args:
            directory: 加载目录
            embedding_dim: 向量维度
            index_type: 索引类型
            
        Returns:
            VectorStore实例
        """
        store = cls(embedding_dim, index_type)
        
        # 加载FAISS索引
        index_path = os.path.join(directory, "index.faiss")
        store.index = faiss.read_index(index_path)
        
        # 加载元数据
        metadata_path = os.path.join(directory, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            store.metadata = json.load(f)
            
        return store 