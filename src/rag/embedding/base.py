"""
基础embedding抽象类
"""
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import torch
import torch.nn.functional as F

class BaseEmbedding(ABC):
    """基础embedding抽象类"""
    
    def __init__(self, model_path: str):
        """
        初始化
        
        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.name = "base_embedding"
        
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        将文本编码为向量
        
        Args:
            texts: 单个文本或文本列表
            normalize: 是否对向量进行归一化
            
        Returns:
            numpy数组，shape为(n_texts, embedding_dim)
        """
        raise NotImplementedError
        
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            a: 向量a
            b: 向量b
            
        Returns:
            余弦相似度值
        """
        if len(a.shape) == 1:
            a = a.reshape(1, -1)
        if len(b.shape) == 1:
            b = b.reshape(1, -1)
            
        a_norm = np.linalg.norm(a, axis=1)
        b_norm = np.linalg.norm(b, axis=1)
        
        # 避免除零
        a_norm = np.where(a_norm == 0, 1e-10, a_norm)
        b_norm = np.where(b_norm == 0, 1e-10, b_norm)
        
        similarity = np.dot(a, b.T) / (a_norm.reshape(-1, 1) @ b_norm.reshape(1, -1))
        return similarity 