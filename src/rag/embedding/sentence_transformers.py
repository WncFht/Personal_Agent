"""
基于sentence-transformers的embedding实现
"""
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedding

class SentenceTransformerEmbedding(BaseEmbedding):
    """基于sentence-transformers的embedding实现"""
    
    def __init__(self, model_path: str = "models/bge-base-zh-v1.5"):
        """
        初始化
        
        Args:
            model_path: 模型路径或名称，默认使用bge-base-zh-v1.5
                      如果模型不存在，会自动从HuggingFace下载
        """
        super().__init__(model_path)
        try:
            self.model = SentenceTransformer(model_path)
        except:
            # 如果本地模型不存在，则从HuggingFace下载
            self.model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
        self.name = "bge_embedding"
        
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        将文本编码为向量
        
        Args:
            texts: 单个文本或文本列表
            normalize: 是否对向量进行归一化
            
        Returns:
            numpy数组，shape为(n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # 对于BGE模型，需要添加特殊前缀以提高效果
        texts = [f"为这个句子生成表示：{text}" for text in texts]
            
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        return embeddings 