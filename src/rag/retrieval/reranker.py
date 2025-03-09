"""
重排序模块
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from loguru import logger

class Reranker:
    """重排序基类"""
    
    def __init__(self):
        pass
        
    def rerank(self, 
              query: str, 
              documents: List[Tuple[str, float, Dict]], 
              top_k: int = 3) -> List[Tuple[str, float, Dict]]:
        """
        对检索结果进行重排序
        
        Args:
            query: 查询文本
            documents: 检索结果列表，每项为(文本, 得分, 元数据)
            top_k: 返回结果数量
            
        Returns:
            重排序后的结果列表
        """
        raise NotImplementedError("子类必须实现此方法")

class BGEM3Reranker(Reranker):
    """基于BGE-M3的重排序器"""
    
    def __init__(self, model_path: str = "models/bge-reranker-base", device: str = "cuda"):
        """
        初始化
        
        Args:
            model_path: 模型路径
            device: 设备类型，'cuda'或'cpu'
        """
        super().__init__()
        self.model_path = model_path
        self.device = device
        self._load_model()
        
    def _load_model(self):
        """加载模型"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_path, device=self.device)
            logger.info(f"成功加载重排序模型: {self.model_path}")
        except ImportError:
            logger.error("未安装sentence-transformers，无法使用重排序功能")
            raise ImportError("请安装sentence-transformers: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"加载重排序模型失败: {e}")
            raise
            
    def rerank(self, 
              query: str, 
              documents: List[Tuple[str, float, Dict]], 
              top_k: int = 3) -> List[Tuple[str, float, Dict]]:
        """
        使用BGE-M3模型对检索结果进行重排序
        
        Args:
            query: 查询文本
            documents: 检索结果列表，每项为(文本, 得分, 元数据)
            top_k: 返回结果数量
            
        Returns:
            重排序后的结果列表
        """
        if not documents:
            return []
            
        # 准备输入数据
        texts = [doc[0] for doc in documents]
        metadata = [doc[2] for doc in documents]
        
        # 构建模型输入
        model_inputs = [(query, text) for text in texts]
        
        # 计算相似度得分
        try:
            scores = self.model.predict(model_inputs)
            
            # 组合结果
            reranked_results = [(texts[i], float(scores[i]), metadata[i]) 
                               for i in range(len(texts))]
            
            # 按得分排序
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            
            # 返回top_k结果
            return reranked_results[:top_k]
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            # 如果重排序失败，返回原始结果
            return documents[:top_k] 