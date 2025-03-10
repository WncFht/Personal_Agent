from typing import List, Union
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from loguru import logger

class EmbeddingModel:
    """Embedding模型封装类"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """初始化embedding模型
        
        Args:
            model_path: 模型路径或huggingface模型ID
            device: 设备类型，'cpu'或'cuda'
        """
        self.device = device
        
        # 加载模型和分词器
        logger.info(f"Loading embedding model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        
        logger.info("Embedding model loaded successfully")
        
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """将文本编码为向量
        
        Args:
            texts: 单个文本或文本列表
            normalize: 是否对向量进行归一化
            
        Returns:
            np.ndarray: 文本向量或向量列表
        """
        # 确保输入是列表
        if isinstance(texts, str):
            texts = [texts]
            
        # 对文本进行编码
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # 将数据移到指定设备
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # 获取embedding
        with torch.no_grad():
            outputs = self.model(**encoded)
            # 使用最后一层隐藏状态的平均值作为句子表示
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        # 转换为numpy数组
        embeddings = embeddings.cpu().numpy()
        
        # 归一化
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
        # 如果输入是单个文本，返回单个向量
        if len(texts) == 1:
            return embeddings[0]
            
        return embeddings 