from typing import Dict, Any, List, Optional
import numpy as np
from loguru import logger

from .component import Component
from ..embedding.embedding_model import EmbeddingModel

class EmbeddingManager(Component):
    """嵌入模型管理器，负责文本向量化"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化嵌入模型管理器
        
        Args:
            config: 组件配置
        """
        super().__init__(config)
        self.embedding_model = None
    
    def initialize(self) -> None:
        """初始化嵌入模型管理器"""
        # 创建嵌入模型
        self.embedding_model = EmbeddingModel(
            model_path=self.config.get("embedding_model_id", "models/bge-base-zh-v1.5"),
            device=self.config.get("device", "cpu")
        )
        
        # 注册到依赖容器
        self.container.register("embedding_manager", self)
        
        # 注册事件处理器
        self.register_event_handlers()
        
        logger.info("嵌入模型管理器初始化完成")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """获取文本的嵌入向量
        
        Args:
            text: 文本
            
        Returns:
            嵌入向量
        """
        return self.embedding_model.encode(text)
    
    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """批量获取文本的嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        return self.embedding_model.encode_batch(texts)
    
    def register_event_handlers(self) -> None:
        """注册事件处理器"""
        # 监听配置更新事件
        self.event_system.subscribe("config_updated", self._on_config_updated)
    
    def _on_config_updated(self, config: Dict[str, Any]) -> None:
        """配置更新事件处理器
        
        Args:
            config: 更新后的配置
        """
        # 检查是否需要更新嵌入模型
        if (config.get("embedding_model_id") != self.config.get("embedding_model_id") or
            config.get("device") != self.config.get("device")):
            
            logger.info("配置已更改，重新初始化嵌入模型")
            
            # 更新配置
            self.config.update(config)
            
            # 重新创建嵌入模型
            self.embedding_model = EmbeddingModel(
                model_path=self.config.get("embedding_model_id", "models/bge-base-zh-v1.5"),
                device=self.config.get("device", "cpu")
            )
            
            # 发布嵌入模型更新事件
            self.event_system.publish("embedding_model_updated", embedding_model=self.embedding_model)
    
    def cleanup(self) -> None:
        """清理资源"""
        # 如果嵌入模型有需要清理的资源，在这里处理
        pass 