from typing import Dict, Any, List
from loguru import logger

from .component import Component
from ..utils.text_splitter import TextSplitter

class TextProcessor(Component):
    """文本处理组件，负责文本分块等操作"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化文本处理组件
        
        Args:
            config: 组件配置
        """
        super().__init__(config)
        self.text_splitter = None
    
    def initialize(self) -> None:
        """初始化文本处理组件"""
        # 创建文本分割器
        self.text_splitter = TextSplitter(
            chunk_size=self.config.get("chunk_size", 500),
            chunk_overlap=self.config.get("chunk_overlap", 50),
            use_model=self.config.get("use_model_for_splitting", False),
            model_path=self.config.get("sentence_splitter_model", "damo/nlp_bert_document-segmentation_chinese-base"),
            device=self.config.get("device", "cpu")
        )
        
        # 注册到依赖容器
        self.container.register("text_processor", self)
        
        # 注册事件处理器
        self.register_event_handlers()
        
        logger.info("文本处理组件初始化完成")
    
    def split_text(self, text: str) -> List[str]:
        """分割文本
        
        Args:
            text: 要分割的文本
            
        Returns:
            文本块列表
        """
        chunks = self.text_splitter.split_text(text)
        logger.debug(f"文本被分割为 {len(chunks)} 个块")
        return chunks
    
    def split_documents(self, documents: List[str]) -> List[str]:
        """分割多个文档
        
        Args:
            documents: 文档列表
            
        Returns:
            所有文档的文本块列表
        """
        return self.text_splitter.split_documents(documents)
    
    def register_event_handlers(self) -> None:
        """注册事件处理器"""
        # 监听配置更新事件
        self.event_system.subscribe("config_updated", self._on_config_updated)
    
    def _on_config_updated(self, config: Dict[str, Any]) -> None:
        """配置更新事件处理器
        
        Args:
            config: 更新后的配置
        """
        # 检查是否需要更新文本分割器
        if (config.get("chunk_size") != self.config.get("chunk_size") or
            config.get("chunk_overlap") != self.config.get("chunk_overlap") or
            config.get("use_model_for_splitting") != self.config.get("use_model_for_splitting") or
            config.get("sentence_splitter_model") != self.config.get("sentence_splitter_model") or
            config.get("device") != self.config.get("device")):
            
            logger.info("配置已更改，重新初始化文本分割器")
            
            # 更新配置
            self.config.update(config)
            
            # 重新创建文本分割器
            self.text_splitter = TextSplitter(
                chunk_size=self.config.get("chunk_size", 500),
                chunk_overlap=self.config.get("chunk_overlap", 50),
                use_model=self.config.get("use_model_for_splitting", False),
                model_path=self.config.get("sentence_splitter_model", "damo/nlp_bert_document-segmentation_chinese-base"),
                device=self.config.get("device", "cpu")
            )