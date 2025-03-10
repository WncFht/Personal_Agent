from typing import Dict, Any, Optional, List, Tuple, Iterator, Union
from datetime import datetime
from loguru import logger

# 导入组件类
from .component import Component
from .config_manager import ConfigManager
from .text_processor import TextProcessor
from .embedding_manager import EmbeddingManager
from .llm_manager import LLMManager
from .retrieval_manager import RetrievalManager
from .rag_manager import RAGManager

# 导入工具
from ...utils.event_system import event_system
from ...utils.dependency_container import container
from ...rss.models import Entry

class SystemManager:
    """系统管理器，用于初始化和协调所有组件"""
    
    def __init__(self, config: Union[str, Dict[str, Any]] = "config/app_config.json"):
        """初始化系统管理器
        
        Args:
            config: 配置文件路径或配置字典
        """
        self.config = config
        self.components = []
        self.initialized = False
    
    def initialize(self) -> None:
        """初始化系统"""
        if self.initialized:
            logger.warning("系统已经初始化")
            return
        
        logger.info("开始初始化系统...")
        
        # 清理依赖容器和事件系统
        container.clear()
        event_system.clear()
        
        # 注册系统管理器到依赖容器
        container.register("system_manager", self)
        
        # 初始化配置管理器
        if isinstance(self.config, str):
            # 如果是字符串，则作为配置文件路径
            config_manager = ConfigManager(config_path=self.config)
        else:
            # 如果是字典，则作为配置内容
            config_manager = ConfigManager()
            config_manager.update_config(self.config, save_to_user_config=False)
        
        config_manager.initialize()
        self.components.append(config_manager)
        
        # 获取配置
        config = config_manager.config
        
        # 初始化文本处理器
        text_processor = TextProcessor(config)
        text_processor.initialize()
        self.components.append(text_processor)
        
        # 初始化嵌入模型管理器
        embedding_manager = EmbeddingManager(config)
        embedding_manager.initialize()
        self.components.append(embedding_manager)
        
        # 初始化LLM管理器
        llm_manager = LLMManager(config)
        llm_manager.initialize()
        self.components.append(llm_manager)
        
        # 初始化检索管理器
        retrieval_manager = RetrievalManager(config)
        retrieval_manager.initialize()
        self.components.append(retrieval_manager)
        
        # 初始化RAG管理器
        rag_manager = RAGManager(config)
        rag_manager.initialize()
        self.components.append(rag_manager)
        
        self.initialized = True
        logger.info("系统初始化完成")
    
    def shutdown(self) -> None:
        """关闭系统，清理资源"""
        if not self.initialized:
            return
        
        logger.info("开始关闭系统...")
        
        # 逆序清理组件资源
        for component in reversed(self.components):
            try:
                component.cleanup()
            except Exception as e:
                logger.error(f"清理组件 {component.__class__.__name__} 时出错: {e}")
        
        # 清理依赖容器和事件系统
        container.clear()
        event_system.clear()
        
        self.components = []
        self.initialized = False
        logger.info("系统已关闭")
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """更新系统配置
        
        Args:
            new_config: 新配置
        """
        if not self.initialized:
            logger.error("系统未初始化，无法更新配置")
            return
        
        config_manager = container.get("config_manager")
        config_manager.update_config(new_config)
    
    def process_entries(self, entries: List[Entry]) -> None:
        """处理RSS条目
        
        Args:
            entries: RSS条目列表
        """
        if not self.initialized:
            logger.error("系统未初始化，无法处理条目")
            return
        
        rag_manager = container.get("rag_manager")
        rag_manager.process_entries(entries)
    
    def load_from_rss_db(self, db_path: str, days: int = 30, incremental: bool = True) -> None:
        """从RSS数据库加载数据
        
        Args:
            db_path: 数据库文件路径
            days: 加载最近几天的数据
            incremental: 是否增量加载（只加载上次处理后的新数据）
        """
        if not self.initialized:
            logger.error("系统未初始化，无法加载数据")
            return
        
        rag_manager = container.get("rag_manager")
        rag_manager.load_from_rss_db(db_path, days, incremental)
    
    def answer(self, 
              query: str,
              feed_id: Optional[int] = None,
              date_range: Optional[Tuple[datetime, datetime]] = None,
              top_k: Optional[int] = None) -> str:
        """回答问题
        
        Args:
            query: 问题
            feed_id: 指定RSS源ID
            date_range: 日期范围(开始时间, 结束时间)
            top_k: 检索结果数量
            
        Returns:
            str: 回答
        """
        if not self.initialized:
            logger.error("系统未初始化，无法回答问题")
            return "系统未初始化，无法回答问题。"
        
        rag_manager = container.get("rag_manager")
        return rag_manager.answer(query, feed_id, date_range, top_k)
    
    def answer_stream(self, 
                     query: str,
                     feed_id: Optional[int] = None,
                     date_range: Optional[Tuple[datetime, datetime]] = None,
                     top_k: Optional[int] = None) -> Iterator[str]:
        """流式回答问题
        
        Args:
            query: 问题
            feed_id: 指定RSS源ID
            date_range: 日期范围(开始时间, 结束时间)
            top_k: 检索结果数量
            
        Returns:
            Iterator[str]: 流式回答
        """
        if not self.initialized:
            logger.error("系统未初始化，无法回答问题")
            yield "系统未初始化，无法回答问题。"
            return
        
        rag_manager = container.get("rag_manager")
        yield from rag_manager.answer_stream(query, feed_id, date_range, top_k)
    
    def save_state(self) -> None:
        """保存系统状态"""
        if not self.initialized:
            logger.error("系统未初始化，无法保存状态")
            return
        
        retrieval_manager = container.get("retrieval_manager")
        retrieval_manager.save_state()
        
        rag_manager = container.get("rag_manager")
        rag_manager._save_last_processed_timestamp()
        
        logger.info("系统状态已保存")
    
    def clean_old_entries(self) -> None:
        """清理过期的RSS条目"""
        if not self.initialized:
            logger.error("系统未初始化，无法清理条目")
            return
        
        retrieval_manager = container.get("retrieval_manager")
        retrieval_manager.clean_old_entries()
        
        logger.info("过期条目已清理") 