from typing import Dict, List, Optional, Tuple, Union, Iterator
from datetime import datetime, timedelta
from loguru import logger

from .config import RAGConfig
from .core.system_manager import SystemManager
from ..rss.models import Entry
from ..utils.dependency_container import container
from .templates import ORIGINAL_RAG_PROMPT_TEMPLATE, ENHANCED_RAG_PROMPT_TEMPLATE

class RSSRAG:
    """RSS-RAG系统主类（新版本，基于模块化架构）"""
    
    def __init__(self, config: RAGConfig):
        """初始化RSS-RAG系统
        
        Args:
            config: 配置对象
        """
        self.config = config
        
        # 将RAGConfig转换为字典
        config_dict = {
            "base_dir": config.base_dir,
            "device": config.device,
            "llm_type": config.llm_type,
            "llm_model_id": config.llm_model_id,
            "embedding_model_id": config.embedding_model_id,
            "reranker_model_id": config.reranker_model_id,
            "system_prompt": config.system_prompt,
            "deepseek_api_key": config.deepseek_api_key,
            "deepseek_model": config.deepseek_model,
            "deepseek_base_url": config.deepseek_base_url,
            "openai_api_key": config.openai_api_key,
            "openai_model": config.openai_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "use_model_for_splitting": config.use_model_for_splitting,
            "sentence_splitter_model": config.sentence_splitter_model,
            "top_k": config.top_k,
            "search_weights": config.search_weights,
            "use_reranker": config.use_reranker,
            "use_query_enhancement": config.use_query_enhancement,
            "use_cache": config.use_cache,
            "cache_dir": config.cache_dir,
            "cache_size": config.cache_size,
            "cache_ttl": config.cache_ttl,
            "max_history_days": config.max_history_days,
            "update_interval": config.update_interval,
            "use_parallel_processing": config.use_parallel_processing,
            "max_workers": config.max_workers
        }
        
        # 创建系统管理器
        self.system_manager = SystemManager()
        
        # 初始化系统
        try:
            # 先初始化系统
            self.system_manager.initialize()
            
            # 更新配置
            self.system_manager.update_config(config_dict)
            
            logger.info("RSS-RAG系统初始化成功")
        except Exception as e:
            logger.error(f"RSS-RAG系统初始化失败: {e}")
            raise
    
    def process_entry(self, entry: Entry) -> None:
        """处理单个RSS条目
        
        Args:
            entry: RSS条目
        """
        self.system_manager.process_entries([entry])
    
    def process_entries(self, entries: List[Entry]) -> None:
        """批量处理RSS条目
        
        Args:
            entries: RSS条目列表
        """
        self.system_manager.process_entries(entries)
    
    def load_from_rss_db(self, db_path: str, days: int = 30, incremental: bool = True) -> None:
        """从RSS数据库加载数据
        
        Args:
            db_path: 数据库文件路径
            days: 加载最近几天的数据
            incremental: 是否增量加载（只加载上次处理后的新数据）
        """
        self.system_manager.load_from_rss_db(db_path, days, incremental)
    
    def search(self, 
              query: str,
              feed_id: Optional[int] = None,
              date_range: Optional[Tuple[datetime, datetime]] = None,
              top_k: Optional[int] = None) -> List[Tuple[str, float, Dict]]:
        """搜索相关内容"""
        retrieval_manager = container.get("retrieval_manager")
        return retrieval_manager.search(query, feed_id, date_range, top_k)
    
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
        return self.system_manager.answer(query, feed_id, date_range, top_k)
    
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
        return self.system_manager.answer_stream(query, feed_id, date_range, top_k)
    
    def clean_old_entries(self) -> None:
        """清理过期的RSS条目"""
        self.system_manager.clean_old_entries()
    
    def save_state(self) -> None:
        """保存系统状态"""
        self.system_manager.save_state()
    
    def load_state(self, db_path: Optional[str] = None, days: int = 30) -> None:
        """加载系统状态
        
        Args:
            db_path: RSS数据库路径，如果提供则在没有找到保存状态时从数据库加载
            days: 加载最近几天的数据
        """
        # 尝试加载状态
        retrieval_manager = container.get("retrieval_manager")
        
        # 检查是否有数据
        doc_count = len(retrieval_manager.retriever.documents) if hasattr(retrieval_manager, 'retriever') else 0
        
        # 如果没有数据且提供了数据库路径，则从数据库加载
        if doc_count == 0 and db_path:
            logger.info("从 RSS 数据库加载数据...")
            self.load_from_rss_db(db_path, days)
            # 保存新的状态
            self.save_state()
            logger.info("系统状态已更新并保存") 