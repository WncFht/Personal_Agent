"""
RSS 服务接口
定义 RSS 相关操作的接口
"""

from abc import abstractmethod
from typing import List, Optional, Dict, Any, Tuple

from .base_service import Service


class RSSService(Service):
    """RSS 服务接口"""
    
    @property
    def name(self) -> str:
        """服务名称"""
        return "rss_service"
    
    @abstractmethod
    def get_feeds(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取所有 RSS 源
        
        Args:
            category (Optional[str], optional): 分类过滤. Defaults to None.
            
        Returns:
            List[Dict[str, Any]]: RSS 源列表
        """
        pass
    
    @abstractmethod
    def get_feed_by_id(self, feed_id: int) -> Optional[Dict[str, Any]]:
        """
        根据 ID 获取 RSS 源
        
        Args:
            feed_id (int): RSS 源 ID
            
        Returns:
            Optional[Dict[str, Any]]: RSS 源信息，不存在时返回 None
        """
        pass
    
    @abstractmethod
    def add_feed(self, url: str, category: str = "") -> Tuple[int, str]:
        """
        添加 RSS 源
        
        Args:
            url (str): RSS 源 URL
            category (str, optional): 分类. Defaults to "".
            
        Returns:
            Tuple[int, str]: (feed_id, message)，feed_id 为 -1 表示添加失败
        """
        pass
    
    @abstractmethod
    def delete_feed(self, feed_id: int) -> bool:
        """
        删除 RSS 源
        
        Args:
            feed_id (int): RSS 源 ID
            
        Returns:
            bool: 是否删除成功
        """
        pass
    
    @abstractmethod
    def update_feeds(self, feed_id: Optional[int] = None) -> Dict[str, Any]:
        """
        更新 RSS 源
        
        Args:
            feed_id (Optional[int], optional): 指定要更新的 RSS 源 ID. Defaults to None.
            
        Returns:
            Dict[str, Any]: 更新结果
        """
        pass
    
    @abstractmethod
    def get_entries(self, 
                   feed_id: Optional[int] = None, 
                   limit: int = 20, 
                   offset: int = 0,
                   unread_only: bool = False) -> List[Dict[str, Any]]:
        """
        获取条目
        
        Args:
            feed_id (Optional[int], optional): RSS 源 ID. Defaults to None.
            limit (int, optional): 返回条目数量限制. Defaults to 20.
            offset (int, optional): 分页偏移量. Defaults to 0.
            unread_only (bool, optional): 只返回未读条目. Defaults to False.
            
        Returns:
            List[Dict[str, Any]]: 条目列表
        """
        pass
    
    @abstractmethod
    def get_entry_by_id(self, entry_id: int) -> Optional[Dict[str, Any]]:
        """
        根据 ID 获取条目
        
        Args:
            entry_id (int): 条目 ID
            
        Returns:
            Optional[Dict[str, Any]]: 条目信息，不存在时返回 None
        """
        pass
    
    @abstractmethod
    def mark_entry_read(self, entry_id: int, read: bool = True) -> bool:
        """
        标记条目为已读/未读
        
        Args:
            entry_id (int): 条目 ID
            read (bool, optional): 是否已读. Defaults to True.
            
        Returns:
            bool: 是否标记成功
        """
        pass
    
    @abstractmethod
    def mark_feed_read(self, feed_id: int, read: bool = True) -> int:
        """
        标记 RSS 源所有条目为已读/未读
        
        Args:
            feed_id (int): RSS 源 ID
            read (bool, optional): 是否已读. Defaults to True.
            
        Returns:
            int: 标记的条目数量
        """
        pass
    
    @abstractmethod
    def get_feed_stats(self) -> Dict[str, Any]:
        """
        获取 RSS 源统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        pass 