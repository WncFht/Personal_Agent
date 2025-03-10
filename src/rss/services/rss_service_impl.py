"""
RSS 服务实现
实现 RSS 相关操作
"""

import logging
from typing import List, Optional, Dict, Any, Tuple, cast

from src.core.services import RSSService
from src.rss.storage import RSSStorage
from src.rss.parser import RSSParser
from src.rss.models import Feed, Entry

logger = logging.getLogger(__name__)


class RSSServiceImpl(RSSService):
    """RSS 服务实现"""
    
    def __init__(self, db_path: str = "data/rss.db"):
        """
        初始化 RSS 服务
        
        Args:
            db_path (str, optional): 数据库路径. Defaults to "data/rss.db".
        """
        self.db_path = db_path
        self.storage = None
        self.parser = None
    
    def initialize(self, service_manager) -> None:
        """
        初始化服务
        
        Args:
            service_manager: 服务管理器实例
        """
        logger.info(f"Initializing RSS service with database: {self.db_path}")
        self.storage = RSSStorage(self.db_path)
        self.parser = RSSParser()
    
    def shutdown(self) -> None:
        """关闭服务"""
        logger.info("Shutting down RSS service")
        # 目前没有需要特别关闭的资源
        pass
    
    def get_feeds(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取所有 RSS 源
        
        Args:
            category (Optional[str], optional): 分类过滤. Defaults to None.
            
        Returns:
            List[Dict[str, Any]]: RSS 源列表
        """
        feeds = self.storage.get_feeds(category)
        return [self._feed_to_dict(feed) for feed in feeds]
    
    def get_feed_by_id(self, feed_id: int) -> Optional[Dict[str, Any]]:
        """
        根据 ID 获取 RSS 源
        
        Args:
            feed_id (int): RSS 源 ID
            
        Returns:
            Optional[Dict[str, Any]]: RSS 源信息，不存在时返回 None
        """
        feed = self.storage.get_feed_by_id(feed_id)
        if not feed:
            return None
        return self._feed_to_dict(feed)
    
    def add_feed(self, url: str, category: str = "") -> Tuple[int, str]:
        """
        添加 RSS 源
        
        Args:
            url (str): RSS 源 URL
            category (str, optional): 分类. Defaults to "".
            
        Returns:
            Tuple[int, str]: (feed_id, message)，feed_id 为 -1 表示添加失败
        """
        # 解析 RSS 源
        feed = self.parser.parse_feed(url)
        if not feed:
            return -1, f"无法解析RSS源: {url}"
        
        # 设置分类
        feed.category = category
        
        # 添加到数据库
        feed_id = self.storage.add_feed(feed)
        if feed_id <= 0:
            return -1, f"添加RSS源失败: {url}"
        
        # 获取条目
        entries = self.parser.parse_entries(url, feed_id)
        count = self.storage.add_entries(entries)
        
        return feed_id, f"成功添加RSS源: {feed.title}\n已添加 {count} 个条目"
    
    def delete_feed(self, feed_id: int) -> bool:
        """
        删除 RSS 源
        
        Args:
            feed_id (int): RSS 源 ID
            
        Returns:
            bool: 是否删除成功
        """
        return self.storage.delete_feed(feed_id)
    
    def update_feeds(self, feed_id: Optional[int] = None) -> Dict[str, Any]:
        """
        更新 RSS 源
        
        Args:
            feed_id (Optional[int], optional): 指定要更新的 RSS 源 ID. Defaults to None.
            
        Returns:
            Dict[str, Any]: 更新结果
        """
        result = {
            "success": True,
            "total_feeds": 0,
            "updated_feeds": 0,
            "new_entries": 0,
            "errors": []
        }
        
        if feed_id:
            # 更新指定的 RSS 源
            feed = self.storage.get_feed_by_id(feed_id)
            if not feed:
                result["success"] = False
                result["errors"].append(f"未找到RSS源: ID {feed_id}")
                return result
                
            result["total_feeds"] = 1
            
            try:
                entries = self.parser.parse_entries(feed.url, feed.id)
                count = self.storage.add_entries(entries)
                self.storage.update_feed_timestamp(feed.id)
                
                result["updated_feeds"] = 1
                result["new_entries"] = count
            except Exception as e:
                result["success"] = False
                result["errors"].append(f"更新RSS源失败: {feed.title} - {str(e)}")
        else:
            # 更新所有 RSS 源
            feeds = self.storage.get_feeds()
            result["total_feeds"] = len(feeds)
            
            for feed in feeds:
                try:
                    entries = self.parser.parse_entries(feed.url, feed.id)
                    count = self.storage.add_entries(entries)
                    self.storage.update_feed_timestamp(feed.id)
                    
                    result["updated_feeds"] += 1
                    result["new_entries"] += count
                except Exception as e:
                    result["errors"].append(f"更新RSS源失败: {feed.title} - {str(e)}")
        
        if result["errors"]:
            result["success"] = False
            
        return result
    
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
        entries = self.storage.get_entries(feed_id, limit, offset, unread_only)
        return [self._entry_to_dict(entry) for entry in entries]
    
    def get_entry_by_id(self, entry_id: int) -> Optional[Dict[str, Any]]:
        """
        根据 ID 获取条目
        
        Args:
            entry_id (int): 条目 ID
            
        Returns:
            Optional[Dict[str, Any]]: 条目信息，不存在时返回 None
        """
        entry = self.storage.get_entry_by_id(entry_id)
        if not entry:
            return None
        return self._entry_to_dict(entry)
    
    def mark_entry_read(self, entry_id: int, read: bool = True) -> bool:
        """
        标记条目为已读/未读
        
        Args:
            entry_id (int): 条目 ID
            read (bool, optional): 是否已读. Defaults to True.
            
        Returns:
            bool: 是否标记成功
        """
        return self.storage.mark_entry_read(entry_id, read)
    
    def mark_feed_read(self, feed_id: int, read: bool = True) -> int:
        """
        标记 RSS 源所有条目为已读/未读
        
        Args:
            feed_id (int): RSS 源 ID
            read (bool, optional): 是否已读. Defaults to True.
            
        Returns:
            int: 标记的条目数量
        """
        return self.storage.mark_feed_read(feed_id, read)
    
    def get_feed_stats(self) -> Dict[str, Any]:
        """
        获取 RSS 源统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        feeds = self.storage.get_feeds()
        
        # 计算总条目数和未读条目数
        total_entries = 0
        total_unread = 0
        feed_stats = []
        
        for feed in feeds:
            entries_count = self.storage.get_entries_count(feed.id)
            unread_count = self.storage.get_unread_count(feed.id)
            
            total_entries += entries_count
            total_unread += unread_count
            
            feed_stats.append({
                "id": feed.id,
                "title": feed.title,
                "category": feed.category or "未分类",
                "entries": entries_count,
                "unread": unread_count,
                "last_updated": feed.last_updated.isoformat() if feed.last_updated else None
            })
        
        # 按分类统计
        categories = {}
        for feed in feed_stats:
            category = feed["category"]
            if category not in categories:
                categories[category] = {
                    "feeds": 0,
                    "entries": 0,
                    "unread": 0
                }
            
            categories[category]["feeds"] += 1
            categories[category]["entries"] += feed["entries"]
            categories[category]["unread"] += feed["unread"]
        
        # 转换为列表
        category_stats = [
            {
                "name": name,
                "feeds": stats["feeds"],
                "entries": stats["entries"],
                "unread": stats["unread"]
            }
            for name, stats in categories.items()
        ]
        
        return {
            "total_feeds": len(feeds),
            "total_entries": total_entries,
            "total_unread": total_unread,
            "feeds": feed_stats,
            "categories": category_stats
        }
    
    def _feed_to_dict(self, feed: Feed) -> Dict[str, Any]:
        """
        将 Feed 对象转换为字典
        
        Args:
            feed (Feed): Feed 对象
            
        Returns:
            Dict[str, Any]: 字典表示
        """
        return {
            "id": feed.id,
            "title": feed.title,
            "url": feed.url,
            "link": feed.link,
            "description": feed.description,
            "category": feed.category or "未分类",
            "last_updated": feed.last_updated.isoformat() if feed.last_updated else None,
            "unread_count": self.storage.get_unread_count(feed.id) if feed.id else 0
        }
    
    def _entry_to_dict(self, entry: Entry) -> Dict[str, Any]:
        """
        将 Entry 对象转换为字典
        
        Args:
            entry (Entry): Entry 对象
            
        Returns:
            Dict[str, Any]: 字典表示
        """
        feed = None
        if entry.feed_id:
            feed = self.storage.get_feed_by_id(entry.feed_id)
            
        return {
            "id": entry.id,
            "feed_id": entry.feed_id,
            "feed_title": feed.title if feed else None,
            "title": entry.title,
            "link": entry.link,
            "published_date": entry.published_date.isoformat() if entry.published_date else None,
            "content": entry.content,
            "summary": entry.summary,
            "read": entry.read
        } 