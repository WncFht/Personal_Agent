"""
RSS数据模型定义
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

@dataclass
class Feed:
    """RSS源数据模型"""
    title: str
    url: str
    site_url: str
    description: str
    last_updated: datetime
    category: Optional[str] = None
    id: Optional[int] = None
    
@dataclass
class Entry:
    """RSS条目数据模型"""
    feed_id: int
    title: str
    link: str
    published_date: datetime
    author: str
    summary: str
    content: str
    read_status: bool = False
    id: Optional[int] = None 