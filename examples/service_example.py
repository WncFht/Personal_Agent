#!/usr/bin/env python
"""
服务架构示例
展示如何使用新的服务架构
"""

import os
import sys
import logging
from typing import List, Dict, Any

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.services import ServiceManager
from src.core.services import RSSService
from src.rss.services import RSSServiceImpl


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def print_feeds(feeds: List[Dict[str, Any]]) -> None:
    """打印 RSS 源列表"""
    print("\n=== RSS 源列表 ===")
    if not feeds:
        print("没有 RSS 源")
        return
        
    for feed in feeds:
        print(f"ID: {feed['id']}, 标题: {feed['title']}, 分类: {feed['category']}, 未读: {feed['unread_count']}")


def main():
    """主函数"""
    # 创建服务管理器
    service_manager = ServiceManager()
    
    # 创建 RSS 服务
    rss_service = RSSServiceImpl(db_path="data/rss.db")
    
    # 注册服务
    service_manager.register(RSSService, rss_service)
    
    # 初始化服务
    try:
        service_manager.initialize()
        logger.info("服务初始化成功")
    except Exception as e:
        logger.error(f"服务初始化失败: {e}")
        return
    
    try:
        # 获取 RSS 服务
        rss_service = service_manager.get(RSSService)
        
        # 获取所有 RSS 源
        feeds = rss_service.get_feeds()
        print_feeds(feeds)
        
        # 获取统计信息
        stats = rss_service.get_feed_stats()
        print(f"\n总 RSS 源数: {stats['total_feeds']}")
        print(f"总条目数: {stats['total_entries']}")
        print(f"总未读数: {stats['total_unread']}")
        
        # 按分类统计
        print("\n=== 分类统计 ===")
        for category in stats['categories']:
            print(f"分类: {category['name']}, RSS 源数: {category['feeds']}, "
                  f"条目数: {category['entries']}, 未读数: {category['unread']}")
        
    finally:
        # 关闭服务
        service_manager.shutdown()
        logger.info("服务已关闭")


if __name__ == "__main__":
    main() 