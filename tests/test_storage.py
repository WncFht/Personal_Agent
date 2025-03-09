#!/usr/bin/env python
"""
RSS数据存储测试
"""
import os
import sys
import unittest
import tempfile
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rss.storage import RSSStorage
from src.rss.models import Feed, Entry

class TestRSSStorage(unittest.TestCase):
    """RSS数据存储测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 使用临时文件作为测试数据库
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.storage = RSSStorage(self.temp_db.name)
        
        # 创建测试数据
        self.test_feed = Feed(
            title="测试Feed",
            url="https://example.com/feed.xml",
            site_url="https://example.com",
            description="测试描述",
            last_updated=datetime.now(),
            category="测试分类"
        )
        
        # 添加测试Feed并获取ID
        self.feed_id = self.storage.add_feed(self.test_feed)
        
        # 创建测试条目
        self.test_entry = Entry(
            feed_id=self.feed_id,
            title="测试条目",
            link="https://example.com/entry1",
            published_date=datetime.now(),
            author="测试作者",
            summary="测试摘要",
            content="测试内容",
            read_status=False
        )
        
        # 添加测试条目
        self.storage.add_entries([self.test_entry])
    
    def tearDown(self):
        """测试后清理"""
        # 关闭数据库连接
        del self.storage
        
        # 删除临时数据库文件
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_add_feed(self):
        """测试添加Feed"""
        # 添加新Feed
        new_feed = Feed(
            title="新Feed",
            url="https://example.com/new-feed.xml",
            site_url="https://example.com/new",
            description="新描述",
            last_updated=datetime.now()
        )
        
        feed_id = self.storage.add_feed(new_feed)
        self.assertGreater(feed_id, 0)
        
        # 添加重复URL的Feed
        duplicate_feed = Feed(
            title="重复Feed",
            url="https://example.com/feed.xml",  # 与test_feed相同的URL
            site_url="https://example.com",
            description="重复描述",
            last_updated=datetime.now()
        )
        
        feed_id = self.storage.add_feed(duplicate_feed)
        self.assertEqual(feed_id, self.feed_id)  # 应返回已存在的Feed ID
    
    def test_update_feed(self):
        """测试更新Feed"""
        # 更新标题
        success = self.storage.update_feed(self.feed_id, title="更新的标题")
        self.assertTrue(success)
        
        # 验证更新
        feed = self.storage.get_feed_by_id(self.feed_id)
        self.assertEqual(feed.title, "更新的标题")
        
        # 更新多个字段
        success = self.storage.update_feed(
            self.feed_id,
            site_url="https://example.com/updated",
            description="更新的描述",
            category="更新的分类"
        )
        self.assertTrue(success)
        
        # 验证更新
        feed = self.storage.get_feed_by_id(self.feed_id)
        self.assertEqual(feed.site_url, "https://example.com/updated")
        self.assertEqual(feed.description, "更新的描述")
        self.assertEqual(feed.category, "更新的分类")
        
        # 更新不存在的Feed
        success = self.storage.update_feed(9999, title="不存在")
        self.assertFalse(success)
    
    def test_delete_feed(self):
        """测试删除Feed"""
        # 添加一个新Feed用于删除
        new_feed = Feed(
            title="要删除的Feed",
            url="https://example.com/delete-feed.xml",
            site_url="https://example.com/delete",
            description="要删除的描述",
            last_updated=datetime.now()
        )
        
        feed_id = self.storage.add_feed(new_feed)
        
        # 删除Feed
        success = self.storage.delete_feed(feed_id)
        self.assertTrue(success)
        
        # 验证删除
        feed = self.storage.get_feed_by_id(feed_id)
        self.assertIsNone(feed)
        
        # 删除不存在的Feed
        success = self.storage.delete_feed(9999)
        self.assertFalse(success)
    
    def test_get_feeds(self):
        """测试获取Feeds"""
        # 添加一个带分类的Feed
        category_feed = Feed(
            title="分类Feed",
            url="https://example.com/category-feed.xml",
            site_url="https://example.com/category",
            description="分类描述",
            last_updated=datetime.now(),
            category="特定分类"
        )
        
        self.storage.add_feed(category_feed)
        
        # 获取所有Feed
        feeds = self.storage.get_feeds()
        self.assertGreaterEqual(len(feeds), 2)  # 至少有两个Feed
        
        # 按分类获取Feed
        feeds = self.storage.get_feeds(category="特定分类")
        self.assertEqual(len(feeds), 1)
        self.assertEqual(feeds[0].title, "分类Feed")
        
        # 获取不存在的分类
        feeds = self.storage.get_feeds(category="不存在的分类")
        self.assertEqual(len(feeds), 0)
    
    def test_add_entries(self):
        """测试添加条目"""
        # 创建多个条目
        entries = [
            Entry(
                feed_id=self.feed_id,
                title=f"条目{i}",
                link=f"https://example.com/entry{i+10}",  # 使用更大的序号避免冲突
                published_date=datetime.now(),
                author="作者",
                summary="摘要",
                content="内容",
                read_status=False
            )
            for i in range(1, 4)
        ]
        
        # 添加条目
        count = self.storage.add_entries(entries)
        self.assertEqual(count, 3)
        
        # 添加重复条目
        count = self.storage.add_entries(entries)
        self.assertEqual(count, 0)  # 不应添加任何条目
        
        # 添加空列表
        count = self.storage.add_entries([])
        self.assertEqual(count, 0)
    
    def test_get_entries(self):
        """测试获取条目"""
        # 添加多个条目
        entries = [
            Entry(
                feed_id=self.feed_id,
                title=f"条目{i}",
                link=f"https://example.com/entry{i}",
                published_date=datetime.now(),
                author="作者",
                summary="摘要",
                content="内容",
                read_status=i % 2 == 0  # 偶数为已读
            )
            for i in range(1, 11)
        ]
        
        self.storage.add_entries(entries)
        
        # 获取所有条目
        all_entries = self.storage.get_entries()
        self.assertGreaterEqual(len(all_entries), 10)
        
        # 获取特定Feed的条目
        feed_entries = self.storage.get_entries(feed_id=self.feed_id)
        self.assertGreaterEqual(len(feed_entries), 10)
        
        # 获取限制数量的条目
        limited_entries = self.storage.get_entries(limit=5)
        self.assertEqual(len(limited_entries), 5)
        
        # 获取未读条目
        unread_entries = self.storage.get_entries(unread_only=True)
        for entry in unread_entries:
            self.assertFalse(entry.read_status)
    
    def test_mark_entry_as_read(self):
        """测试标记条目为已读"""
        # 获取一个未读条目
        entries = self.storage.get_entries(unread_only=True)
        if not entries:
            self.skipTest("没有未读条目可供测试")
        
        entry_id = entries[0].id
        
        # 标记为已读
        success = self.storage.mark_entry_as_read(entry_id)
        self.assertTrue(success)
        
        # 验证标记
        entry = self.storage.get_entry_by_id(entry_id)
        self.assertTrue(entry.read_status)
        
        # 标记不存在的条目
        success = self.storage.mark_entry_as_read(9999)
        self.assertFalse(success)
    
    def test_mark_all_entries_as_read(self):
        """测试标记所有条目为已读"""
        # 添加多个未读条目
        entries = [
            Entry(
                feed_id=self.feed_id,
                title=f"未读条目{i}",
                link=f"https://example.com/unread{i}",
                published_date=datetime.now(),
                author="作者",
                summary="摘要",
                content="内容",
                read_status=False
            )
            for i in range(1, 6)
        ]
        
        self.storage.add_entries(entries)
        
        # 标记所有条目为已读
        count = self.storage.mark_all_entries_as_read()
        self.assertGreaterEqual(count, 5)
        
        # 验证标记
        unread_count = self.storage.get_unread_count()
        self.assertEqual(unread_count, 0)
    
    def test_get_unread_count(self):
        """测试获取未读数量"""
        # 添加多个条目，一半已读一半未读
        entries = [
            Entry(
                feed_id=self.feed_id,
                title=f"计数条目{i}",
                link=f"https://example.com/count{i}",
                published_date=datetime.now(),
                author="作者",
                summary="摘要",
                content="内容",
                read_status=i % 2 == 0  # 偶数为已读
            )
            for i in range(1, 11)
        ]
        
        self.storage.add_entries(entries)
        
        # 获取总未读数量
        total_unread = self.storage.get_unread_count()
        self.assertGreaterEqual(total_unread, 5)
        
        # 获取特定Feed的未读数量
        feed_unread = self.storage.get_unread_count(self.feed_id)
        self.assertGreaterEqual(feed_unread, 5)

if __name__ == '__main__':
    unittest.main() 