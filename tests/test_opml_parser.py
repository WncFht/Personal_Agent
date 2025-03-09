"""
测试OPML解析器功能
"""
import os
import sys
import unittest
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rss.opml_parser import OPMLParser

class TestOPMLParser(unittest.TestCase):
    """测试OPML解析器"""
    
    def setUp(self):
        """测试前准备"""
        self.parser = OPMLParser()
        
        # 创建测试OPML内容
        self.test_opml = """<?xml version="1.0" encoding="UTF-8"?>
        <opml version="2.0">
            <head>
                <title>RSS订阅列表</title>
            </head>
            <body>
                <outline text="新闻">
                    <outline text="科技新闻" title="科技新闻" type="rss" xmlUrl="https://example.com/tech.xml" htmlUrl="https://example.com/tech" />
                    <outline text="财经新闻" title="财经新闻" type="rss" xmlUrl="https://example.com/finance.xml" htmlUrl="https://example.com/finance" />
                </outline>
                <outline text="博客" title="博客" type="rss" xmlUrl="https://example.com/blog.xml" htmlUrl="https://example.com/blog" />
            </body>
        </opml>
        """
        
        # 创建测试OPML文件
        self.test_file = "tests/data/test.opml"
        os.makedirs(os.path.dirname(self.test_file), exist_ok=True)
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write(self.test_opml)
            
    def test_parse_opml_text(self):
        """测试解析OPML文本"""
        feeds = self.parser.parse_opml_text(self.test_opml)
        
        # 应该解析出3个源
        self.assertEqual(len(feeds), 3)
        
        # 检查URL
        urls = [feed.url for feed in feeds]
        self.assertTrue("https://example.com/tech.xml" in urls)
        self.assertTrue("https://example.com/finance.xml" in urls)
        self.assertTrue("https://example.com/blog.xml" in urls)
        
        # 检查分类
        for feed in feeds:
            if feed.url == "https://example.com/tech.xml" or feed.url == "https://example.com/finance.xml":
                self.assertEqual(feed.category, "新闻")
            elif feed.url == "https://example.com/blog.xml":
                self.assertIsNone(feed.category)
        
    def test_parse_opml_file(self):
        """测试解析OPML文件"""
        feeds = self.parser.parse_opml(self.test_file)
        
        # 应该解析出3个源
        self.assertEqual(len(feeds), 3)
        
        # 检查标题
        titles = [feed.title for feed in feeds]
        self.assertTrue("科技新闻" in titles)
        self.assertTrue("财经新闻" in titles)
        self.assertTrue("博客" in titles)
        
    def test_nonexistent_file(self):
        """测试解析不存在的文件"""
        feeds = self.parser.parse_opml("nonexistent.opml")
        self.assertEqual(len(feeds), 0)
        
    def tearDown(self):
        """测试后清理"""
        # 删除测试文件
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
            
if __name__ == "__main__":
    unittest.main() 