#!/usr/bin/env python
"""
RSS解析器测试
"""
import os
import sys
import unittest
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rss.parser import RSSParser
from src.rss.models import Feed, Entry

class TestRSSParser(unittest.TestCase):
    """RSS解析器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.parser = RSSParser()
        self.test_feed_url = "http://example.com/feed.xml"
        
    def test_validate_url(self):
        """测试URL验证"""
        # 有效的URL
        valid_urls = [
            "http://example.com",
            "https://example.com/feed.xml",
            "https://sub.example.com/path/to/feed?param=value"
        ]
        for url in valid_urls:
            self.assertTrue(self.parser.validate_url(url))
            
        # 无效的URL
        invalid_urls = [
            "",
            "not_a_url",
            "ftp://example.com",
            "http:/example.com",
            "http//example.com"
        ]
        for url in invalid_urls:
            self.assertFalse(self.parser.validate_url(url))
            
    def test_clean_html(self):
        """测试HTML清理"""
        # 测试基本HTML清理
        html = "<p>这是<b>一段</b>文本</p>"
        cleaned = self.parser._clean_html(html)
        self.assertEqual(cleaned, "这是一段文本")
        
        # 测试嵌套HTML
        html = "<div><p>第一段</p><p>第二段</p></div>"
        cleaned = self.parser._clean_html(html)
        self.assertEqual(cleaned, "第一段 第二段")
        
        # 测试特殊字符
        html = "&lt;标题&gt; &amp; 内容"
        cleaned = self.parser._clean_html(html)
        self.assertEqual(cleaned, "<标题> & 内容")
        
    # def test_create_summary(self):
    #     """测试摘要创建"""
    #     # 测试普通文本
    #     text = "这是一段测试文本，用于测试摘要生成功能。"
    #     summary = self.parser._create_summary(text, max_length=10)
    #     self.assertEqual(summary, "这是一段...")
        
    #     # 测试长文本
    #     long_text = "这是一段非常长的测试文本。" * 10
    #     summary = self.parser._create_summary(long_text, max_length=20)
    #     self.assertTrue(len(summary) <= 23)  # 20 + "..."
    #     self.assertTrue(summary.endswith("..."))
        
    #     # 测试HTML文本
    #     html_text = "<p>这是<b>HTML</b>文本</p>"
    #     summary = self.parser._create_summary(html_text, max_length=15)
    #     self.assertNotIn("<", summary)
    #     self.assertNotIn(">", summary)
        
    def test_create_summary_edge_cases(self):
        """测试创建摘要的边界情况"""
        # 测试空字符串
        empty_text = ""
        summary = self.parser._create_summary(empty_text)
        self.assertEqual(summary, "")
        
        # 测试只有空白字符的字符串
        whitespace_only = "   \t\n  "
        summary = self.parser._create_summary(whitespace_only)
        self.assertEqual(summary, "")
        
        # 测试只有HTML标签的字符串
        html_only = "<div><p><span></span></p></div>"
        summary = self.parser._create_summary(html_only)
        self.assertEqual(summary, "")
        
        # 测试max_length为0的情况
        text = "这是测试文本"
        summary = self.parser._create_summary(text, max_length=0)
        self.assertEqual(summary, "")
        
        # 测试max_length为1的情况
        summary = self.parser._create_summary(text, max_length=1)
        self.assertEqual(summary, "这")
        
        # 测试max_length为负数的情况
        summary = self.parser._create_summary(text, max_length=-1)
        self.assertEqual(summary, "")
        
    def test_parse_date(self):
        """测试日期解析"""
        # 测试常见的日期格式
        date_strings = [
            "2024-03-09T12:00:00Z",
            "Sat, 09 Mar 2024 12:00:00 GMT",
            "2024-03-09 12:00:00",
            "09/03/2024 12:00:00"
        ]
        for date_str in date_strings:
            parsed = self.parser._parse_date(date_str)
            self.assertIsInstance(parsed, datetime)
            
        # 测试无效的日期
        invalid_dates = [
            "",
            "not a date",
            "2024/13/45"
        ]
        for date_str in invalid_dates:
            parsed = self.parser._parse_date(date_str)
            self.assertEqual(parsed, datetime.now().replace(microsecond=0))
            
    def test_normalize_url(self):
        """测试URL规范化"""
        base_url = "http://example.com/blog"
        
        # 测试相对URL
        relative_urls = {
            "/article/1": "http://example.com/article/1",
            "article/1": "http://example.com/blog/article/1",
            "../article/1": "http://example.com/article/1",
            "//cdn.example.com/img.jpg": "http://cdn.example.com/img.jpg"
        }
        
        for rel_url, expected in relative_urls.items():
            normalized = self.parser._normalize_url(rel_url, base_url)
            self.assertEqual(normalized, expected)
            
        # 测试绝对URL
        absolute_urls = [
            "http://example.com/article/1",
            "https://example.com/article/1"
        ]
        for abs_url in absolute_urls:
            normalized = self.parser._normalize_url(abs_url, base_url)
            self.assertEqual(normalized, abs_url)

if __name__ == '__main__':
    unittest.main() 