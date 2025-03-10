import sys
import os
import unittest
import shutil
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.config import RAGConfig
from src.rag.rss_rag import RSSRAG
from src.rss.models import Entry

class MockEntry:
    """模拟RSS条目"""
    def __init__(self, feed_id, title, content, published_date=None):
        self.feed_id = feed_id
        self.title = title
        self.link = f"https://example.com/{feed_id}/{title.lower().replace(' ', '-')}"
        self.published_date = published_date or datetime.now()
        self.author = "Test Author"
        self.summary = f"Summary of {title}"
        self.content = content
        self.read_status = False

class TestRAG(unittest.TestCase):
    """测试RAG功能"""
    
    def setUp(self):
        """测试前准备"""
        # 清理测试数据库目录
        test_db_dir = "tests/data/test_rag_db"
        if os.path.exists(test_db_dir):
            shutil.rmtree(test_db_dir)
        os.makedirs(test_db_dir, exist_ok=True)
        
        # 创建测试配置
        self.config = RAGConfig(
            base_dir="tests/data/test_rag_db",
            device="cpu",
            embedding_model_id="models/bge-base-zh-v1.5",
            llm_type="openai",
            llm_model_id="gpt-3.5-turbo",  # 使用环境变量中的API密钥
            chunk_size=100,
            chunk_overlap=20,
            top_k=2
        )
        
        # 创建RSSRAG实例
        self.rag = RSSRAG(self.config)
        
        # 创建测试数据
        self.test_entries = [
            MockEntry(
                feed_id=1,
                title="Python编程技巧",
                content="Python是一种流行的编程语言。它具有简洁的语法和丰富的库。使用Python可以快速开发应用程序。"
            ),
            MockEntry(
                feed_id=2,
                title="人工智能最新进展",
                content="近年来，人工智能技术取得了显著进步。深度学习模型在图像识别和自然语言处理方面表现出色。"
            ),
            MockEntry(
                feed_id=1,
                title="数据科学入门",
                content="数据科学结合了统计学、编程和领域知识。Python是数据科学中最常用的编程语言之一。"
            )
        ]
        
        # 处理测试数据
        for entry in self.test_entries:
            self.rag.process_entry(entry)
            
    def test_search(self):
        """测试搜索功能"""
        try:
            # 处理测试条目
            for entry in self.test_entries:
                self.rag.process_entry(entry)
                
            # 测试基本搜索
            results = self.rag.search("Python编程")
            
            # 验证结果
            self.assertIsNotNone(results)
            self.assertGreater(len(results), 0)
            
            # 验证结果格式
            for text, score, metadata in results:
                self.assertIsInstance(text, str)
                self.assertIsInstance(score, float)
                self.assertIsInstance(metadata, dict)
                
            # 测试带过滤条件的搜索
            filtered_results = self.rag.search("Python编程", feed_id=1)
            self.assertGreaterEqual(len(filtered_results), 0)
            
        except Exception as e:
            self.fail(f"测试搜索功能失败: {str(e)}")
    
    def test_answer(self):
        """测试回答功能"""
        try:
            # 处理测试条目
            for entry in self.test_entries:
                self.rag.process_entry(entry)
                
            # 测试回答
            answer = self.rag.answer("Python有什么特点？")
            
            # 验证结果
            self.assertIsNotNone(answer)
            self.assertIsInstance(answer, str)
            self.assertGreater(len(answer), 0)
            
            # 测试带过滤条件的回答
            filtered_answer = self.rag.answer("Python有什么特点？", feed_id=1)
            self.assertIsNotNone(filtered_answer)
            self.assertIsInstance(filtered_answer, str)
            
        except Exception as e:
            self.fail(f"测试回答功能失败: {str(e)}")
                
    def tearDown(self):
        """测试后清理"""
        try:
            # 关闭系统
            if hasattr(self, 'rag') and self.rag is not None:
                self.rag.system_manager.shutdown()
                
            # 清理测试数据库目录
            test_db_dir = "tests/data/test_rag_db"
            if os.path.exists(test_db_dir):
                shutil.rmtree(test_db_dir)
        except Exception as e:
            print(f"清理资源时出错: {str(e)}")
        
if __name__ == "__main__":
    unittest.main() 