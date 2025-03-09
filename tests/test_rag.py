import sys
import os
import unittest
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
        # 创建测试配置
        self.config = RAGConfig(
            base_dir="tests/data/test_rag_db",
            device="gpu",
            embedding_model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
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
        # 测试基本搜索
        results = self.rag.search("Python编程")
        self.assertTrue(len(results) > 0)
        
        # 测试按feed_id过滤
        results = self.rag.search("数据科学", feed_id=1)
        self.assertTrue(len(results) > 0)
        for _, _, metadata in results:
            self.assertEqual(metadata['feed_id'], 1)
            
        # 测试按日期过滤
        yesterday = datetime.now() - timedelta(days=1)
        results = self.rag.search("人工智能", date_range=(yesterday, datetime.now()))
        self.assertTrue(len(results) > 0)
        
    def test_answer(self):
        """测试回答功能"""
        # 注意：此测试需要有效的OpenAI API密钥
        try:
            answer = self.rag.answer("Python有什么特点？")
            self.assertTrue(len(answer) > 0)
            print(f"回答: {answer}")
        except Exception as e:
            print(f"测试回答功能失败: {e}")
            # 如果没有API密钥，跳过测试
            if "api_key" in str(e).lower():
                self.skipTest("缺少OpenAI API密钥")
            else:
                raise
                
    def tearDown(self):
        """测试后清理"""
        # 保存状态（可选）
        self.rag.save_state()
        
if __name__ == "__main__":
    unittest.main() 