import sys
import os
import unittest
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 先导入基础设施
from src.utils.event_system import event_system
from src.utils.dependency_container import container

# 然后导入组件
from src.rag.core.component import Component
from src.rag.core.config_manager import ConfigManager
from src.rag.core.text_processor import TextProcessor
from src.rag.core.embedding_manager import EmbeddingManager
from src.rag.core.llm_manager import LLMManager
from src.rag.core.retrieval_manager import RetrievalManager
from src.rag.core.rag_manager import RAGManager
from src.rag.core.system_manager import SystemManager

class TestModularArchitecture(unittest.TestCase):
    """测试模块化架构"""
    
    def setUp(self):
        """测试前准备"""
        # 清理依赖容器和事件系统
        container.clear()
        event_system.clear()
        
        # 创建测试目录
        os.makedirs("tests/data/test_modular", exist_ok=True)
        
        # 基本配置
        self.config = {
            "base_dir": "tests/data/test_modular",
            "device": "cpu",
            "llm_type": "tiny",
            "llm_model_id": "models/tiny_llm_sft_92m",
            "embedding_model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "chunk_size": 100,
            "chunk_overlap": 20,
            "top_k": 2
        }
    
    def test_config_manager(self):
        """测试配置管理器"""
        # 创建配置管理器
        config_manager = ConfigManager("tests/data/test_modular/config.json")
        config_manager.initialize()
        
        # 测试配置获取
        config_manager.set("device", "cpu")
        self.assertEqual(config_manager.get("device"), "cpu")
        
        # 测试配置设置
        config_manager.set("device", "cuda")
        self.assertEqual(config_manager.get("device"), "cuda")
        
        # 测试配置更新
        config_manager.update_config({"device": "cpu", "llm_type": "openai"})
        self.assertEqual(config_manager.get("device"), "cpu")
        self.assertEqual(config_manager.get("llm_type"), "openai")
    
    def test_event_system(self):
        """测试事件系统"""
        # 创建测试事件处理器
        received_events = []
        
        def on_test_event(data: str):
            received_events.append(data)
        
        # 订阅事件
        event_system.subscribe("test_event", on_test_event)
        
        # 发布事件
        event_system.publish("test_event", data="test_data")
        
        # 验证事件处理
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0], "test_data")
        
        # 取消订阅
        event_system.unsubscribe("test_event", on_test_event)
        
        # 再次发布事件
        event_system.publish("test_event", data="test_data_2")
        
        # 验证事件未处理
        self.assertEqual(len(received_events), 1)
    
    def test_dependency_container(self):
        """测试依赖容器"""
        # 注册实例
        test_instance = {"key": "value"}
        container.register("test_instance", test_instance)
        
        # 获取实例
        retrieved_instance = container.get("test_instance")
        self.assertEqual(retrieved_instance, test_instance)
        
        # 注册工厂
        def test_factory():
            return {"factory_key": "factory_value"}
        
        container.register_factory("test_factory", test_factory)
        
        # 获取工厂创建的实例
        factory_instance = container.get("test_factory")
        self.assertEqual(factory_instance["factory_key"], "factory_value")
    
    def test_component_initialization(self):
        """测试组件初始化"""
        # 创建配置管理器
        config_manager = ConfigManager("tests/data/test_modular/config.json")
        config_manager.initialize()
        
        # 创建文本处理器
        text_processor = TextProcessor(self.config)
        text_processor.initialize()
        
        # 验证组件注册
        self.assertIsNotNone(container.get_or_default("text_processor"))
        
        # 测试文本分割
        chunks = text_processor.split_text("这是一个测试文本，用于测试文本分割功能。这是第二个句子。")
        self.assertTrue(len(chunks) > 0)
    
    def test_system_manager(self):
        """测试系统管理器"""
        # 创建系统管理器
        system = SystemManager("tests/data/test_modular/config.json")
        
        # 初始化系统
        system.initialize()
        
        # 验证组件注册
        self.assertIsNotNone(container.get_or_default("config_manager"))
        self.assertIsNotNone(container.get_or_default("text_processor"))
        self.assertIsNotNone(container.get_or_default("embedding_manager"))
        self.assertIsNotNone(container.get_or_default("llm_manager"))
        self.assertIsNotNone(container.get_or_default("retrieval_manager"))
        self.assertIsNotNone(container.get_or_default("rag_manager"))
        
        # 更新配置
        system.update_config({"device": "cuda"})
        config_manager = container.get("config_manager")
        self.assertEqual(config_manager.get("device"), "cuda")
        
        # 关闭系统
        system.shutdown()
        
        # 验证清理
        with self.assertRaises(KeyError):
            container.get("config_manager")
    
    def test_config_event_propagation(self):
        """测试配置事件传播"""
        # 创建系统管理器
        system = SystemManager("tests/data/test_modular/config.json")
        system.initialize()
        
        # 记录配置更新事件
        config_updates = []
        
        def on_config_updated(config: Dict[str, Any]):
            config_updates.append(config.get("device"))
        
        # 订阅配置更新事件
        event_system.subscribe("config_updated", on_config_updated)
        
        # 更新配置
        system.update_config({"device": "cuda"})
        
        # 验证事件传播
        self.assertEqual(len(config_updates), 1)
        self.assertEqual(config_updates[0], "cuda")
        
        # 清理
        system.shutdown()
    
    def tearDown(self):
        """测试后清理"""
        # 清理依赖容器和事件系统
        container.clear()
        event_system.clear()

if __name__ == "__main__":
    unittest.main() 