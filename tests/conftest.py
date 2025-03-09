"""
测试配置文件
"""
import os
import sys
import pytest
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def test_config():
    """测试配置"""
    return {
        "base_dir": "tests/data",
        "models": {
            "embedding": {
                "model_path": "models/bge-base-zh-v1.5",
                "model_type": "bge"
            }
        },
        "vector_store": {
            "directory": "tests/data/vector_store",
            "index_type": "l2"
        },
        "text_splitter": {
            "chunk_size": 100,
            "chunk_overlap": 20,
            "separator": "\n"
        },
        "retrieval": {
            "top_k": 3
        }
    }

@pytest.fixture
def test_texts():
    """测试文本数据"""
    return [
        "这是第一段测试文本，用于测试文本分块功能。",
        "这是第二段测试文本，包含多个句子。这些句子将被分块处理。每个块都有特定的大小。",
        "这是第三段测试文本，用于测试向量存储和检索功能。我们将测试文本转换为向量，并进行相似度搜索。"
    ]

@pytest.fixture
def test_metadata():
    """测试元数据"""
    return [
        {"id": 1, "source": "test1", "timestamp": "2024-03-09T12:00:00"},
        {"id": 2, "source": "test2", "timestamp": "2024-03-09T12:01:00"},
        {"id": 3, "source": "test3", "timestamp": "2024-03-09T12:02:00"}
    ] 