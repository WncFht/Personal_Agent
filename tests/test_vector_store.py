"""
测试向量存储功能
"""
import pytest
import numpy as np
import os
import shutil
from src.rag.retrieval import VectorStore

@pytest.fixture
def vector_store():
    """创建向量存储实例"""
    return VectorStore(embedding_dim=768)  # BGE模型的维度是768

@pytest.fixture
def test_vectors():
    """创建测试向量数据"""
    # 创建10个随机向量
    return np.random.randn(10, 768).astype(np.float32)

@pytest.fixture
def test_metadata():
    """创建测试元数据"""
    return [
        {
            "id": i,
            "source": f"test{i}",
            "timestamp": f"2024-03-09T12:{i:02d}:00"
        }
        for i in range(1, 11)  # 创建10个元数据项，与test_vectors对应
    ]

@pytest.fixture
def cleanup_test_dir():
    """清理测试目录"""
    test_dir = "tests/data/vector_store"
    yield test_dir
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_vector_store_init():
    """测试向量存储初始化"""
    store = VectorStore(embedding_dim=768)
    assert store.index is not None
    assert len(store.metadata) == 0

def test_add_vectors(vector_store, test_vectors, test_metadata):
    """测试添加向量"""
    vector_store.add(test_vectors[:5], test_metadata[:5])
    assert vector_store.index.ntotal == 5
    assert len(vector_store.metadata) == 5

def test_search_vectors(vector_store, test_vectors, test_metadata):
    """测试向量搜索"""
    # 添加向量
    vector_store.add(test_vectors, test_metadata)
    
    # 使用第一个向量作为查询向量
    query_vector = test_vectors[0]
    distances, indices, metadata = vector_store.search(query_vector, k=3)
    
    assert len(distances) == 3
    assert len(indices) == 3
    assert len(metadata) == 3
    assert indices[0] == 0  # 第一个结果应该是查询向量本身
    assert metadata[0] == test_metadata[0]

def test_save_and_load(vector_store, test_vectors, test_metadata, cleanup_test_dir):
    """测试保存和加载"""
    # 添加向量
    vector_store.add(test_vectors, test_metadata)
    
    # 保存
    vector_store.save(cleanup_test_dir)
    
    # 加载
    loaded_store = VectorStore.load(cleanup_test_dir, embedding_dim=768)
    
    # 验证加载的数据
    assert loaded_store.index.ntotal == len(test_vectors)
    assert len(loaded_store.metadata) == len(test_metadata)
    
    # 验证搜索结果一致性
    query_vector = test_vectors[0]
    original_results = vector_store.search(query_vector, k=3)
    loaded_results = loaded_store.search(query_vector, k=3)
    
    assert np.allclose(original_results[0], loaded_results[0])  # 比较距离
    assert np.array_equal(original_results[1], loaded_results[1])  # 比较索引
    assert original_results[2] == loaded_results[2]  # 比较元数据

def test_invalid_add(vector_store):
    """测试无效的添加操作"""
    vectors = np.random.randn(3, 768).astype(np.float32)
    metadata = [{"id": 1}, {"id": 2}]  # 长度不匹配
    
    # 测试向量和元数据数量不匹配
    with pytest.raises(ValueError):
        vector_store.add(vectors, metadata)
    
    # 测试错误的查询向量维度
    wrong_dim_vector = np.random.randn(100).astype(np.float32)  # 使用100维向量而不是768维
    with pytest.raises(ValueError, match="查询向量维度不匹配"):
        vector_store.search(wrong_dim_vector)

def test_empty_store_search(vector_store):
    """测试空存储的搜索操作"""
    query_vector = np.random.randn(768).astype(np.float32)
    distances, indices, metadata = vector_store.search(query_vector, k=1)  # 使用k=1避免k=0的情况
    
    # 空存储应该返回空结果
    assert len(distances) == 0
    assert len(indices) == 0
    assert len(metadata) == 0 