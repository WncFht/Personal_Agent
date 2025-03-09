"""
测试embedding功能
"""
import pytest
import numpy as np
from src.rag.embedding import SentenceTransformerEmbedding

@pytest.fixture
def embedding_model():
    """创建embedding模型实例"""
    return SentenceTransformerEmbedding()

def test_embedding_init(embedding_model):
    """测试embedding模型初始化"""
    assert embedding_model.name == "bge_embedding"
    assert embedding_model.model is not None

def test_encode_single_text(embedding_model):
    """测试单个文本编码"""
    text = "这是一个测试文本"
    embedding = embedding_model.encode(text)
    
    print(f"输入文本: {text}")
    print(f"编码维度: {embedding.shape}")
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[1] == 768  # BGE模型输出维度应该是768
    assert len(embedding.shape) == 2  # 应该是2D向量，shape为(1, 768)

def test_encode_multiple_texts(embedding_model):
    """测试多个文本编码"""
    texts = ["第一个测试文本", "第二个测试文本", "第三个测试文本"]
    embeddings = embedding_model.encode(texts)
    
    print(f"输入文本数量: {len(texts)}")
    print(f"编码结果维度: {embeddings.shape}")
    
    assert isinstance(embeddings, np.ndarray)
    assert len(embeddings.shape) == 2  # 应该是二维数组
    assert embeddings.shape[0] == len(texts)  # 第一维应该等于文本数量
    assert embeddings.shape[1] == 768  # 第二维应该是768

def test_normalize_embeddings(embedding_model):
    """测试向量归一化"""
    text = "测试文本"
    embedding = embedding_model.encode(text)
    norm = np.linalg.norm(embedding, axis=1)[0]
    print(f"归一化后的向量范数: {norm}")
    
    # 计算向量的L2范数，应该接近1
    assert np.abs(norm - 1.0) < 1e-6

def test_cosine_similarity(embedding_model):
    """测试余弦相似度计算"""
    text1 = "今天天气很好"
    text2 = "今天是个好天气"
    text3 = "今天股市下跌"
    
    emb1 = embedding_model.encode(text1)  # 保持2D形状
    emb2 = embedding_model.encode(text2)  # 保持2D形状
    emb3 = embedding_model.encode(text3)  # 保持2D形状
    
    # 相似文本的相似度应该更高
    sim12 = embedding_model.cosine_similarity(emb1, emb2)[0, 0]  # 使用索引获取单个值
    sim13 = embedding_model.cosine_similarity(emb1, emb3)[0, 0]  # 使用索引获取单个值
    
    print(f"文本1: {text1}")
    print(f"文本2: {text2}")
    print(f"文本3: {text3}")
    print(f"相似度1-2: {sim12:.4f}")
    print(f"相似度1-3: {sim13:.4f}")
    
    assert sim12 > sim13  # text1和text2的相似度应该大于text1和text3的相似度

def test_batch_encoding_consistency(embedding_model):
    """测试批量编码的一致性"""
    text = "测试文本"
    
    # 单独编码和批量编码都保持2D形状
    emb1 = embedding_model.encode(text)
    emb2 = embedding_model.encode([text])
    
    print(f"单独编码维度: {emb1.shape}")
    print(f"批量编码维度: {emb2.shape}")
    
    # 两种方式的结果应该相同
    assert np.allclose(emb1, emb2, rtol=1e-5) 