"""
测试文本分块功能
"""
import pytest
from src.rag.utils import TextSplitter

def test_text_splitter_init():
    """测试TextSplitter初始化"""
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    assert splitter.chunk_size == 100
    assert splitter.chunk_overlap == 20
    assert splitter.separator == "\n"
    assert splitter.keep_separator is False

def test_split_single_text():
    """测试单个文本分块"""
    text = "这是一个测试文本。" * 10  # 重复10次以确保超过chunk_size
    splitter = TextSplitter(chunk_size=20, chunk_overlap=5)  # 减小chunk_size以确保分块
    chunks = splitter.split_text(text)
    
    print(f"原始文本长度: {len(text)}")
    print(f"分块数量: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"块{i+1}长度: {len(chunk)}")
    
    assert len(chunks) > 1  # 应该被分成多个块
    for chunk in chunks:
        assert len(chunk) <= 20  # 每个块不应超过chunk_size

def test_split_with_separator():
    """测试使用分隔符分块"""
    text = "第一段\n第二段\n第三段\n第四段"
    splitter = TextSplitter(chunk_size=100, separator="\n", keep_separator=True)
    chunks = splitter.split_text(text)
    
    assert len(chunks) == 1  # 文本总长度未超过chunk_size，应该只有一个块
    assert chunks[0].count("\n") == 3  # 应该保留所有分隔符

def test_split_documents(test_texts):
    """测试多文档分块"""
    # 创建更长的测试文本
    long_texts = [
        text + "。" + text  # 将每个文本加倍
        for text in test_texts
    ]
    
    splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
    chunks = splitter.split_documents(long_texts)
    
    print(f"原始文档数量: {len(long_texts)}")
    print(f"分块后数量: {len(chunks)}")
    for i, text in enumerate(long_texts):
        print(f"文档{i+1}长度: {len(text)}")
    for i, chunk in enumerate(chunks):
        print(f"块{i+1}长度: {len(chunk)}")
    
    assert len(chunks) > len(long_texts)  # 应该产生更多的块
    for chunk in chunks:
        assert len(chunk) <= 50  # 每个块不应超过chunk_size

def test_overlap():
    """测试块之间的重叠"""
    text = "一二三四五六七八九十" * 10  # 100个字符
    splitter = TextSplitter(chunk_size=30, chunk_overlap=10)
    chunks = splitter.split_text(text)
    
    print(f"原始文本: {text[:50]}...")
    print(f"分块数量: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"块{i+1}: {chunk}")
    
    # 检查相邻块之间是否有重叠
    for i in range(len(chunks) - 1):
        chunk1_end = chunks[i][-10:]  # 取第一个块的最后10个字符
        chunk2_start = chunks[i + 1][:10]  # 取第二个块的前10个字符
        print(f"块{i+1}结尾: {chunk1_end}")
        print(f"块{i+2}开头: {chunk2_start}")
        assert chunk1_end == chunk2_start  # 应该相等 