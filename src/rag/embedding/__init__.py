"""
Embedding模块
"""
from .base import BaseEmbedding
from .sentence_transformers import SentenceTransformerEmbedding

__all__ = ["BaseEmbedding", "SentenceTransformerEmbedding"] 