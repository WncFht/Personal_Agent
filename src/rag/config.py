from dataclasses import dataclass
from typing import Dict, Optional, Literal

@dataclass
class RAGConfig:
    """RAG系统配置类"""
    
    # 基础配置
    base_dir: str = "data/rag_db"
    device: str = "cuda"
    
    # 模型配置
    # llm_model_id: str = "models/chatglm3-6b"  # 或其他支持的模型
    llm_model_id: str = "models/tiny_llm_sft_92m"
    embedding_model_id: str = "models/bge-base-zh-v1.5"
    reranker_model_id: str = "models/bge-reranker-base"
    
    # LLM配置
    llm_type: Literal["tiny", "openai", "huggingface", "deepseek"] = "deepseek"  # LLM类型
    system_prompt: str = "你是一个有用的AI助手。"  # 系统提示词
    
    # DeepSeek配置
    deepseek_api_key: str = ""  # DeepSeek API密钥
    deepseek_model: str = "deepseek-chat"  # DeepSeek模型名称
    deepseek_base_url: str = "https://api.deepseek.com"  # DeepSeek API基础URL
    
    # OpenAI配置
    openai_api_key: str = ""  # OpenAI API密钥
    openai_model: str = "gpt-3.5-turbo"  # OpenAI模型名称
    
    # 文本分割配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    use_model_for_splitting: bool = False  # 是否使用模型进行分句
    sentence_splitter_model: str = "damo/nlp_bert_document-segmentation_chinese-base"
    
    # 检索配置
    top_k: int = 3
    search_weights: Optional[Dict[str, float]] = None  # 混合检索权重
    use_reranker: bool = True # 是否使用重排序
    
    # 查询增强配置
    use_query_enhancement: bool = False # 是否使用查询增强
    
    # 缓存配置
    use_cache: bool = True
    cache_dir: str = "data/cache"
    cache_size: int = 1000
    cache_ttl: int = 3600  # 缓存过期时间（秒）
    
    # RSS特定配置
    max_history_days: int = 30  # RSS条目最大保留天数
    update_interval: int = 3600  # RSS更新间隔（秒）
    
    # 并行处理配置
    use_parallel_processing: bool = True  # 是否使用并行处理
    max_workers: Optional[int] = None  # 最大工作线程数，None表示使用CPU核心数 