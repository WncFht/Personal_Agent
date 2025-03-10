import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from loguru import logger

from .component import Component
from ..retrieval.hybrid_retriever import HybridRetriever, Document

class RetrievalManager(Component):
    """检索管理器，负责文档检索"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化检索管理器
        
        Args:
            config: 组件配置
        """
        super().__init__(config)
        self.retriever = None
    
    def initialize(self) -> None:
        """初始化检索管理器"""
        # 获取嵌入模型管理器
        embedding_manager = self.get_dependency("embedding_manager")
        if not embedding_manager:
            logger.error("无法获取嵌入模型管理器，检索管理器初始化失败")
            return
        
        # 创建检索器
        self.retriever = HybridRetriever(
            embedding_model=embedding_manager.embedding_model,
            base_dir=self.config.get("base_dir", "data/rag_db"),
            cache_size=self.config.get("cache_size", 1000),
            cache_ttl=self.config.get("cache_ttl", 3600),
            use_reranker=self.config.get("use_reranker", True),
            reranker_model_path=self.config.get("reranker_model_id", "models/bge-reranker-base"),
            device=self.config.get("device", "cpu")
        )
        
        # 创建必要的目录
        os.makedirs(self.config.get("base_dir", "data/rag_db"), exist_ok=True)
        
        # 注册到依赖容器
        self.container.register("retrieval_manager", self)
        
        # 注册事件处理器
        self.register_event_handlers()
        
        # 加载状态
        self._load_state()
        
        logger.info("检索管理器初始化完成")
    
    def add_document(self, text: str, metadata: Dict[str, Any]) -> None:
        """添加文档
        
        Args:
            text: 文档文本
            metadata: 文档元数据
        """
        self.retriever.add_document(text, metadata)
    
    def search(self, 
              query: str,
              feed_id: Optional[int] = None,
              date_range: Optional[Tuple[datetime, datetime]] = None,
              top_k: Optional[int] = None) -> List[Tuple[str, float, Dict]]:
        """搜索相关内容
        
        Args:
            query: 查询文本
            feed_id: 指定RSS源ID
            date_range: 日期范围(开始时间, 结束时间)
            top_k: 检索结果数量
            
        Returns:
            检索结果列表，每项为(文本, 得分, 元数据)
        """
        logger.info(f"开始搜索: {query}")
        if feed_id:
            logger.info(f"限定 feed_id: {feed_id}")
        if date_range:
            logger.info(f"限定日期范围: {date_range[0]} 到 {date_range[1]}")
            
        # 构建元数据过滤条件
        metadata_filters = {}
        if feed_id is not None:
            metadata_filters['feed_id'] = feed_id
            
        if date_range is not None:
            start_date, end_date = date_range
            metadata_filters['published_date'] = lambda x: start_date <= datetime.fromisoformat(x) <= end_date
            
        # 执行检索
        results = self.retriever.search(
            query=query,
            top_k=top_k or self.config.get("top_k", 3),
            metadata_filters=metadata_filters,
            weights=self.config.get("search_weights")
        )
        
        logger.info(f"找到 {len(results)} 条相关结果")
        for i, (text, score, metadata) in enumerate(results):
            logger.info(f"结果 {i+1}:")
            logger.info(f"  - 来源: {metadata.get('feed_title', '未知')}")
            logger.info(f"  - 标题: {metadata.get('title', '未知')}")
            logger.info(f"  - 相关度得分: {score:.4f}")
            
        return results
    
    def save_state(self) -> None:
        """保存检索器状态"""
        state_file = os.path.join(self.config.get("base_dir", "data/rag_db"), 'retriever_state.json')
        self.retriever.save(state_file)
        logger.info(f"检索器状态已保存到 {state_file}")
    
    def _load_state(self) -> None:
        """加载检索器状态"""
        state_file = os.path.join(self.config.get("base_dir", "data/rag_db"), 'retriever_state.json')
        if os.path.exists(state_file):
            self.retriever.load(state_file)
            logger.info(f"已从 {state_file} 加载检索器状态")
            self._validate_loaded_data()
        else:
            logger.warning("未找到保存的检索器状态")
    
    def _validate_loaded_data(self) -> None:
        """验证数据是否正确加载"""
        doc_count = len(self.retriever.documents)
        emb_count = len(self.retriever.document_embeddings)
        
        if doc_count == 0:
            logger.warning("没有加载任何文档！")
        else:
            logger.info(f"已加载 {doc_count} 个文档")
            
        if doc_count != emb_count:
            logger.error(f"文档数量 ({doc_count}) 与向量数量 ({emb_count}) 不匹配！")
        
        # 检查向量维度
        if emb_count > 0:
            emb_dim = self.retriever.document_embeddings[0].shape[0]
            logger.info(f"向量维度: {emb_dim}")
            
        # 检查元数据
        metadata_fields = set()
        for doc in self.retriever.documents[:min(10, doc_count)]:
            metadata_fields.update(doc.metadata.keys())
        
        logger.info(f"元数据字段: {', '.join(metadata_fields)}")
        
        # 检查BM25参数
        logger.info(f"BM25参数: k1={self.retriever.k1}, b={self.retriever.b}, avgdl={self.retriever.avgdl:.2f}")
        logger.info(f"文档频率词典大小: {len(self.retriever.doc_freqs)}")
    
    def clean_old_entries(self) -> None:
        """清理过期的RSS条目"""
        max_history_days = self.config.get("max_history_days", 30)
        if max_history_days <= 0:
            return
            
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=max_history_days)
        
        # 过滤文档列表
        new_documents = []
        new_embeddings = []
        
        for doc, emb in zip(self.retriever.documents, self.retriever.document_embeddings):
            doc_date = datetime.fromisoformat(doc.metadata['published_date'])
            if doc_date > cutoff_date:
                new_documents.append(doc)
                new_embeddings.append(emb)
                
        # 更新检索器
        self.retriever.documents = new_documents
        self.retriever.document_embeddings = new_embeddings
        self.retriever.initialize_bm25()
        
        # 重新计算BM25参数
        for doc in self.retriever.documents:
            self.retriever._update_bm25_params(doc.text)
            
        logger.info(f"清理了 {max_history_days} 天前的条目")
    
    def register_event_handlers(self) -> None:
        """注册事件处理器"""
        # 监听配置更新事件
        self.event_system.subscribe("config_updated", self._on_config_updated)
        
        # 监听嵌入模型更新事件
        self.event_system.subscribe("embedding_model_updated", self._on_embedding_model_updated)
    
    def _on_config_updated(self, config: Dict[str, Any]) -> None:
        """配置更新事件处理器
        
        Args:
            config: 更新后的配置
        """
        # 检查是否需要更新检索器
        if (config.get("use_reranker") != self.config.get("use_reranker") or
            config.get("reranker_model_id") != self.config.get("reranker_model_id") or
            config.get("device") != self.config.get("device") or
            config.get("cache_size") != self.config.get("cache_size") or
            config.get("cache_ttl") != self.config.get("cache_ttl")):
            
            logger.info("配置已更改，更新检索器参数")
            
            # 更新配置
            self.config.update(config)
            
            # 更新检索器参数
            self.retriever.use_reranker = config.get("use_reranker", True)
            self.retriever.cache_size = config.get("cache_size", 1000)
            self.retriever.cache_ttl = config.get("cache_ttl", 3600)
            
            # 如果重排序器模型或设备发生变化，重新加载重排序器
            if (config.get("reranker_model_id") != self.config.get("reranker_model_id") or
                config.get("device") != self.config.get("device")):
                if self.retriever.use_reranker:
                    self.retriever._load_reranker(
                        config.get("reranker_model_id", "models/bge-reranker-base"),
                        config.get("device", "cpu")
                    )
    
    def _on_embedding_model_updated(self, embedding_model: Any) -> None:
        """嵌入模型更新事件处理器
        
        Args:
            embedding_model: 更新后的嵌入模型
        """
        logger.info("嵌入模型已更新，更新检索器的嵌入模型")
        self.retriever.embedding_model = embedding_model
        
        # 如果有文档，重新计算所有文档的嵌入向量
        if self.retriever.documents:
            logger.info(f"重新计算 {len(self.retriever.documents)} 个文档的嵌入向量")
            self.retriever.document_embeddings = []
            for doc in self.retriever.documents:
                embedding = self.retriever.embedding_model.encode(doc.text)
                self.retriever.document_embeddings.append(embedding)
            
            # 保存更新后的状态
            self.save_state()
    
    def cleanup(self) -> None:
        """清理资源"""
        # 保存状态
        self.save_state() 