import os
from typing import Dict, List, Optional, Tuple, Union, Iterator
from datetime import datetime, timedelta
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .config import RAGConfig
from .retrieval.hybrid_retriever import HybridRetriever
from .embedding.embedding_model import EmbeddingModel
from .utils.text_splitter import TextSplitter
from .llm.base_llm import BaseLLM
from .llm.openai_llm import OpenAILLM
from .llm.huggingface_llm import HuggingFaceLLM
from .llm.tiny_llm import TinyLLM
from ..rss.models import Entry

# 原始RAG提示词模板
ORIGINAL_RAG_PROMPT_TEMPLATE = """参考信息：
{context}
---
问题：
{question}
---
请根据上述参考信息回答问题。如果参考信息不足以回答问题，请说明无法回答。回答要简洁、准确，并尽可能基于参考信息。
"""

# 增强RAG提示词模板
ENHANCED_RAG_PROMPT_TEMPLATE = """参考信息：
{context}
---
我的问题或指令：
{question}
---
我的回答：
{answer}
---
请根据上述参考信息回答和我的问题或指令，修正我的回答。前面的参考信息和我的回答可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你修正的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复。
你修正的回答:"""

class RSSRAG:
    """RSS-RAG系统主类"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # 初始化文本分割器
        self.text_splitter = TextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            use_model=config.use_model_for_splitting,
            model_path=config.sentence_splitter_model,
            device=config.device
        )
        
        # 初始化embedding模型
        self.embedding_model = EmbeddingModel(
            model_path=config.embedding_model_id,
            device=config.device
        )
        
        # 初始化检索器
        self.retriever = HybridRetriever(
            embedding_model=self.embedding_model,
            base_dir=config.base_dir,
            cache_size=config.cache_size,
            cache_ttl=config.cache_ttl,
            use_reranker=config.use_reranker,
            reranker_model_path=config.reranker_model_id,
            device=config.device
        )
        
        # 初始化LLM
        self._init_llm(config)
        
        # 创建必要的目录
        os.makedirs(config.base_dir, exist_ok=True)
        os.makedirs(config.cache_dir, exist_ok=True)
        
        # 最后处理的时间戳
        self.last_processed_timestamp = None
        self._load_last_processed_timestamp()
        
        # 选择提示词模板
        self.use_query_enhancement = config.use_query_enhancement
        
        logger.info("RSS-RAG系统初始化成功")
        
    def _load_last_processed_timestamp(self):
        """加载最后处理的时间戳"""
        timestamp_file = os.path.join(self.config.base_dir, 'last_processed.txt')
        if os.path.exists(timestamp_file):
            with open(timestamp_file, 'r') as f:
                timestamp_str = f.read().strip()
                if timestamp_str:
                    self.last_processed_timestamp = datetime.fromisoformat(timestamp_str)
                    logger.info(f"加载最后处理时间戳: {self.last_processed_timestamp}")

    def _save_last_processed_timestamp(self):
        """保存最后处理的时间戳"""
        timestamp_file = os.path.join(self.config.base_dir, 'last_processed.txt')
        with open(timestamp_file, 'w') as f:
            f.write(datetime.now().isoformat())
        self.last_processed_timestamp = datetime.now()
        logger.info(f"更新最后处理时间戳: {self.last_processed_timestamp}")

    def _init_llm(self, config: RAGConfig):
        """初始化LLM模型"""
        model_id = config.llm_model_id
        
        # 根据模型ID选择合适的LLM实现
        if config.use_tiny_llm:
            self.llm = TinyLLM(
                model_path=model_id,
                device=config.device,
                temperature=0.7,
                system_prompt=config.system_prompt
            )
        elif model_id.startswith("gpt-") or "openai" in model_id.lower():
            self.llm = OpenAILLM(
                model_name=model_id,
                temperature=0.7
            )
        else:
            # 默认使用HuggingFace模型
            self.llm = HuggingFaceLLM(
                model_path=model_id,
                device=config.device,
                temperature=0.7
            )
            
        logger.info(f"初始化LLM: {model_id}")
        
    def process_entry(self, entry: Entry):
        """处理单个RSS条目
        
        Args:
            entry: RSS条目
        """
        # 提取文本内容
        text = f"{entry.title}\n{entry.summary}\n{entry.content}"
        
        # 构建元数据
        metadata = {
            'feed_id': entry.feed_id,
            'title': entry.title,
            'link': entry.link,
            'published_date': entry.published_date.isoformat(),
            'author': entry.author
        }
        
        # 文本分块
        chunks = self.text_splitter.split_text(text)
        logger.debug(f"条目 '{entry.title}' 被分割为 {len(chunks)} 个块")
        
        # 添加到检索器
        for i, chunk in enumerate(chunks):
            self.retriever.add_document(chunk, metadata)
            if i == 0 or (i+1) % 10 == 0:
                logger.debug(f"已处理 {i+1}/{len(chunks)} 个文本块")
            
    def process_entries(self, entries: List[Entry]):
        """批量处理RSS条目
        
        Args:
            entries: RSS条目列表
        """
        if not entries:
            logger.info("没有条目需要处理")
            return
            
        logger.info(f"开始处理 {len(entries)} 个RSS条目")
        
        # 使用并行处理
        if self.config.use_parallel_processing:
            self._process_entries_parallel(entries)
        else:
            self._process_entries_sequential(entries)
            
        logger.info(f"完成处理 {len(entries)} 个RSS条目")
    
    def _process_entries_sequential(self, entries: List[Entry]):
        """顺序处理RSS条目"""
        for i, entry in enumerate(entries):
            try:
                logger.debug(f"处理条目 {i+1}/{len(entries)}: {entry.title}")
                self.process_entry(entry)
                if (i + 1) % 10 == 0 or i + 1 == len(entries):
                    logger.info(f"已处理 {i + 1}/{len(entries)} 个条目")
            except Exception as e:
                logger.error(f"处理条目时出错: {e}")
    
    def _process_entries_parallel(self, entries: List[Entry]):
        """并行处理RSS条目"""
        max_workers = self.config.max_workers or os.cpu_count()
        logger.info(f"使用 {max_workers} 个线程并行处理 {len(entries)} 个条目")
        
        processed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_entry, entry): entry for entry in entries}
            
            # 使用tqdm显示进度
            for future in tqdm(as_completed(futures), total=len(entries), desc="处理RSS条目"):
                try:
                    future.result()
                    processed_count += 1
                    if processed_count % 10 == 0 or processed_count == len(entries):
                        logger.info(f"已并行处理 {processed_count}/{len(entries)} 个条目")
                except Exception as e:
                    logger.error(f"处理条目时出错: {e}")
        
        logger.info(f"并行处理完成，共处理 {processed_count}/{len(entries)} 个条目")
    
    def load_from_rss_db(self, db_path: str, days: int = 30, incremental: bool = True):
        """从RSS数据库加载数据
        
        Args:
            db_path: 数据库文件路径
            days: 加载最近几天的数据
            incremental: 是否增量加载（只加载上次处理后的新数据）
        """
        import sqlite3
        from datetime import datetime, timedelta
        
        logger.info(f"开始从 {db_path} 加载RSS数据...")
        cutoff_date = datetime.now() - timedelta(days=days)
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # 获取表结构信息
            cursor.execute("PRAGMA table_info(entries)")
            columns = {row['name'] for row in cursor.fetchall()}
            logger.debug(f"数据库表结构: {columns}")
            
            # 构建查询条件
            conditions = ["e.published_date > ?"]
            params = [cutoff_date.isoformat()]
            
            # 如果是增量加载且有上次处理的时间戳，则只加载新数据
            if incremental and self.last_processed_timestamp:
                conditions.append("e.published_date > ?")
                params.append(self.last_processed_timestamp.isoformat())
                logger.info(f"增量加载 {self.last_processed_timestamp} 之后的数据")
            
            # 查询条目
            query = f'''
                SELECT e.*, f.title as feed_title 
                FROM entries e 
                JOIN feeds f ON e.feed_id = f.id 
                WHERE {' AND '.join(conditions)}
                ORDER BY e.published_date ASC
            '''
            
            cursor.execute(query, params)
            entries = cursor.fetchall()
            logger.info(f"找到 {len(entries)} 条RSS条目")
            
            # 创建Entry对象列表
            entry_objects = []
            for entry_data in entries:
                try:
                    # 安全获取字段，如果不存在则使用默认值
                    entry_dict = dict(entry_data)
                    
                    entry = Entry(
                        id=entry_dict.get('id', 0),
                        feed_id=entry_dict.get('feed_id', 0),
                        title=entry_dict.get('title', ''),
                        link=entry_dict.get('link', ''),
                        published_date=datetime.fromisoformat(entry_dict.get('published_date', datetime.now().isoformat())),
                        author=entry_dict.get('author', ''),
                        summary=entry_dict.get('summary', ''),
                        content=entry_dict.get('content', ''),
                        read_status=bool(entry_dict.get('read', 0))  # 转换为布尔值
                    )
                    entry_objects.append(entry)
                except Exception as e:
                    logger.error(f"处理条目时出错: {e}, 数据: {dict(entry_data)}")
                    continue
            
            # 处理条目（分块和向量化）
            if entry_objects:
                logger.info(f"开始处理 {len(entry_objects)} 条RSS条目...")
                self.process_entries(entry_objects)
                logger.info(f"完成处理 {len(entry_objects)} 条RSS条目")
            else:
                logger.info("没有新的RSS条目需要处理")
        
        except Exception as e:
            logger.error(f"加载RSS数据时出错: {e}")
            raise
        finally:
            conn.close()
        
        # 更新最后处理的时间戳
        self._save_last_processed_timestamp()
        logger.info("RSS数据加载完成")

    def search(self, 
              query: str,
              feed_id: Optional[int] = None,
              date_range: Optional[Tuple[datetime, datetime]] = None,
              top_k: Optional[int] = None) -> List[Tuple[str, float, Dict]]:
        """搜索相关内容"""
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
            top_k=top_k or self.config.top_k,
            metadata_filters=metadata_filters,
            weights=self.config.search_weights
        )
        
        logger.info(f"找到 {len(results)} 条相关结果")
        for i, (text, score, metadata) in enumerate(results):
            logger.info(f"结果 {i+1}:")
            logger.info(f"  - 来源: {metadata.get('feed_title', '未知')}")
            logger.info(f"  - 标题: {metadata.get('title', '未知')}")
            logger.info(f"  - 相关度得分: {score:.4f}")
            
        return results
    
    def answer(self, 
              query: str,
              feed_id: Optional[int] = None,
              date_range: Optional[Tuple[datetime, datetime]] = None,
              top_k: Optional[int] = None) -> str:
        """回答问题
        
        Args:
            query: 问题
            feed_id: 指定RSS源ID
            date_range: 日期范围(开始时间, 结束时间)
            top_k: 检索结果数量
            
        Returns:
            str: 回答
        """
        # 使用查询增强
        if self.use_query_enhancement:
            return self._answer_with_enhancement(query, feed_id, date_range, top_k)
        else:
            return self._answer_without_enhancement(query, feed_id, date_range, top_k)
    
    def _answer_without_enhancement(self, 
                                  query: str,
                                  feed_id: Optional[int] = None,
                                  date_range: Optional[Tuple[datetime, datetime]] = None,
                                  top_k: Optional[int] = None) -> str:
        """不使用查询增强的回答方法"""
        # 搜索相关内容
        search_results = self.search(
            query=query,
            feed_id=feed_id,
            date_range=date_range,
            top_k=top_k
        )
        
        # 如果没有找到相关内容
        if not search_results:
            return "抱歉，没有找到相关信息。"
            
        # 构建上下文
        context_texts = []
        for text, score, metadata in search_results:
            # 添加元数据信息
            source_info = f"来源: {metadata.get('title', '未知')}"
            if 'published_date' in metadata:
                try:
                    date = datetime.fromisoformat(metadata['published_date']).strftime("%Y-%m-%d")
                    source_info += f" ({date})"
                except:
                    pass
                    
            # 添加到上下文
            context_texts.append(f"{source_info}\n{text}")
            
        context = "\n\n".join(context_texts)
        
        # 构建提示词
        prompt = ORIGINAL_RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=query
        )
        
        # 生成回答
        answer = self.llm.generate(prompt)
        
        return answer
    
    def _answer_with_enhancement(self, 
                               query: str,
                               feed_id: Optional[int] = None,
                               date_range: Optional[Tuple[datetime, datetime]] = None,
                               top_k: Optional[int] = None) -> str:
        """使用查询增强的回答方法"""
        logger.info("使用查询增强策略")
        
        # 先用LLM生成初步回答
        logger.info("生成初步回答...")
        initial_answer = self.llm.generate(query)
        logger.info(f"初步回答: {initial_answer[:100]}...")
        
        # 增强查询
        enhanced_query = query + " " + initial_answer + " " + query
        logger.info("使用增强查询进行检索...")
        
        # 搜索相关内容
        search_results = self.search(
            query=enhanced_query,
            feed_id=feed_id,
            date_range=date_range,
            top_k=top_k
        )
        
        # 如果没有找到相关内容
        if not search_results:
            return initial_answer  # 返回初步回答
            
        # 构建上下文
        context_texts = []
        for text, score, metadata in search_results:
            # 添加元数据信息
            source_info = f"来源: {metadata.get('title', '未知')}"
            if 'published_date' in metadata:
                try:
                    date = datetime.fromisoformat(metadata['published_date']).strftime("%Y-%m-%d")
                    source_info += f" ({date})"
                except:
                    pass
                    
            # 添加到上下文
            context_texts.append(f"{source_info}\n{text}")
            
        context = "\n\n".join(context_texts)
        
        # 构建增强提示词
        prompt = ENHANCED_RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=query,
            answer=initial_answer
        )
        
        # 生成最终回答
        logger.info("生成最终回答...")
        final_answer = self.llm.generate(prompt)
        
        return final_answer
        
    def answer_stream(self, 
                     query: str,
                     feed_id: Optional[int] = None,
                     date_range: Optional[Tuple[datetime, datetime]] = None,
                     top_k: Optional[int] = None) -> Iterator[str]:
        """流式回答问题
        
        Args:
            query: 问题
            feed_id: 指定RSS源ID
            date_range: 日期范围(开始时间, 结束时间)
            top_k: 检索结果数量
            
        Returns:
            Iterator[str]: 流式回答
        """
        # 使用查询增强
        if self.use_query_enhancement:
            yield from self._answer_stream_with_enhancement(query, feed_id, date_range, top_k)
        else:
            yield from self._answer_stream_without_enhancement(query, feed_id, date_range, top_k)
    
    def _answer_stream_without_enhancement(self, 
                                         query: str,
                                         feed_id: Optional[int] = None,
                                         date_range: Optional[Tuple[datetime, datetime]] = None,
                                         top_k: Optional[int] = None) -> Iterator[str]:
        """不使用查询增强的流式回答方法"""
        # 搜索相关内容
        search_results = self.search(
            query=query,
            feed_id=feed_id,
            date_range=date_range,
            top_k=top_k
        )
        
        # 如果没有找到相关内容
        if not search_results:
            yield "抱歉，没有找到相关信息。"
            return
            
        # 构建上下文
        context_texts = []
        for text, score, metadata in search_results:
            # 添加元数据信息
            source_info = f"来源: {metadata.get('title', '未知')}"
            if 'published_date' in metadata:
                try:
                    date = datetime.fromisoformat(metadata['published_date']).strftime("%Y-%m-%d")
                    source_info += f" ({date})"
                except:
                    pass
                    
            # 添加到上下文
            context_texts.append(f"{source_info}\n{text}")
            
        context = "\n\n".join(context_texts)
        
        # 构建提示词
        prompt = ORIGINAL_RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=query
        )
        
        # 流式生成回答
        for token in self.llm.generate_stream(prompt):
            yield token
    
    def _answer_stream_with_enhancement(self, 
                                      query: str,
                                      feed_id: Optional[int] = None,
                                      date_range: Optional[Tuple[datetime, datetime]] = None,
                                      top_k: Optional[int] = None) -> Iterator[str]:
        """使用查询增强的流式回答方法"""
        logger.info("使用查询增强策略")
        
        # 先用LLM生成初步回答
        logger.info("生成初步回答...")
        initial_answer = self.llm.generate(query)
        logger.info(f"初步回答: {initial_answer[:100]}...")
        
        # 增强查询
        enhanced_query = query + " " + initial_answer + " " + query
        logger.info("使用增强查询进行检索...")
        
        # 搜索相关内容
        search_results = self.search(
            query=enhanced_query,
            feed_id=feed_id,
            date_range=date_range,
            top_k=top_k
        )
        
        # 如果没有找到相关内容
        if not search_results:
            yield initial_answer  # 返回初步回答
            return
            
        # 构建上下文
        context_texts = []
        for text, score, metadata in search_results:
            # 添加元数据信息
            source_info = f"来源: {metadata.get('title', '未知')}"
            if 'published_date' in metadata:
                try:
                    date = datetime.fromisoformat(metadata['published_date']).strftime("%Y-%m-%d")
                    source_info += f" ({date})"
                except:
                    pass
                    
            # 添加到上下文
            context_texts.append(f"{source_info}\n{text}")
            
        context = "\n\n".join(context_texts)
        
        # 构建增强提示词
        prompt = ENHANCED_RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=query,
            answer=initial_answer
        )
        
        # 流式生成最终回答
        logger.info("生成最终回答...")
        for token in self.llm.generate_stream(prompt):
            yield token
    
    def clean_old_entries(self):
        """清理过期的RSS条目"""
        if self.config.max_history_days <= 0:
            return
            
        cutoff_date = datetime.now() - timedelta(days=self.config.max_history_days)
        
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
            
        logger.info(f"清理了 {self.config.max_history_days} 天前的条目")
        
    def save_state(self):
        """保存系统状态"""
        # 保存检索器状态
        state_file = os.path.join(self.config.base_dir, 'retriever_state.json')
        self.retriever.save(state_file)
        # 保存时间戳
        self._save_last_processed_timestamp()
        logger.info(f"系统状态已保存到 {state_file}")
        
    def load_state(self, db_path: Optional[str] = None, days: int = 30):
        """加载系统状态
        
        Args:
            db_path: RSS数据库路径，如果提供则在没有找到保存状态时从数据库加载
            days: 加载最近几天的数据
        """
        state_file = os.path.join(self.config.base_dir, 'retriever_state.json')
        if os.path.exists(state_file):
            self.retriever.load(state_file)
            logger.info(f"已从 {state_file} 加载系统状态")
            self._validate_loaded_data()
        else:
            logger.warning("未找到保存的系统状态")
            if db_path:
                logger.info("从 RSS 数据库加载数据...")
                self.load_from_rss_db(db_path, days)
                # 保存新的状态
                self.save_state()
                logger.info("系统状态已更新并保存")
                self._validate_loaded_data()
            else:
                logger.warning("未提供 RSS 数据库路径，无法加载数据")
    
    def _validate_loaded_data(self):
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
        
        return doc_count > 0 