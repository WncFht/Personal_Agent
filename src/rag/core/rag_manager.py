import os
from typing import Dict, Any, List, Tuple, Optional, Iterator
from datetime import datetime, timedelta
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .component import Component
from ..templates import ORIGINAL_RAG_PROMPT_TEMPLATE, ENHANCED_RAG_PROMPT_TEMPLATE
from ...rss.models import Entry

class RAGManager(Component):
    """RAG管理器，负责协调各组件完成RAG任务"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化RAG管理器
        
        Args:
            config: 组件配置
        """
        super().__init__(config)
        self.last_processed_timestamp = None
    
    def initialize(self) -> None:
        """初始化RAG管理器"""
        # 加载最后处理的时间戳
        self._load_last_processed_timestamp()
        
        # 注册到依赖容器
        self.container.register("rag_manager", self)
        
        # 注册事件处理器
        self.register_event_handlers()
        
        logger.info("RAG管理器初始化完成")
    
    def _load_last_processed_timestamp(self) -> None:
        """加载最后处理的时间戳"""
        timestamp_file = os.path.join(self.config.get("base_dir", "data/rag_db"), 'last_processed.txt')
        if os.path.exists(timestamp_file):
            with open(timestamp_file, 'r') as f:
                timestamp_str = f.read().strip()
                if timestamp_str:
                    self.last_processed_timestamp = datetime.fromisoformat(timestamp_str)
                    logger.info(f"加载最后处理时间戳: {self.last_processed_timestamp}")

    def _save_last_processed_timestamp(self) -> None:
        """保存最后处理的时间戳"""
        timestamp_file = os.path.join(self.config.get("base_dir", "data/rag_db"), 'last_processed.txt')
        with open(timestamp_file, 'w') as f:
            f.write(datetime.now().isoformat())
        self.last_processed_timestamp = datetime.now()
        logger.info(f"更新最后处理时间戳: {self.last_processed_timestamp}")
    
    def process_entry(self, entry: Entry) -> None:
        """处理单个RSS条目
        
        Args:
            entry: RSS条目
        """
        # 获取文本处理器和检索管理器
        text_processor = self.get_dependency("text_processor")
        retrieval_manager = self.get_dependency("retrieval_manager")
        
        if not text_processor or not retrieval_manager:
            logger.error("无法获取文本处理器或检索管理器，处理条目失败")
            return
        
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
        chunks = text_processor.split_text(text)
        logger.debug(f"条目 '{entry.title}' 被分割为 {len(chunks)} 个块")
        
        # 添加到检索器
        for i, chunk in enumerate(chunks):
            retrieval_manager.add_document(chunk, metadata)
            if i == 0 or (i+1) % 10 == 0:
                logger.debug(f"已处理 {i+1}/{len(chunks)} 个文本块")
    
    def process_entries(self, entries: List[Entry]) -> None:
        """批量处理RSS条目
        
        Args:
            entries: RSS条目列表
        """
        if not entries:
            logger.info("没有条目需要处理")
            return
            
        logger.info(f"开始处理 {len(entries)} 个RSS条目")
        
        # 使用并行处理
        if self.config.get("use_parallel_processing", True):
            self._process_entries_parallel(entries)
        else:
            self._process_entries_sequential(entries)
            
        # 保存检索器状态
        retrieval_manager = self.get_dependency("retrieval_manager")
        if retrieval_manager:
            retrieval_manager.save_state()
        
        # 更新时间戳
        self._save_last_processed_timestamp()
            
        logger.info(f"完成处理 {len(entries)} 个RSS条目")
    
    def _process_entries_sequential(self, entries: List[Entry]) -> None:
        """顺序处理RSS条目"""
        for i, entry in enumerate(entries):
            try:
                logger.debug(f"处理条目 {i+1}/{len(entries)}: {entry.title}")
                self.process_entry(entry)
                if (i + 1) % 10 == 0 or i + 1 == len(entries):
                    logger.info(f"已处理 {i + 1}/{len(entries)} 个条目")
            except Exception as e:
                logger.error(f"处理条目时出错: {e}")
    
    def _process_entries_parallel(self, entries: List[Entry]) -> None:
        """并行处理RSS条目"""
        max_workers = self.config.get("max_workers") or os.cpu_count()
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
    
    def load_from_rss_db(self, db_path: str, days: int = 30, incremental: bool = True) -> None:
        """从RSS数据库加载数据
        
        Args:
            db_path: 数据库文件路径
            days: 加载最近几天的数据
            incremental: 是否增量加载（只加载上次处理后的新数据）
        """
        import sqlite3
        
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
        
        logger.info("RSS数据加载完成")
    
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
        if self.config.get("use_query_enhancement", False):
            return self._answer_with_enhancement(query, feed_id, date_range, top_k)
        else:
            return self._answer_without_enhancement(query, feed_id, date_range, top_k)
    
    def _answer_without_enhancement(self, 
                                  query: str,
                                  feed_id: Optional[int] = None,
                                  date_range: Optional[Tuple[datetime, datetime]] = None,
                                  top_k: Optional[int] = None) -> str:
        """不使用查询增强的回答方法"""
        # 获取检索管理器和LLM管理器
        retrieval_manager = self.get_dependency("retrieval_manager")
        llm_manager = self.get_dependency("llm_manager")
        
        if not retrieval_manager or not llm_manager:
            logger.error("无法获取检索管理器或LLM管理器，回答问题失败")
            return "系统错误，无法回答问题。"
        
        # 搜索相关内容
        search_results = retrieval_manager.search(
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
        answer = llm_manager.generate(prompt)
        
        return answer
    
    def _answer_with_enhancement(self, 
                               query: str,
                               feed_id: Optional[int] = None,
                               date_range: Optional[Tuple[datetime, datetime]] = None,
                               top_k: Optional[int] = None) -> str:
        """使用查询增强的回答方法"""
        # 获取检索管理器和LLM管理器
        retrieval_manager = self.get_dependency("retrieval_manager")
        llm_manager = self.get_dependency("llm_manager")
        
        if not retrieval_manager or not llm_manager:
            logger.error("无法获取检索管理器或LLM管理器，回答问题失败")
            return "系统错误，无法回答问题。"
        
        logger.info("使用查询增强策略")
        
        # 先用LLM生成初步回答
        logger.info("生成初步回答...")
        initial_answer = llm_manager.generate(query)
        logger.info(f"初步回答: {initial_answer[:100]}...")
        
        # 增强查询
        enhanced_query = query + " " + initial_answer + " " + query
        logger.info("使用增强查询进行检索...")
        
        # 搜索相关内容
        search_results = retrieval_manager.search(
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
        final_answer = llm_manager.generate(prompt)
        
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
        if self.config.get("use_query_enhancement", False):
            yield from self._answer_stream_with_enhancement(query, feed_id, date_range, top_k)
        else:
            yield from self._answer_stream_without_enhancement(query, feed_id, date_range, top_k)
    
    def _answer_stream_without_enhancement(self, 
                                         query: str,
                                         feed_id: Optional[int] = None,
                                         date_range: Optional[Tuple[datetime, datetime]] = None,
                                         top_k: Optional[int] = None) -> Iterator[str]:
        """不使用查询增强的流式回答方法"""
        # 获取检索管理器和LLM管理器
        retrieval_manager = self.get_dependency("retrieval_manager")
        llm_manager = self.get_dependency("llm_manager")
        
        if not retrieval_manager or not llm_manager:
            logger.error("无法获取检索管理器或LLM管理器，回答问题失败")
            yield "系统错误，无法回答问题。"
            return
        
        # 搜索相关内容
        search_results = retrieval_manager.search(
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
        for token in llm_manager.generate_stream(prompt):
            yield token
    
    def _answer_stream_with_enhancement(self, 
                                      query: str,
                                      feed_id: Optional[int] = None,
                                      date_range: Optional[Tuple[datetime, datetime]] = None,
                                      top_k: Optional[int] = None) -> Iterator[str]:
        """使用查询增强的流式回答方法"""
        # 获取检索管理器和LLM管理器
        retrieval_manager = self.get_dependency("retrieval_manager")
        llm_manager = self.get_dependency("llm_manager")
        
        if not retrieval_manager or not llm_manager:
            logger.error("无法获取检索管理器或LLM管理器，回答问题失败")
            yield "系统错误，无法回答问题。"
            return
        
        logger.info("使用查询增强策略")
        
        # 先用LLM生成初步回答
        logger.info("生成初步回答...")
        initial_answer = llm_manager.generate(query)
        logger.info(f"初步回答: {initial_answer[:100]}...")
        
        # 增强查询
        enhanced_query = query + " " + initial_answer + " " + query
        logger.info("使用增强查询进行检索...")
        
        # 搜索相关内容
        search_results = retrieval_manager.search(
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
        for token in llm_manager.generate_stream(prompt):
            yield token
    
    def register_event_handlers(self) -> None:
        """注册事件处理器"""
        # 监听配置更新事件
        self.event_system.subscribe("config_updated", self._on_config_updated)
    
    def _on_config_updated(self, config: Dict[str, Any]) -> None:
        """配置更新事件处理器
        
        Args:
            config: 更新后的配置
        """
        # 更新配置
        self.config.update(config)
    
    def cleanup(self) -> None:
        """清理资源"""
        # 保存最后处理的时间戳
        self._save_last_processed_timestamp() 