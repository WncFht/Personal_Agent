#!/usr/bin/env python
"""
RSS-RAG Gradio用户界面
提供友好的Web界面，用于RSS源管理和智能问答
"""
import os
import sys
import json
import gradio as gr
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from loguru import logger

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rss.storage import RSSStorage
from src.rss.parser import RSSParser
from src.rss.opml_parser import OPMLParser
from src.rag.config import RAGConfig
from src.rag.rss_rag import RSSRAG

# 默认配置
DEFAULT_CONFIG = {
    "base_dir": "data/rag_db",
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu",
    "llm_type": "tiny",
    "llm_model_id": "models/tiny_llm_sft_92m",
    "embedding_model_id": "models/bge-base-zh-v1.5",
    "system_prompt": "你是一个有用的AI助手，擅长回答关于科技和人工智能的问题。",
    "chunk_size": 800,
    "chunk_overlap": 100,
    "top_k": 5,
    "use_reranker": True,
    "use_query_enhancement": True
}

# 全局变量
db_path = "data/rss.db"
config_path = "config/app_config.json"
rag_system = None
storage = None

def load_config() -> Dict:
    """加载配置"""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
    return DEFAULT_CONFIG

def save_config(config: Dict):
    """保存配置"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def init_system(config: Dict = None):
    """初始化系统"""
    global rag_system, storage
    
    if config is None:
        config = load_config()
    
    # 创建RAG配置
    rag_config = RAGConfig(**config)
    
    # 初始化存储
    storage = RSSStorage(db_path)
    
    # 初始化RAG系统
    rag_system = RSSRAG(rag_config)
    
    # 加载状态
    try:
        rag_system.load_state(db_path=db_path)
        logger.info("RAG系统已加载")
    except Exception as e:
        logger.error(f"加载RAG系统状态失败: {e}")
        logger.info("正在重新构建索引...")
        rag_system.load_from_rss_db(db_path, incremental=False)
        rag_system.save_state()

def answer_question(question: str, feed_id: Optional[int] = None, days: Optional[int] = None, 
                   show_references: bool = True, stream: bool = True) -> str:
    """回答问题"""
    global rag_system
    
    if rag_system is None:
        init_system()
    
    # 构建日期范围
    date_range = None
    if days:
        start_date = datetime.now() - timedelta(days=int(days))
        date_range = (start_date, datetime.now())
    
    # 获取feed_id
    feed_id_int = None
    if feed_id and feed_id.isdigit():
        feed_id_int = int(feed_id)
    
    # 获取检索结果
    search_results = rag_system.search(
        query=question,
        feed_id=feed_id_int,
        date_range=date_range
    )
    
    # 格式化参考信息
    references = ""
    if show_references and search_results:
        references = "### 参考信息:\n\n"
        for i, result in enumerate(search_results):
            doc, score = result
            metadata = doc.metadata
            title = metadata.get('title', '未知标题')
            source = metadata.get('feed_title', '未知来源')
            date = metadata.get('published_date', '未知日期')
            link = metadata.get('link', '#')
            
            references += f"**{i+1}. [{title}]({link})** (来源: {source}, 日期: {date})\n\n"
            references += f"{doc.page_content[:200]}...\n\n"
    
    # 生成回答
    if stream:
        answer = ""
        for token in rag_system.answer_stream(
            query=question,
            feed_id=feed_id_int,
            date_range=date_range
        ):
            answer += token
            yield answer + (f"\n\n{references}" if show_references else "")
    else:
        answer = rag_system.answer(
            query=question,
            feed_id=feed_id_int,
            date_range=date_range
        )
        return answer + (f"\n\n{references}" if show_references else "")

def get_feeds() -> List[Tuple[str, str]]:
    """获取所有RSS源"""
    global storage
    
    if storage is None:
        storage = RSSStorage(db_path)
    
    feeds = storage.get_feeds()
    return [("", "全部")] + [(str(feed.id), feed.title) for feed in feeds]

def add_rss_feed(url: str, category: str = "") -> str:
    """添加RSS源"""
    global storage, rag_system
    
    if storage is None:
        storage = RSSStorage(db_path)
    
    parser = RSSParser()
    
    # 解析RSS源
    feed = parser.parse_feed(url)
    if not feed:
        return f"错误: 无法解析RSS源 {url}"
    
    # 设置分类
    feed.category = category
    
    # 添加到数据库
    feed_id = storage.add_feed(feed)
    if feed_id <= 0:
        return f"错误: 添加RSS源失败 {url}"
    
    # 获取条目
    entries = parser.parse_entries(url, feed_id)
    count = storage.add_entries(entries)
    
    # 更新RAG系统
    if rag_system is not None:
        rag_system.process_entries(entries)
        rag_system.save_state()
    
    return f"成功添加RSS源: {feed.title}\n已添加 {count} 个条目"

def import_opml_file(file_obj) -> str:
    """导入OPML文件"""
    global storage, rag_system
    
    if storage is None:
        storage = RSSStorage(db_path)
    
    if file_obj is None:
        return "错误: 未选择文件"
    
    file_path = file_obj.name
    
    # 解析OPML文件
    parser = OPMLParser()
    feeds = parser.parse_opml(file_path)
    
    if not feeds:
        return f"从OPML文件中未找到RSS源: {file_path}"
    
    # 添加到数据库
    added_count = 0
    skipped_count = 0
    error_count = 0
    total_entries = 0
    
    rss_parser = RSSParser()
    
    for feed in feeds:
        try:
            # 检查是否已存在
            existing_feeds = storage.get_feeds_by_url(feed.url)
            if existing_feeds:
                skipped_count += 1
                continue
                
            # 添加到数据库
            feed_id = storage.add_feed(feed)
            if feed_id > 0:
                added_count += 1
                
                # 获取条目
                try:
                    entries = rss_parser.parse_entries(feed.url, feed_id)
                    count = storage.add_entries(entries)
                    total_entries += count
                    
                    # 更新RAG系统
                    if rag_system is not None:
                        rag_system.process_entries(entries)
                except Exception as e:
                    logger.error(f"获取条目失败: {e}")
            else:
                error_count += 1
        except Exception as e:
            logger.error(f"处理RSS源时出错: {feed.title} - {e}")
            error_count += 1
    
    # 保存RAG状态
    if rag_system is not None:
        rag_system.save_state()
    
    return f"导入完成:\n- 添加了 {added_count} 个RSS源\n- 跳过了 {skipped_count} 个已存在的源\n- 失败了 {error_count} 个源\n- 总计添加了 {total_entries} 个条目"

def delete_feed(feed_id: str) -> str:
    """删除RSS源"""
    global storage, rag_system
    
    if storage is None:
        storage = RSSStorage(db_path)
    
    if not feed_id or not feed_id.isdigit():
        return "请选择要删除的RSS源"
    
    feed_id_int = int(feed_id)
    
    # 获取源信息
    feed = storage.get_feed(feed_id_int)
    if not feed:
        return f"错误: 未找到ID为 {feed_id} 的RSS源"
    
    # 删除源
    success = storage.delete_feed(feed_id_int)
    if not success:
        return f"错误: 删除RSS源失败 {feed.title}"
    
    # 重新构建RAG索引
    if rag_system is not None:
        rag_system.load_from_rss_db(db_path, incremental=False)
        rag_system.save_state()
    
    return f"成功删除RSS源: {feed.title}"

def update_feeds() -> str:
    """更新所有RSS源"""
    global storage, rag_system
    
    if storage is None:
        storage = RSSStorage(db_path)
    
    parser = RSSParser()
    feeds = storage.get_feeds()
    
    if not feeds:
        return "没有RSS源可更新"
    
    total_new_entries = 0
    updated_feeds = 0
    error_feeds = 0
    
    for feed in feeds:
        try:
            # 获取新条目
            entries = parser.parse_entries(feed.url, feed.id)
            count = storage.add_entries(entries)
            
            if count > 0:
                total_new_entries += count
                updated_feeds += 1
                
                # 更新RAG系统
                if rag_system is not None and entries:
                    rag_system.process_entries(entries)
            
            # 更新时间戳
            storage.update_feed_timestamp(feed.id)
        except Exception as e:
            logger.error(f"更新RSS源失败: {feed.title} - {e}")
            error_feeds += 1
    
    # 保存RAG状态
    if rag_system is not None:
        rag_system.save_state()
    
    return f"更新完成:\n- 更新了 {updated_feeds} 个RSS源\n- 添加了 {total_new_entries} 个新条目\n- 失败了 {error_feeds} 个源"

def get_feed_stats():
    """获取RSS源统计信息"""
    global storage
    
    if storage is None:
        storage = RSSStorage(db_path)
    
    feeds = storage.get_feeds()
    
    if not feeds:
        return "没有RSS源", None
    
    # 收集数据
    feed_data = []
    for feed in feeds:
        entry_count = storage.get_entry_count(feed_id=feed.id)
        feed_data.append({
            "id": feed.id,
            "title": feed.title,
            "category": feed.category or "未分类",
            "entries": entry_count,
            "last_updated": feed.last_updated
        })
    
    # 创建DataFrame
    df = pd.DataFrame(feed_data)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 按分类统计
    category_counts = df.groupby('category')['entries'].sum()
    category_counts.plot.pie(ax=ax1, autopct='%1.1f%%', title='按分类统计条目数')
    
    # 按源统计
    top_feeds = df.sort_values('entries', ascending=False).head(10)
    top_feeds.plot.bar(x='title', y='entries', ax=ax2, title='条目数最多的10个源')
    
    plt.tight_layout()
    
    # 生成表格HTML
    df_html = df.to_html(index=False)
    
    return df_html, fig

def update_system_config(config_dict):
    """更新系统配置"""
    global rag_system
    
    # 保存配置
    save_config(config_dict)
    
    # 重新初始化系统
    init_system(config_dict)
    
    return "配置已更新，系统已重新初始化"

def create_ui():
    """创建Gradio界面"""
    with gr.Blocks(title="RSS-RAG 智能问答系统", theme=gr.themes.Soft()) as app:
        gr.Markdown("# RSS-RAG 智能问答系统")
        
        with gr.Tabs() as tabs:
            # 问答标签页
            with gr.TabItem("智能问答"):
                with gr.Row():
                    with gr.Column(scale=3):
                        question_input = gr.Textbox(label="问题", placeholder="请输入您的问题...", lines=3)
                        
                        with gr.Row():
                            feed_dropdown = gr.Dropdown(choices=get_feeds(), label="RSS源", value="")
                            days_input = gr.Number(label="时间范围（天）", value=30)
                        
                        with gr.Row():
                            show_refs_checkbox = gr.Checkbox(label="显示参考信息", value=True)
                            stream_checkbox = gr.Checkbox(label="流式输出", value=True)
                        
                        ask_button = gr.Button("提问", variant="primary")
                    
                    with gr.Column(scale=7):
                        answer_output = gr.Markdown(label="回答")
            
            # RSS源管理标签页
            with gr.TabItem("RSS源管理"):
                with gr.Tabs() as rss_tabs:
                    # 添加RSS源
                    with gr.TabItem("添加RSS源"):
                        with gr.Row():
                            url_input = gr.Textbox(label="RSS源URL", placeholder="https://example.com/feed.xml")
                            category_input = gr.Textbox(label="分类", placeholder="科技、新闻等")
                        
                        add_button = gr.Button("添加", variant="primary")
                        add_result = gr.Textbox(label="结果")
                    
                    # 导入OPML
                    with gr.TabItem("导入OPML"):
                        opml_file = gr.File(label="OPML文件")
                        import_button = gr.Button("导入", variant="primary")
                        import_result = gr.Textbox(label="结果")
                    
                    # 管理RSS源
                    with gr.TabItem("管理RSS源"):
                        with gr.Row():
                            manage_feed_dropdown = gr.Dropdown(choices=get_feeds(), label="选择RSS源", value="")
                            delete_button = gr.Button("删除", variant="stop")
                        
                        update_button = gr.Button("更新所有RSS源", variant="secondary")
                        manage_result = gr.Textbox(label="结果")
                    
                    # RSS源统计
                    with gr.TabItem("RSS源统计"):
                        stats_button = gr.Button("生成统计", variant="secondary")
                        stats_table = gr.HTML(label="RSS源统计表")
                        stats_plot = gr.Plot(label="统计图表")
            
            # 系统配置标签页
            with gr.TabItem("系统配置"):
                config_dict = load_config()
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 基础配置")
                        device_dropdown = gr.Dropdown(
                            choices=["cuda", "cpu"], 
                            label="设备", 
                            value=config_dict.get("device", "cuda")
                        )
                        base_dir_input = gr.Textbox(
                            label="数据目录", 
                            value=config_dict.get("base_dir", "data/rag_db")
                        )
                    
                    with gr.Column():
                        gr.Markdown("### LLM配置")
                        llm_type_dropdown = gr.Dropdown(
                            choices=["tiny", "openai", "huggingface", "deepseek"], 
                            label="LLM类型", 
                            value=config_dict.get("llm_type", "tiny")
                        )
                        llm_model_input = gr.Textbox(
                            label="模型路径/名称", 
                            value=config_dict.get("llm_model_id", "models/tiny_llm_sft_92m")
                        )
                        system_prompt_input = gr.Textbox(
                            label="系统提示词", 
                            value=config_dict.get("system_prompt", "你是一个有用的AI助手。"),
                            lines=3
                        )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### API配置（如果使用API模型）")
                        openai_api_key_input = gr.Textbox(
                            label="OpenAI API密钥", 
                            value=config_dict.get("openai_api_key", ""),
                            type="password"
                        )
                        deepseek_api_key_input = gr.Textbox(
                            label="DeepSeek API密钥", 
                            value=config_dict.get("deepseek_api_key", ""),
                            type="password"
                        )
                    
                    with gr.Column():
                        gr.Markdown("### RAG配置")
                        embedding_model_input = gr.Textbox(
                            label="Embedding模型路径", 
                            value=config_dict.get("embedding_model_id", "models/bge-base-zh-v1.5")
                        )
                        chunk_size_input = gr.Number(
                            label="分块大小", 
                            value=config_dict.get("chunk_size", 800)
                        )
                        chunk_overlap_input = gr.Number(
                            label="分块重叠", 
                            value=config_dict.get("chunk_overlap", 100)
                        )
                        top_k_input = gr.Number(
                            label="检索结果数量", 
                            value=config_dict.get("top_k", 5)
                        )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 高级选项")
                        use_reranker_checkbox = gr.Checkbox(
                            label="使用重排序", 
                            value=config_dict.get("use_reranker", True)
                        )
                        use_query_enhancement_checkbox = gr.Checkbox(
                            label="使用查询增强", 
                            value=config_dict.get("use_query_enhancement", True)
                        )
                        use_cache_checkbox = gr.Checkbox(
                            label="使用缓存", 
                            value=config_dict.get("use_cache", True)
                        )
                
                save_config_button = gr.Button("保存配置", variant="primary")
                config_result = gr.Textbox(label="结果")
        
        # 设置事件处理
        # 问答功能
        ask_button.click(
            fn=answer_question,
            inputs=[question_input, feed_dropdown, days_input, show_refs_checkbox, stream_checkbox],
            outputs=answer_output
        )
        
        # RSS源管理
        add_button.click(
            fn=add_rss_feed,
            inputs=[url_input, category_input],
            outputs=add_result
        )
        
        import_button.click(
            fn=import_opml_file,
            inputs=opml_file,
            outputs=import_result
        )
        
        delete_button.click(
            fn=delete_feed,
            inputs=manage_feed_dropdown,
            outputs=manage_result
        )
        
        update_button.click(
            fn=update_feeds,
            inputs=[],
            outputs=manage_result
        )
        
        stats_button.click(
            fn=get_feed_stats,
            inputs=[],
            outputs=[stats_table, stats_plot]
        )
        
        # 系统配置
        def save_config_fn(device, base_dir, llm_type, llm_model, system_prompt,
                         openai_api_key, deepseek_api_key, embedding_model,
                         chunk_size, chunk_overlap, top_k,
                         use_reranker, use_query_enhancement, use_cache):
            config = {
                "device": device,
                "base_dir": base_dir,
                "llm_type": llm_type,
                "llm_model_id": llm_model,
                "system_prompt": system_prompt,
                "openai_api_key": openai_api_key,
                "deepseek_api_key": deepseek_api_key,
                "embedding_model_id": embedding_model,
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
                "top_k": int(top_k),
                "use_reranker": use_reranker,
                "use_query_enhancement": use_query_enhancement,
                "use_cache": use_cache
            }
            return update_system_config(config)
        
        save_config_button.click(
            fn=save_config_fn,
            inputs=[
                device_dropdown, base_dir_input, llm_type_dropdown, llm_model_input, system_prompt_input,
                openai_api_key_input, deepseek_api_key_input, embedding_model_input,
                chunk_size_input, chunk_overlap_input, top_k_input,
                use_reranker_checkbox, use_query_enhancement_checkbox, use_cache_checkbox
            ],
            outputs=config_result
        )
        
        # 标签页切换事件
        rss_tabs.select(
            fn=lambda: gr.update(choices=get_feeds()),
            inputs=[],
            outputs=manage_feed_dropdown
        )
        
        tabs.select(
            fn=lambda: gr.update(choices=get_feeds()),
            inputs=[],
            outputs=feed_dropdown
        )
    
    return app

def main(host="0.0.0.0", port=7860, share=False):
    """主函数"""
    # 确保数据目录存在
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # 初始化系统
    init_system()
    
    # 创建并启动UI
    app = create_ui()
    app.launch(server_name=host, server_port=port, share=share)

if __name__ == "__main__":
    main() 