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

# 设置matplotlib中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except Exception as e:
    print(f"设置matplotlib中文字体失败: {e}")

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rss.storage import RSSStorage
from src.rss.parser import RSSParser
from src.rss.opml_parser import OPMLParser
from src.rag.config import RAGConfig
from src.rag.rss_rag import RSSRAG
from src.rag.core.config_manager import ConfigManager

# 全局变量
db_path = "data/rss.db"
config_manager = None
rag_system = None
storage = None

def init_system():
    """初始化系统"""
    global rag_system, storage, config_manager
    
    # 初始化配置管理器
    config_manager = ConfigManager(
        config_path="config/app_config.json",
        user_config_path="config/user_config.json",
        auto_reload=True
    )
    # 确保初始化配置管理器
    config_manager.initialize()
    
    # 初始化存储
    storage = RSSStorage(db_path)
    
    # 初始化RAG系统
    rag_system = RSSRAG(config_manager=config_manager)
    
    # 加载状态
    try:
        rag_system.load_state(db_path=db_path)
        logger.info("RAG系统已加载")
    except Exception as e:
        logger.error(f"加载RAG系统状态失败: {e}")
        logger.info("正在重新构建索引...")
        rag_system.load_from_rss_db(db_path, incremental=False)
        rag_system.save_state()

def answer_question(question: str, feed_title: str, days: Optional[int] = None, 
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
    
    # 获取所有RSS源
    feed_choices = get_feeds()
    
    # 根据标题查找对应的feed_id
    feed_id = ""
    for id, title in feed_choices:
        if title == feed_title:
            feed_id = id
            break
    
    # 如果feed_id为空字符串，表示选择了"全部"
    feed_id_int = int(feed_id) if feed_id else None
    
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
            # 解包搜索结果元组
            text, score, metadata = result
            title = metadata.get('title', '未知标题')
            source = metadata.get('feed_title', '未知来源')
            date = metadata.get('published_date', '未知日期')
            link = metadata.get('link', '#')
            
            references += f"**{i+1}. [{title}]({link})** (来源: {source}, 日期: {date})\n\n"
            references += f"{text[:200]}...\n\n"
    
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
    """获取所有RSS源
    
    Returns:
        List[Tuple[str, str]]: RSS源列表，每个元素为 (id, title)
    """
    global storage
    
    if storage is None:
        storage = RSSStorage(db_path)
    
    feeds = storage.get_feeds()
    # 确保第一个选项是"全部"，ID为空字符串
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

def delete_feed(selected_rows) -> str:
    """删除RSS源"""
    global storage, rag_system
    
    if storage is None:
        storage = RSSStorage(db_path)
    
    # 检查是否有选中的行
    if not selected_rows or len(selected_rows) == 0:
        return "请选择要删除的RSS源"
    
    # 获取选中行的第一列（ID）
    try:
        feed_id = selected_rows[0][0]  # 获取第一个选中行的ID
        feed_title = selected_rows[0][1]  # 获取第一个选中行的标题
        
        # 如果ID为空或者标题是"全部"，不执行删除操作
        if not feed_id or feed_title == "全部":
            return "无法删除此RSS源"
        
        # 删除RSS源
        success = storage.delete_feed(int(feed_id))
        if not success:
            return f"错误: 删除RSS源失败 {feed_title}"
        
        # 如果RAG系统已初始化，重新加载数据
        if rag_system:
            rag_system.load_from_rss_db(db_path, incremental=True)
        
        return f"成功删除RSS源: {feed_title}"
    except Exception as e:
        logger.error(f"删除RSS源失败: {e}")
        return f"删除RSS源时发生错误: {str(e)}"

def get_feed_list(category_filter="全部", search_term=""):
    """获取RSS源列表，支持分类筛选和搜索"""
    global storage
    
    if storage is None:
        storage = RSSStorage(db_path)
    
    feeds = storage.get_feeds()
    
    # 构建数据
    feed_data = []
    for feed in feeds:
        # 如果有分类筛选且不匹配，则跳过
        if category_filter != "全部":
            feed_category = feed.category.split(" - ")[0] if " - " in feed.category else "未分类"
            if feed_category != category_filter:
                continue
        
        # 如果有搜索词且不匹配，则跳过
        if search_term and search_term.lower() not in feed.title.lower() and search_term.lower() not in (feed.category or "").lower():
            continue
        
        # 格式化最后更新时间
        last_updated = feed.last_updated.strftime("%Y-%m-%d %H:%M") if feed.last_updated else "未更新"
        
        feed_data.append([
            str(feed.id),
            feed.title,
            feed.category or "未分类",
            last_updated
        ])
    
    return feed_data

def update_feed_list(category_filter, search_term):
    """更新RSS源列表"""
    feed_data = get_feed_list(category_filter, search_term)
    return feed_data

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
        try:
            entry_count = storage.get_entry_count(feed_id=feed.id)
            unread_count = storage.get_unread_count(feed_id=feed.id)
            
            # 提取主分类（如果有分类层级）
            main_category = feed.category.split(" - ")[0] if feed.category and " - " in feed.category else (feed.category or "未分类")
            
            feed_data.append({
                "id": feed.id,
                "title": feed.title,
                "category": main_category,
                "entries": entry_count,
                "unread": unread_count,
                "last_updated": feed.last_updated.strftime("%Y-%m-%d %H:%M") if feed.last_updated else "未更新"
            })
        except Exception as e:
            logger.error(f"获取源统计信息失败: {feed.title} - {e}")
    
    # 创建DataFrame
    df = pd.DataFrame(feed_data)
    
    if len(df) == 0:
        return "无法获取RSS源统计信息", None
    
    try:
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
        
        # 按分类统计
        category_counts = df.groupby('category')['entries'].sum().sort_values(ascending=False)
        
        # 创建英文分类映射
        category_mapping = {}
        for i, cat in enumerate(category_counts.index):
            category_mapping[cat] = f"Category {i+1}"
        
        # 如果分类太多，只显示前5个，其他归为"其他"
        if len(category_counts) > 5:
            other_sum = category_counts[5:].sum()
            category_counts = category_counts[:5]
            category_counts['Other'] = other_sum
            category_mapping['Other'] = 'Other'
        
        # 使用饼图显示分类统计
        wedges, texts, autotexts = ax1.pie(
            category_counts, 
            labels=[category_mapping.get(cat, cat) for cat in category_counts.index],
            autopct='%1.1f%%', 
            startangle=90,
            textprops={'fontsize': 9}
        )
        ax1.set_title('Entries by Category')
        
        # 添加图例，显示分类映射
        legend_labels = [f"{category_mapping[cat]} = {cat}" for cat in category_counts.index if cat in category_mapping]
        if legend_labels:
            ax1.legend(legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=8, ncol=2)
        
        # 按源统计
        top_feeds = df.sort_values('entries', ascending=False).head(8)
        
        # 创建英文标题映射
        title_mapping = {}
        for i, title in enumerate(top_feeds['title']):
            title_mapping[title] = f"Source {i+1}"
        
        # 使用水平条形图，避免标题重叠
        bars = ax2.barh(
            y=range(len(top_feeds)), 
            width=top_feeds['entries'],
            height=0.6
        )
        
        # 设置Y轴标签为英文标识
        ax2.set_yticks(range(len(top_feeds)))
        ax2.set_yticklabels([title_mapping.get(title, title) for title in top_feeds['title']], fontsize=8)
        
        # 在条形图上显示数值
        for i, bar in enumerate(bars):
            ax2.text(
                bar.get_width() + 5, 
                bar.get_y() + bar.get_height()/2, 
                str(top_feeds['entries'].iloc[i]),
                va='center'
            )
        
        ax2.set_title('Top Sources by Entry Count')
        ax2.set_xlabel('Number of Entries')
        
        # 添加图例，显示标题映射
        legend_labels = [f"{title_mapping[title]} = {title}" for title in top_feeds['title'] if title in title_mapping]
        if legend_labels:
            ax2.legend(legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize=8, ncol=2)
        
        plt.tight_layout()
        
        # 生成表格HTML
        # 格式化DataFrame以便更好地显示
        display_df = df.copy()
        display_df = display_df[['title', 'category', 'entries', 'unread', 'last_updated']]
        display_df.columns = ['RSS源', '分类', '条目数', '未读数', '最后更新']
        
        # 按条目数排序
        display_df = display_df.sort_values('条目数', ascending=False)
        
        df_html = display_df.to_html(index=False, classes='table table-striped table-hover')
        
        # 添加Bootstrap样式
        df_html = f"""
        <style>
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 1rem;
        }}
        .table-striped tbody tr:nth-of-type(odd) {{
            background-color: rgba(0,0,0,.05);
        }}
        .table-hover tbody tr:hover {{
            background-color: rgba(0,0,0,.075);
        }}
        .table th, .table td {{
            padding: 0.75rem;
            border-top: 1px solid #dee2e6;
        }}
        </style>
        {df_html}
        """
        
        return df_html, fig
    except Exception as e:
        logger.error(f"生成统计图表失败: {e}")
        # 如果图表生成失败，至少返回表格数据
        display_df = df.copy()
        display_df = display_df[['title', 'category', 'entries', 'unread', 'last_updated']]
        display_df.columns = ['RSS源', '分类', '条目数', '未读数', '最后更新']
        df_html = display_df.to_html(index=False)
        return f"图表生成失败: {e}<br/>{df_html}", None

def update_system_config(
    device, base_dir, llm_type, llm_model, system_prompt,
    openai_api_key, deepseek_api_key, embedding_model,
    chunk_size, chunk_overlap, top_k, use_reranker,
    use_query_enhancement, use_cache
) -> str:
    """更新系统配置
    
    Args:
        device: 设备类型
        base_dir: 数据目录
        llm_type: LLM类型
        llm_model: LLM模型
        system_prompt: 系统提示词
        openai_api_key: OpenAI API密钥
        deepseek_api_key: DeepSeek API密钥
        embedding_model: Embedding模型
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
        top_k: 检索结果数量
        use_reranker: 是否使用重排序
        use_query_enhancement: 是否使用查询增强
        use_cache: 是否使用缓存
        
    Returns:
        str: 更新结果
    """
    global rag_system, config_manager
    
    if rag_system is None or config_manager is None:
        init_system()
    
    try:
        # 构建配置更新字典
        config_updates = {
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
        
        # 更新配置
        success = rag_system.update_config(config_updates, save_to_user_config=True)
        
        if success:
            return "配置已更新"
        else:
            return "配置更新失败，请检查配置值是否有效"
    except Exception as e:
        logger.error(f"更新配置失败: {e}")
        return f"更新配置失败: {str(e)}"

def get_current_config() -> Dict:
    """获取当前配置
    
    Returns:
        Dict: 当前配置
    """
    global config_manager
    
    if config_manager is None:
        init_system()
    
    return config_manager.get_config()

def get_config_schema() -> Dict:
    """获取配置模式信息
    
    Returns:
        Dict: 配置模式信息
    """
    global config_manager
    
    if config_manager is None:
        init_system()
    
    return config_manager.get_config_schema()

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
                            # 获取所有RSS源
                            feed_choices = get_feeds()
                            # 设置默认值为第一个选项（"全部"）
                            feed_dropdown = gr.Dropdown(
                                choices=[title for _, title in feed_choices], 
                                label="RSS源", 
                                value="全部"
                            )
                            days_input = gr.Number(label="时间范围（天）", value=30)
                        
                        with gr.Row():
                            show_refs_checkbox = gr.Checkbox(label="显示参考信息", value=True)
                            stream_checkbox = gr.Checkbox(label="流式输出", value=True)
                        
                        ask_button = gr.Button("提问", variant="primary")
                    
                    with gr.Column(scale=7):
                        answer_output = gr.Markdown(label="回答")
            
            # RSS源管理标签页
            with gr.TabItem("RSS源管理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## RSS源操作")
                        with gr.Tabs() as rss_tabs:
                            # 添加RSS源
                            with gr.TabItem("添加RSS源"):
                                with gr.Row():
                                    url_input = gr.Textbox(label="RSS源URL", placeholder="https://example.com/feed.xml")
                                
                                with gr.Row():
                                    category_input = gr.Textbox(label="分类", placeholder="科技、新闻等")
                                
                                with gr.Row():
                                    add_button = gr.Button("添加", variant="primary")
                            
                            # 导入OPML
                            with gr.TabItem("导入OPML"):
                                with gr.Row():
                                    opml_file = gr.File(label="OPML文件", file_types=[".opml", ".xml"])
                                
                                with gr.Row():
                                    import_button = gr.Button("导入", variant="primary")
                        
                        gr.Markdown("## 批量操作")
                        with gr.Row():
                            update_button = gr.Button("更新所有RSS源", variant="secondary")
                        
                        with gr.Row():
                            stats_button = gr.Button("生成统计", variant="secondary")
                        
                        with gr.Row():
                            add_result = gr.Textbox(label="操作结果", lines=5)
                    
                    with gr.Column(scale=2):
                        gr.Markdown("## RSS源列表")
                        
                        # 获取所有RSS源
                        feed_choices = get_feeds()
                        
                        # 分类过滤
                        with gr.Row():
                            categories = list(set([feed[1].split(" - ")[0] if " - " in feed[1] else "未分类" for feed in feed_choices[1:]]))
                            categories = ["全部"] + sorted(categories)
                            category_filter = gr.Dropdown(
                                choices=categories,
                                label="按分类筛选",
                                value="全部"
                            )
                            
                            search_input = gr.Textbox(
                                label="搜索RSS源",
                                placeholder="输入关键词搜索...",
                                show_label=False
                            )
                        
                        # RSS源列表
                        with gr.Row():
                            feed_list = gr.Dataframe(
                                headers=["ID", "标题", "分类", "最后更新"],
                                datatype=["str", "str", "str", "str"],
                                col_count=(4, "fixed"),
                                label="RSS源列表",
                                interactive=False,
                                wrap=True
                            )
                        
                        # 操作按钮
                        with gr.Row():
                            refresh_list_button = gr.Button("刷新列表", variant="secondary")
                            delete_button = gr.Button("删除选中的源", variant="stop")
                        
                        # 选中的源详情
                        with gr.Row():
                            feed_detail = gr.JSON(label="源详情", visible=False)
                            manage_result = gr.Textbox(label="操作结果", lines=3)
                
                # RSS源统计
                with gr.Row(visible=False) as stats_row:
                    with gr.Column(scale=1):
                        stats_table = gr.HTML(label="RSS源统计表")
                    
                    with gr.Column(scale=1):
                        stats_plot = gr.Plot(label="统计图表")
            
            # 系统配置标签页
            with gr.TabItem("系统配置"):
                # 获取当前配置和配置模式
                current_config = get_current_config()
                config_schema = get_config_schema()
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## 基础配置")
                        device = gr.Dropdown(
                            choices=["cpu", "cuda"], 
                            value=current_config.get("device", "cpu"),
                            label="设备"
                        )
                        base_dir = gr.Textbox(
                            value=current_config.get("base_dir", "data/rag_db"),
                            label="数据目录"
                        )
                        
                        gr.Markdown("## LLM配置")
                        llm_type = gr.Dropdown(
                            choices=["tiny", "openai", "huggingface", "deepseek"], 
                            value=current_config.get("llm_type", "tiny"),
                            label="LLM类型"
                        )
                        llm_model = gr.Textbox(
                            value=current_config.get("llm_model_id", "models/tiny_llm_sft_92m"),
                            label="LLM模型"
                        )
                        system_prompt = gr.Textbox(
                            value=current_config.get("system_prompt", "你是一个有用的AI助手。"),
                            label="系统提示词",
                            lines=3
                        )
                        
                        gr.Markdown("## API密钥")
                        openai_api_key = gr.Textbox(
                            value=current_config.get("openai_api_key", ""),
                            label="OpenAI API密钥",
                            type="password"
                        )
                        deepseek_api_key = gr.Textbox(
                            value=current_config.get("deepseek_api_key", ""),
                            label="DeepSeek API密钥",
                            type="password"
                        )
                    
                    with gr.Column():
                        gr.Markdown("## 模型配置")
                        embedding_model = gr.Textbox(
                            value=current_config.get("embedding_model_id", "models/bge-base-zh-v1.5"),
                            label="Embedding模型"
                        )
                        
                        gr.Markdown("## 文本分割配置")
                        chunk_size = gr.Slider(
                            minimum=50, maximum=2000, step=50,
                            value=current_config.get("chunk_size", 500),
                            label="分块大小"
                        )
                        chunk_overlap = gr.Slider(
                            minimum=0, maximum=500, step=10,
                            value=current_config.get("chunk_overlap", 50),
                            label="分块重叠"
                        )
                        
                        gr.Markdown("## 检索配置")
                        top_k = gr.Slider(
                            minimum=1, maximum=20, step=1,
                            value=current_config.get("top_k", 3),
                            label="检索结果数量"
                        )
                        use_reranker = gr.Checkbox(
                            value=current_config.get("use_reranker", True),
                            label="使用重排序"
                        )
                        use_query_enhancement = gr.Checkbox(
                            value=current_config.get("use_query_enhancement", False),
                            label="使用查询增强"
                        )
                        
                        gr.Markdown("## 缓存配置")
                        use_cache = gr.Checkbox(
                            value=current_config.get("use_cache", True),
                            label="使用缓存"
                        )
                
                with gr.Row():
                    with gr.Column():
                        save_config_btn = gr.Button("保存配置", variant="primary")
                        reset_config_btn = gr.Button("重置为默认配置", variant="secondary")
                    
                    with gr.Column():
                        config_status = gr.Textbox(label="状态", interactive=False)
            
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
                outputs=add_result
            )
            
            delete_button.click(
                fn=delete_feed,
                inputs=feed_list,  # 这里需要修改delete_feed函数以接受dataframe选择
                outputs=manage_result
            )
            
            update_button.click(
                fn=update_feeds,
                inputs=[],
                outputs=add_result
            )
            
            stats_button.click(
                fn=show_feed_stats,
                inputs=[],
                outputs=[stats_row, stats_table, stats_plot]
            )
            
            # 系统配置
            save_config_btn.click(
                fn=update_system_config,
                inputs=[
                    device, base_dir, llm_type, llm_model, system_prompt,
                    openai_api_key, deepseek_api_key, embedding_model,
                    chunk_size, chunk_overlap, top_k, use_reranker,
                    use_query_enhancement, use_cache
                ],
                outputs=config_status
            )
            
            # 重置配置按钮事件
            def reset_config():
                global config_manager
                if config_manager is None:
                    init_system()
                
                config_manager.reset_to_default()
                return "配置已重置为默认值，请刷新页面查看更新后的配置"
            
            reset_config_btn.click(
                fn=reset_config,
                inputs=None,
                outputs=config_status
            )
            
            # 标签页切换事件
            rss_tabs.select(
                fn=lambda: None,
                inputs=[],
                outputs=[]
            )
            
            tabs.select(
                fn=lambda tab_index: gr.update(choices=get_feeds()) if tab_index == 0 else None,
                inputs=[gr.State(0)],
                outputs=feed_dropdown
            ).then(
                fn=lambda tab_index: update_feed_list("全部", "") if tab_index == 1 else None,
                inputs=[gr.State(1)],
                outputs=feed_list
            )
            
            # 初始化RSS源列表
            app.load(
                fn=update_feed_list,
                inputs=[gr.State("全部"), gr.State("")],
                outputs=feed_list
            )
    
    return app

def main(host="127.0.0.1", port=7860, share=False):
    """主函数"""
    # 确保数据目录存在
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # 初始化系统
    init_system()
    
    # 创建并启动UI
    app = create_ui()
    app.launch(server_name=host, server_port=port, share=share)

def show_feed_stats():
    """显示RSS源统计"""
    html, plot = get_feed_stats()
    # 设置统计行可见
    return gr.update(visible=True), html, plot

def hide_feed_stats():
    """隐藏RSS源统计"""
    # 设置统计行不可见
    return gr.update(visible=False), None, None

if __name__ == "__main__":
    main() 