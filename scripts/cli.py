#!/usr/bin/env python
"""
RSS-RAG 命令行工具
用于测试RSS解析器和数据存储功能
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rss.parser import RSSParser
from src.rss.storage import RSSStorage
from src.rss.models import Feed, Entry
from src.rss.opml_parser import OPMLParser
from src.rag.config import RAGConfig
from src.rag.rss_rag import RSSRAG

def setup_db(args):
    """初始化数据库"""
    storage = RSSStorage(args.db_path)
    print(f"数据库已初始化: {args.db_path}")

def add_feed(args):
    """添加RSS源"""
    parser = RSSParser()
    storage = RSSStorage(args.db_path)
    
    feed = parser.parse_feed(args.url)
    if not feed:
        print(f"无法解析RSS源: {args.url}")
        return
    
    feed_id = storage.add_feed(feed)
    if feed_id > 0:
        print(f"RSS源已添加: {feed.title} (ID: {feed_id})")
        
        # 获取条目
        entries = parser.parse_entries(args.url, feed_id)
        count = storage.add_entries(entries)
        print(f"已添加 {count} 个条目")
    else:
        print(f"添加RSS源失败: {args.url}")

def list_feeds(args):
    """列出所有RSS源"""
    storage = RSSStorage(args.db_path)
    feeds = storage.get_feeds(args.category)
    
    if not feeds:
        print("没有找到RSS源")
        return
    
    print(f"找到 {len(feeds)} 个RSS源:")
    for feed in feeds:
        unread_count = storage.get_unread_count(feed.id)
        print(f"ID: {feed.id}, 标题: {feed.title}, 未读: {unread_count}, 分类: {feed.category or '无'}")

def delete_feed(args):
    """删除RSS源"""
    storage = RSSStorage(args.db_path)
    success = storage.delete_feed(args.feed_id)
    
    if success:
        print(f"RSS源已删除: ID {args.feed_id}")
    else:
        print(f"删除RSS源失败: ID {args.feed_id}")

def update_feeds(args):
    """更新所有或指定的RSS源"""
    storage = RSSStorage(args.db_path)
    parser = RSSParser()
    
    if args.feed_id:
        # 更新指定的RSS源
        feed = storage.get_feed_by_id(args.feed_id)
        if not feed:
            print(f"未找到RSS源: ID {args.feed_id}")
            return
            
        print(f"正在更新: {feed.title}")
        entries = parser.parse_entries(feed.url, feed.id)
        print(entries)
        count = storage.add_entries(entries)
        print(count)
        storage.update_feed_timestamp(feed.id)
        print(f"更新完成，新增 {count} 个条目")
    else:
        # 更新所有RSS源
        feeds = storage.get_feeds()
        if not feeds:
            print("没有找到RSS源")
            return
        
        total_new = 0
        for feed in feeds:
            print(f"正在更新: {feed.title}")
            entries = parser.parse_entries(feed.url, feed.id)
            count = storage.add_entries(entries)
            storage.update_feed_timestamp(feed.id)
            total_new += count
            print(f"  - 新增 {count} 个条目")
        
        print(f"更新完成，共新增 {total_new} 个条目")

def list_entries(args):
    """列出条目"""
    storage = RSSStorage(args.db_path)
    
    entries = storage.get_entries(
        feed_id=args.feed_id, 
        limit=args.limit, 
        offset=args.offset,
        unread_only=args.unread
    )
    
    if not entries:
        print("没有找到条目")
        return
    
    print(f"找到 {len(entries)} 个条目:")
    for entry in entries:
        status = "未读" if not entry.read_status else "已读"
        date_str = entry.published_date.strftime("%Y-%m-%d %H:%M")
        print(f"ID: {entry.id}, [{status}] {date_str} - {entry.title}")
        if args.verbose:
            print(f"  链接: {entry.link}")
            print(f"  作者: {entry.author}")
            print(f"  摘要: {entry.summary[:100]}...")
            print()

def view_entry(args):
    """查看条目详情"""
    storage = RSSStorage(args.db_path)
    entry = storage.get_entry_by_id(args.entry_id)
    
    if not entry:
        print(f"未找到条目: ID {args.entry_id}")
        return
    
    feed = storage.get_feed_by_id(entry.feed_id)
    feed_title = feed.title if feed else "未知来源"
    
    print(f"标题: {entry.title}")
    print(f"来源: {feed_title}")
    print(f"链接: {entry.link}")
    print(f"作者: {entry.author}")
    print(f"发布时间: {entry.published_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"状态: {'已读' if entry.read_status else '未读'}")
    print("\n摘要:")
    print(entry.summary)
    
    if args.mark_read and not entry.read_status:
        storage.mark_entry_as_read(entry.id)
        print("\n已标记为已读")
    
    if args.content:
        print("\n内容:")
        # 简单去除HTML标签
        content = entry.content.replace('<', '&lt;').replace('>', '&gt;')
        print(content)

def mark_read(args):
    """标记条目为已读"""
    storage = RSSStorage(args.db_path)
    
    if args.all:
        count = storage.mark_all_entries_as_read(args.feed_id)
        if args.feed_id:
            print(f"已将 {count} 个条目标记为已读 (Feed ID: {args.feed_id})")
        else:
            print(f"已将所有 {count} 个条目标记为已读")
    elif args.entry_id:
        success = storage.mark_entry_as_read(args.entry_id)
        if success:
            print(f"条目已标记为已读: ID {args.entry_id}")
        else:
            print(f"标记条目失败: ID {args.entry_id}")
    else:
        print("请指定条目ID或使用--all参数")

def export_data(args):
    """导出数据"""
    storage = RSSStorage(args.db_path)
    
    data = {
        "feeds": [],
        "entries": []
    }
    
    # 导出Feed
    feeds = storage.get_feeds()
    for feed in feeds:
        feed_dict = {
            "id": feed.id,
            "title": feed.title,
            "url": feed.url,
            "site_url": feed.site_url,
            "description": feed.description,
            "last_updated": feed.last_updated.isoformat(),
            "category": feed.category
        }
        data["feeds"].append(feed_dict)
        
        # 如果需要导出条目
        if not args.feeds_only:
            entries = storage.get_entries(feed_id=feed.id, limit=args.limit)
            for entry in entries:
                entry_dict = {
                    "id": entry.id,
                    "feed_id": entry.feed_id,
                    "title": entry.title,
                    "link": entry.link,
                    "published_date": entry.published_date.isoformat(),
                    "author": entry.author,
                    "summary": entry.summary,
                    "read_status": entry.read_status
                }
                
                # 只有在指定时才导出内容
                if args.include_content:
                    entry_dict["content"] = entry.content
                    
                data["entries"].append(entry_dict)
    
    # 写入文件
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"数据已导出到: {args.output}")
    print(f"  - {len(data['feeds'])} 个RSS源")
    print(f"  - {len(data['entries'])} 个条目")

def ask_question(args):
    """问答功能"""
    storage = RSSStorage(args.db_path)
    
    # 创建RAG配置
    config = RAGConfig(
        base_dir="data/rag_db",
        device="cuda",
        # embedding_model_id=args.embedding_model,
        # llm_model_id=args.llm_model,
        top_k=args.top_k
    )
    
    # 初始化RAG系统
    rag = RSSRAG(config)
    
    # 加载RAG状态
    rag.load_state(db_path=args.db_path, days=args.days or 30)
    
    # 如果指定了强制重建，则重新加载所有数据
    if args.rebuild:
        print("重新构建RAG索引...")
        rag.load_from_rss_db(args.db_path, days=args.days or 30, incremental=False)
        rag.save_state()
    
    print("RAG系统已准备就绪")
    
    # 构建日期范围（如果指定）
    date_range = None
    if args.days:
        start_date = datetime.now() - timedelta(days=args.days)
        date_range = (start_date, datetime.now())
    
    # 回答问题
    print(f"\n问题: {args.question}")
    print("\n回答:")
    
    if args.stream:
        # 流式输出
        for token in rag.answer_stream(
            query=args.question,
            feed_id=args.feed_id,
            date_range=date_range,
            top_k=args.top_k
        ):
            print(token, end='', flush=True)
        print()
    else:
        # 一次性输出
        answer = rag.answer(
            query=args.question,
            feed_id=args.feed_id,
            date_range=date_range,
            top_k=args.top_k
        )
        print(answer)

def import_opml(args):
    """导入OPML文件"""
    storage = RSSStorage(args.db_path)
    parser = OPMLParser()
    
    # 检查文件是否存在
    if not os.path.exists(args.file):
        print(f"错误: OPML文件不存在: {args.file}")
        return
        
    print(f"正在解析OPML文件: {args.file}")
    
    # 解析OPML文件
    feeds = parser.parse_opml(args.file)
    
    if not feeds:
        print(f"从OPML文件中未找到RSS源: {args.file}")
        return
        
    print(f"找到 {len(feeds)} 个RSS源")
    
    # 添加到数据库
    added_count = 0
    skipped_count = 0
    error_count = 0
    rss_parser = RSSParser()
    
    for feed in feeds:
        try:
            # 检查是否已存在
            existing_feeds = storage.get_feeds_by_url(feed.url)
            if existing_feeds:
                print(f"跳过已存在的RSS源: {feed.title} ({feed.url})")
                skipped_count += 1
                continue
                
            # 添加到数据库
            feed_id = storage.add_feed(feed)
            if feed_id > 0:
                added_count += 1
                print(f"已添加RSS源: {feed.title} (ID: {feed_id})")
                
                # 如果指定了获取条目
                if args.fetch_entries:
                    try:
                        print(f"  正在获取条目...")
                        entries = rss_parser.parse_entries(feed.url, feed_id)
                        count = storage.add_entries(entries)
                        print(f"  已添加 {count} 个条目")
                    except Exception as e:
                        print(f"  获取条目失败: {e}")
            else:
                print(f"添加RSS源失败: {feed.title}")
                error_count += 1
        except Exception as e:
            print(f"处理RSS源时出错: {feed.title} - {e}")
            error_count += 1
    
    print(f"\n导入完成: 添加 {added_count} 个RSS源，跳过 {skipped_count} 个，失败 {error_count} 个")
    
    # 如果指定了更新所有源
    if args.update_all and added_count > 0:
        print("\n正在更新所有RSS源...")
        update_feeds(args)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RSS-RAG 命令行工具')
    parser.add_argument('--db', dest='db_path', default='data/rss.db',
                        help='数据库文件路径 (默认: data/rss.db)')
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 初始化数据库
    setup_parser = subparsers.add_parser('setup', help='初始化数据库')
    
    # 添加RSS源
    add_parser = subparsers.add_parser('add', help='添加RSS源')
    add_parser.add_argument('url', help='RSS源URL')
    
    # 列出RSS源
    list_parser = subparsers.add_parser('list', help='列出RSS源')
    list_parser.add_argument('--category', help='按分类过滤')
    
    # 删除RSS源
    delete_parser = subparsers.add_parser('delete', help='删除RSS源')
    delete_parser.add_argument('feed_id', type=int, help='Feed ID')
    
    # 更新RSS源
    update_parser = subparsers.add_parser('update', help='更新RSS源')
    update_parser.add_argument('--feed', dest='feed_id', type=int, help='指定要更新的Feed ID，不指定则更新所有')
    
    # 列出条目
    entries_parser = subparsers.add_parser('entries', help='列出条目')
    entries_parser.add_argument('--feed', dest='feed_id', type=int, help='Feed ID')
    entries_parser.add_argument('--limit', type=int, default=20, help='返回条目数量限制 (默认: 20)')
    entries_parser.add_argument('--offset', type=int, default=0, help='分页偏移量 (默认: 0)')
    entries_parser.add_argument('--unread', action='store_true', help='只显示未读条目')
    entries_parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    
    # 查看条目
    view_parser = subparsers.add_parser('view', help='查看条目详情')
    view_parser.add_argument('entry_id', type=int, help='条目ID')
    view_parser.add_argument('--content', action='store_true', help='显示完整内容')
    view_parser.add_argument('--mark-read', action='store_true', help='标记为已读')
    
    # 标记为已读
    mark_parser = subparsers.add_parser('mark-read', help='标记条目为已读')
    mark_parser.add_argument('--entry', dest='entry_id', type=int, help='条目ID')
    mark_parser.add_argument('--feed', dest='feed_id', type=int, help='Feed ID')
    mark_parser.add_argument('--all', action='store_true', help='标记所有条目')
    
    # 导出数据
    export_parser = subparsers.add_parser('export', help='导出数据')
    export_parser.add_argument('--output', default='data/export.json', help='输出文件路径 (默认: data/export.json)')
    export_parser.add_argument('--feeds-only', action='store_true', help='只导出RSS源')
    export_parser.add_argument('--include-content', action='store_true', help='包含条目内容')
    export_parser.add_argument('--limit', type=int, default=1000, help='每个Feed导出的条目数量限制 (默认: 1000)')
    
    # 问答功能
    ask_parser = subparsers.add_parser('ask', help='提问问题')
    ask_parser.add_argument('question', help='问题')
    ask_parser.add_argument('--feed', dest='feed_id', type=int, help='限制特定Feed')
    ask_parser.add_argument('--days', type=int, help='限制最近N天的内容')
    ask_parser.add_argument('--top-k', type=int, default=3, help='检索结果数量 (默认: 3)')
    # ask_parser.add_argument('--embedding-model', 
    #                       default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    #                       help='Embedding模型ID')
    # ask_parser.add_argument('--llm-model', default='gpt-3.5-turbo',
    #                       help='LLM模型ID')
    ask_parser.add_argument('--stream', action='store_true', help='流式输出')
    ask_parser.add_argument('--rebuild', action='store_true', 
                          help='重新构建RAG索引（否则使用增量更新）')
    
    # 导入OPML
    import_parser = subparsers.add_parser('import', help='导入OPML文件')
    import_parser.add_argument('file', help='OPML文件路径')
    import_parser.add_argument('--fetch', dest='fetch_entries', action='store_true',
                             help='同时获取条目')
    import_parser.add_argument('--update', dest='update_all', action='store_true',
                             help='导入后更新所有RSS源')
    
    args = parser.parse_args()
    
    # 创建数据目录
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
    
    # 执行命令
    if args.command == 'setup':
        setup_db(args)
    elif args.command == 'add':
        add_feed(args)
    elif args.command == 'list':
        list_feeds(args)
    elif args.command == 'delete':
        delete_feed(args)
    elif args.command == 'update':
        update_feeds(args)
    elif args.command == 'entries':
        list_entries(args)
    elif args.command == 'view':
        view_entry(args)
    elif args.command == 'mark-read':
        mark_read(args)
    elif args.command == 'export':
        export_data(args)
    elif args.command == 'ask':
        ask_question(args)
    elif args.command == 'import':
        import_opml(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 