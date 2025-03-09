#!/usr/bin/env python
"""
RSS-RAG 主入口文件
"""
import os
import sys
import argparse

from src.rss.parser import RSSParser
from src.rss.storage import RSSStorage

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RSS-RAG 智能问答系统')
    parser.add_argument('--db', dest='db_path', default='data/rss.db',
                        help='数据库文件路径 (默认: data/rss.db)')
    parser.add_argument('--setup', action='store_true',
                        help='初始化数据库')
    args = parser.parse_args()
    
    # 确保数据目录存在
    os.makedirs(os.path.dirname(os.path.abspath(args.db_path)), exist_ok=True)
    
    # 初始化数据库
    if args.setup:
        storage = RSSStorage(args.db_path)
        print(f"数据库已初始化: {args.db_path}")
        return
    
    # 这里将来会启动Gradio UI
    print("RSS-RAG 智能问答系统")
    print("请使用 --setup 参数初始化数据库")
    print("或使用 scripts/cli.py 进行RSS源管理")

if __name__ == '__main__':
    main() 