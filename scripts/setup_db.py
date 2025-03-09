#!/usr/bin/env python
"""
数据库初始化脚本
"""
import os
import sys
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rss.storage import RSSStorage

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='初始化RSS-RAG数据库')
    parser.add_argument('--db', dest='db_path', default='data/rss.db',
                        help='数据库文件路径 (默认: data/rss.db)')
    args = parser.parse_args()
    
    # 确保数据目录存在
    os.makedirs(os.path.dirname(os.path.abspath(args.db_path)), exist_ok=True)
    
    # 初始化数据库
    storage = RSSStorage(args.db_path)
    print(f"数据库已初始化: {args.db_path}")

if __name__ == '__main__':
    main() 