#!/usr/bin/env python
"""
RSS-RAG Gradio界面启动脚本
"""
import os
import sys
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.ui.app import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RSS-RAG Gradio界面')
    parser.add_argument('--port', type=int, default=7860, help='服务端口 (默认: 7860)')
    parser.add_argument('--host', default='0.0.0.0', help='服务主机 (默认: 0.0.0.0)')
    parser.add_argument('--share', action='store_true', help='创建公共链接')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    # 设置环境变量
    if args.debug:
        os.environ['GRADIO_DEBUG'] = 'true'
    
    # 启动UI
    main(host=args.host, port=args.port, share=args.share) 