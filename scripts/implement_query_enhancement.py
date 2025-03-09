#!/usr/bin/env python
"""
RSS-RAG 查询增强实现工具
用于实现查询增强策略，提高检索质量
"""
import os
import sys
import argparse
import re
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def implement_query_enhancement(rss_rag_path, backup=True):
    """实现查询增强策略"""
    print(f"\n===== 实现查询增强策略 =====")
    
    # 检查文件是否存在
    if not os.path.exists(rss_rag_path):
        print(f"错误: 文件不存在 - {rss_rag_path}")
        return False
    
    # 备份原始文件
    if backup:
        backup_path = f"{rss_rag_path}.bak"
        print(f"备份原始文件到 {backup_path}")
        with open(rss_rag_path, 'r', encoding='utf-8') as src:
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
    
    # 读取文件内容
    with open(rss_rag_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找answer方法
    answer_pattern = r'def answer\(self,\s+query:\s+str,.*?\).*?:'
    answer_match = re.search(answer_pattern, content, re.DOTALL)
    
    if not answer_match:
        print("错误: 无法找到answer方法")
        return False
    
    # 查找search_results行
    search_results_pattern = r'(\s+)(search_results\s*=\s*self\.search\(query,.*?\))'
    search_results_match = re.search(search_results_pattern, content, re.DOTALL)
    
    if not search_results_match:
        print("错误: 无法找到search_results行")
        return False
    
    # 构建替换内容
    indent = search_results_match.group(1)
    original_search = search_results_match.group(2)
    
    # 新的代码
    enhanced_code = f"{indent}# 先用LLM生成初步回答\n"
    enhanced_code += f"{indent}print(\"生成初步回答...\")\n"
    enhanced_code += f"{indent}initial_answer = self.llm.generate(query)\n"
    enhanced_code += f"{indent}print(f\"初步回答: {{initial_answer[:100]}}...\")\n\n"
    enhanced_code += f"{indent}# 增强查询\n"
    enhanced_code += f"{indent}enhanced_query = query + \" \" + initial_answer\n"
    enhanced_code += f"{indent}print(f\"增强查询: {{enhanced_query[:100]}}...\")\n\n"
    enhanced_code += f"{indent}# 使用增强查询检索\n"
    enhanced_code += f"{indent}search_results = self.search(enhanced_query, "
    
    # 替换原始search_results行
    new_content = content.replace(
        original_search,
        original_search.replace("self.search(query,", "self.search(enhanced_query,")
    )
    
    # 在search_results行之前插入增强代码
    new_content = new_content.replace(
        original_search.replace("self.search(query,", "self.search(enhanced_query,"),
        enhanced_code + original_search.split("self.search(")[1]
    )
    
    # 写入修改后的文件
    with open(rss_rag_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("查询增强策略已成功实现!")
    print("\n修改内容:")
    print("1. 在answer方法中添加了初步回答生成")
    print("2. 基于初步回答增强原始查询")
    print("3. 使用增强查询进行检索")
    
    print("\n使用方法:")
    print("运行以下命令测试查询增强效果:")
    print("python scripts/cli.py ask \"你的问题\"")
    
    return True

def implement_chunk_size_filter(text_splitter_path, min_length=100, backup=True):
    """实现分块大小过滤"""
    print(f"\n===== 实现分块大小过滤 =====")
    
    # 检查文件是否存在
    if not os.path.exists(text_splitter_path):
        print(f"错误: 文件不存在 - {text_splitter_path}")
        return False
    
    # 备份原始文件
    if backup:
        backup_path = f"{text_splitter_path}.bak"
        print(f"备份原始文件到 {backup_path}")
        with open(text_splitter_path, 'r', encoding='utf-8') as src:
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
    
    # 读取文件内容
    with open(text_splitter_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找split_text方法
    split_text_pattern = r'def split_text\(self,\s+text:\s+str\).*?:'
    split_text_match = re.search(split_text_pattern, content, re.DOTALL)
    
    if not split_text_match:
        print("错误: 无法找到split_text方法")
        return False
    
    # 查找return语句
    return_pattern = r'(\s+)(return\s+chunks)'
    return_match = re.search(return_pattern, content)
    
    if not return_match:
        print("错误: 无法找到return语句")
        return False
    
    # 构建替换内容
    indent = return_match.group(1)
    original_return = return_match.group(2)
    
    # 新的代码
    filter_code = f"{indent}# 过滤掉过短的片段\n"
    filter_code += f"{indent}chunks = [chunk for chunk in chunks if len(chunk) >= {min_length}]\n"
    filter_code += f"{indent}{original_return}"
    
    # 替换原始return语句
    new_content = content.replace(
        indent + original_return,
        filter_code
    )
    
    # 写入修改后的文件
    with open(text_splitter_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"分块大小过滤已成功实现! 最小分块长度: {min_length}字符")
    
    return True

def update_chunk_size(config_path, chunk_size=800, chunk_overlap=100, backup=True):
    """更新分块大小配置"""
    print(f"\n===== 更新分块大小配置 =====")
    
    # 检查文件是否存在
    if not os.path.exists(config_path):
        print(f"错误: 文件不存在 - {config_path}")
        return False
    
    # 备份原始文件
    if backup:
        backup_path = f"{config_path}.bak"
        print(f"备份原始文件到 {backup_path}")
        with open(config_path, 'r', encoding='utf-8') as src:
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
    
    # 读取文件内容
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找chunk_size参数
    chunk_size_pattern = r'(chunk_size:\s*int\s*=\s*)(\d+)'
    chunk_size_match = re.search(chunk_size_pattern, content)
    
    if not chunk_size_match:
        print("错误: 无法找到chunk_size参数")
        return False
    
    # 查找chunk_overlap参数
    chunk_overlap_pattern = r'(chunk_overlap:\s*int\s*=\s*)(\d+)'
    chunk_overlap_match = re.search(chunk_overlap_pattern, content)
    
    if not chunk_overlap_match:
        print("错误: 无法找到chunk_overlap参数")
        return False
    
    # 替换参数值
    new_content = re.sub(
        chunk_size_pattern,
        f"\\1{chunk_size}",
        content
    )
    
    new_content = re.sub(
        chunk_overlap_pattern,
        f"\\1{chunk_overlap}",
        new_content
    )
    
    # 写入修改后的文件
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"分块大小配置已更新: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    return True

def update_retrieval_weights(hybrid_retriever_path, bm25_weight=0.4, vector_weight=0.6, backup=True):
    """更新检索权重"""
    print(f"\n===== 更新检索权重 =====")
    
    # 检查文件是否存在
    if not os.path.exists(hybrid_retriever_path):
        print(f"错误: 文件不存在 - {hybrid_retriever_path}")
        return False
    
    # 备份原始文件
    if backup:
        backup_path = f"{hybrid_retriever_path}.bak"
        print(f"备份原始文件到 {backup_path}")
        with open(hybrid_retriever_path, 'r', encoding='utf-8') as src:
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
    
    # 读取文件内容
    with open(hybrid_retriever_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找权重设置
    weights_pattern = r"(weights\s*=\s*\{'bm25':\s*)([0-9.]+)(\s*,\s*'vector':\s*)([0-9.]+)(\s*\})"
    weights_match = re.search(weights_pattern, content)
    
    if not weights_match:
        print("错误: 无法找到权重设置")
        return False
    
    # 替换权重值
    new_content = re.sub(
        weights_pattern,
        f"\\1{bm25_weight}\\3{vector_weight}\\5",
        content
    )
    
    # 写入修改后的文件
    with open(hybrid_retriever_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"检索权重已更新: bm25={bm25_weight}, vector={vector_weight}")
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RSS-RAG 查询增强实现工具')
    parser.add_argument('--all', action='store_true',
                        help='实现所有改进')
    parser.add_argument('--query-enhancement', action='store_true',
                        help='实现查询增强策略')
    parser.add_argument('--chunk-filter', action='store_true',
                        help='实现分块大小过滤')
    parser.add_argument('--update-chunk-size', action='store_true',
                        help='更新分块大小配置')
    parser.add_argument('--update-weights', action='store_true',
                        help='更新检索权重')
    parser.add_argument('--min-length', type=int, default=100,
                        help='最小分块长度 (默认: 100)')
    parser.add_argument('--chunk-size', type=int, default=800,
                        help='分块大小 (默认: 800)')
    parser.add_argument('--chunk-overlap', type=int, default=100,
                        help='分块重叠 (默认: 100)')
    parser.add_argument('--bm25-weight', type=float, default=0.4,
                        help='BM25权重 (默认: 0.4)')
    parser.add_argument('--vector-weight', type=float, default=0.6,
                        help='向量检索权重 (默认: 0.6)')
    parser.add_argument('--no-backup', action='store_true',
                        help='不备份原始文件')
    
    args = parser.parse_args()
    
    # 设置文件路径
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
    rss_rag_path = os.path.join(src_dir, 'rag', 'rss_rag.py')
    text_splitter_path = os.path.join(src_dir, 'rag', 'utils', 'text_splitter.py')
    config_path = os.path.join(src_dir, 'rag', 'config.py')
    hybrid_retriever_path = os.path.join(src_dir, 'rag', 'retrieval', 'hybrid_retriever.py')
    
    # 检查文件是否存在
    for path in [rss_rag_path, text_splitter_path, config_path, hybrid_retriever_path]:
        if not os.path.exists(path):
            print(f"错误: 文件不存在 - {path}")
            return
    
    # 如果没有指定具体操作，则默认实现查询增强
    if not (args.all or args.query_enhancement or args.chunk_filter or args.update_chunk_size or args.update_weights):
        args.query_enhancement = True
    
    # 执行选定的操作
    if args.all or args.query_enhancement:
        implement_query_enhancement(rss_rag_path, not args.no_backup)
    
    if args.all or args.chunk_filter:
        implement_chunk_size_filter(text_splitter_path, args.min_length, not args.no_backup)
    
    if args.all or args.update_chunk_size:
        update_chunk_size(config_path, args.chunk_size, args.chunk_overlap, not args.no_backup)
    
    if args.all or args.update_weights:
        update_retrieval_weights(hybrid_retriever_path, args.bm25_weight, args.vector_weight, not args.no_backup)
    
    print("\n===== 所有改进已完成 =====")
    print("请重新启动系统以应用更改")
    print("运行以下命令测试效果:")
    print("python scripts/cli.py ask \"你的问题\"")

if __name__ == '__main__':
    main() 