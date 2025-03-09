"""
OPML解析器，用于导入OPML格式的RSS源
使用listparser库解析OPML文件
"""
import listparser
from typing import List, Dict, Optional
from loguru import logger
from datetime import datetime
import os
import json

from .models import Feed

class OPMLParser:
    """OPML解析器类，使用listparser库"""
    
    def parse_opml(self, file_path: str) -> List[Feed]:
        """
        解析OPML文件，提取RSS源信息
        
        Args:
            file_path: OPML文件路径
            
        Returns:
            Feed对象列表
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"OPML文件不存在: {file_path}")
                return []
                
            # 使用listparser解析OPML文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return self.parse_opml_text(content)
            
        except Exception as e:
            logger.error(f"解析OPML文件失败: {e}")
            return []
            
    def parse_opml_text(self, opml_text: str) -> List[Feed]:
        """
        解析OPML文本内容，提取RSS源信息
        
        Args:
            opml_text: OPML文本内容
            
        Returns:
            Feed对象列表
        """
        try:
            # 使用listparser解析OPML文本
            result = listparser.parse(opml_text)
            
            if not result.feeds:
                logger.warning("OPML文本中未找到RSS源")
                return []
                
            # 转换为Feed对象
            feeds = self._convert_to_feeds(result)
            
            logger.info(f"从OPML文本中解析出 {len(feeds)} 个RSS源")
            return feeds
            
        except Exception as e:
            logger.error(f"解析OPML文本失败: {e}")
            return []
            
    def _convert_to_feeds(self, parse_result) -> List[Feed]:
        """
        将listparser解析结果转换为Feed对象列表
        
        Args:
            parse_result: listparser解析结果
            
        Returns:
            Feed对象列表
        """
        feeds = []
        
        # 处理所有源
        for feed_data in parse_result.feeds:
            try:
                # 获取必要信息
                url = feed_data.get('url', '')
                if not url:
                    continue
                    
                # 获取标题
                title = feed_data.get('title', '')
                if not title:
                    # 尝试使用text作为标题
                    title = feed_data.get('text', 'Unnamed Feed')
                    
                # 获取网站URL
                site_url = feed_data.get('link', '')
                
                # 获取分类
                categories = feed_data.get('categories', [])
                
                # 处理分类，确保是字符串
                category = None
                if categories:
                    # 将所有分类转换为字符串列表
                    category_list = []
                    if isinstance(categories, list):
                        for cat in categories:
                            if isinstance(cat, list):
                                # 如果是嵌套列表，展平并添加所有项
                                category_list.extend(str(item) for item in cat)
                            else:
                                category_list.append(str(cat))
                    else:
                        category_list.append(str(categories))
                    
                    # 用分号连接所有分类
                    if category_list:
                        category = ';'.join(category_list)
                
                # 创建Feed对象
                feed = Feed(
                    title=title,
                    url=url,
                    site_url=site_url,
                    description=f"从OPML导入: {title}",
                    category=category,
                    last_updated=datetime.now()
                )
                
                feeds.append(feed)
            except Exception as e:
                logger.error(f"处理Feed时出错 ({title}): {e}")
                logger.debug(f"Feed数据: {json.dumps(feed_data, default=str)}")
                continue
            
        return feeds 