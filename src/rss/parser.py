"""
RSS解析器模块，用于解析RSS源和条目
"""
# import feedparser
from feedparser import parse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from .models import Feed, Entry
import requests
import cfscrape
import time

class RSSParser:
    """基础RSS解析器"""
    
    def __init__(self, custom_rsshub_url: str = "http://1374d70a.r7.cpolar.top"):
        """初始化解析器"""
        # 创建一个scraper实例
        self.scraper = cfscrape.create_scraper()
        self.custom_rsshub_url = custom_rsshub_url.rstrip('/')  # 移除末尾的斜杠
        
        # 设置通用请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
        }
    
    def validate_url(self, url: str) -> bool:
        """
        验证URL是否有效
        
        Args:
            url: 要验证的URL
            
        Returns:
            布尔值，表示URL是否有效
        """
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except:
            return False
    
    def _clean_html(self, html: str) -> str:
        """
        清理HTML内容
        
        Args:
            html: HTML内容
            
        Returns:
            清理后的纯文本
        """
        if not html:
            return ""
            
        # 使用BeautifulSoup清理HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # 移除script和style标签
        for script in soup(["script", "style"]):
            script.decompose()
            
        # 在块级元素后添加换行符
        for tag in soup.find_all(['p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            tag.append('\n')
            
        # 获取文本并规范化空白字符
        text = soup.get_text()
        text = ' '.join(line.strip() for line in text.splitlines() if line.strip())
        
        # 移除多余的空格
        text = ' '.join(text.split())
        
        return text
        
    def _create_summary(self, text: str, max_length: int = 200) -> str:
        """
        创建文本摘要
        
        Args:
            text: 原始文本
            max_length: 最大长度（包括省略号）
            
        Returns:
            摘要文本
        """
        if not text:
            return ""
            
        # 清理HTML
        text = self._clean_html(text)
        
        # 处理空白字符
        text = ' '.join(text.split())
        
        # 处理无效的max_length
        if max_length <= 0:
            return ""
            
        # 处理max_length=1的特殊情况
        if max_length == 1:
            return text[:1]
            
        # 如果文本长度小于等于max_length，直接返回
        if len(text) <= max_length:
            return text
            
        # 计算实际的最大长度（考虑省略号）
        actual_max = max_length - 3
        
        # 如果actual_max太小，直接返回省略号
        if actual_max <= 0:
            return "..."
            
        # 直接截取指定长度
        # 对于中文文本，每个字符都是一个有效的截断点
        return text[:actual_max] + "..."
        
    def _parse_date(self, date_str: str) -> datetime:
        """
        解析日期字符串
        
        Args:
            date_str: 日期字符串
            
        Returns:
            datetime对象
        """
        try:
            # 尝试解析常见的日期格式
            formats = [
                "%Y-%m-%dT%H:%M:%SZ",  # ISO 8601
                "%a, %d %b %Y %H:%M:%S %Z",  # RFC 822
                "%Y-%m-%d %H:%M:%S",
                "%d/%m/%Y %H:%M:%S"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
                    
            # 如果所有格式都失败，返回当前时间
            return datetime.now().replace(microsecond=0)
        except:
            return datetime.now().replace(microsecond=0)
            
    def _normalize_url(self, url: str, base_url: str) -> str:
        """
        规范化URL
        
        Args:
            url: 原始URL
            base_url: 基础URL
            
        Returns:
            规范化后的URL
        """
        if not url:
            return base_url
            
        # 处理协议相对URL
        if url.startswith('//'):
            return f"http:{url}"
            
        try:
            # 使用urlparse分析URL
            parsed = urlparse(url)
            if parsed.scheme and parsed.netloc:  # 绝对URL
                return url
                
            # 使用urljoin处理相对URL
            normalized = urljoin(base_url, url)
            
            # 确保结果是正确的
            if url.startswith('/'):  # 绝对路径
                parsed_base = urlparse(base_url)
                normalized = f"{parsed_base.scheme}://{parsed_base.netloc}{url}"
            elif url.startswith('..'):  # 上级路径
                normalized = urljoin(base_url, url)
            elif not url.startswith(('http://', 'https://', '/')):  # 相对路径
                if not base_url.endswith('/'):
                    base_url = base_url + '/'
                normalized = urljoin(base_url, url)
                
            return normalized
        except:
            return urljoin(base_url, url)
    
    def _is_rsshub_url(self, url: str) -> bool:
        """判断是否为RSSHub的URL"""
        parsed = urlparse(url)
        return 'rsshub.app' in parsed.netloc or parsed.netloc == urlparse(self.custom_rsshub_url).netloc

    def _convert_to_custom_rsshub_url(self, url: str) -> str:
        """将官方RSSHub URL转换为自定义实例URL"""
        parsed = urlparse(url)
        if 'rsshub.app' in parsed.netloc:
            # 替换域名为自定义实例
            path = parsed.path
            query = f"?{parsed.query}" if parsed.query else ""
            return f"{self.custom_rsshub_url}{path}{query}"
        return url

    def _fetch_rsshub_content(self, url: str) -> str:
        """获取RSSHub内容"""
        try:
            # 转换URL到自定义实例
            custom_url = self._convert_to_custom_rsshub_url(url)
            print(f"正在从自定义RSSHub实例获取内容: {custom_url}")
            
            # 创建新的scraper实例以确保每次请求都是新的会话
            scraper = cfscrape.create_scraper(delay=10)
            
            # 添加更多的请求头以模拟真实浏览器
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0',
                'DNT': '1'
            }
            
            # 首先尝试使用自定义实例
            try:
                response = scraper.get(custom_url, headers=headers, timeout=30)
                response.raise_for_status()
                return response.text
            except Exception as e:
                print(f"从自定义RSSHub实例获取失败: {e}")
                if 'rsshub.app' not in url:  # 如果不是官方URL，则直接返回失败
                    return ""
                    
                # 如果是官方URL，尝试使用原始URL作为备份
                print("尝试使用原始RSSHub URL作为备份...")
                response = scraper.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                return response.text
                
        except Exception as e:
            print(f"获取RSSHub内容失败: {e}")
            return ""

    def parse_feed(self, url: str) -> Optional[Feed]:
        """
        解析RSS源，返回Feed对象
        
        Args:
            url: RSS源URL
            
        Returns:
            Feed对象，如果解析失败则返回None
        """
        if not self.validate_url(url):
            return None
            
        try:
            # 如果是RSSHub的URL，使用特殊处理
            if self._is_rsshub_url(url):
                content = self._fetch_rsshub_content(url)
                if not content:
                    return None
                parsed = parse(content)
            else:
                parsed = parse(url)
            
            if parsed.bozo and not parsed.entries:
                print(f"解析RSS源失败: {url}")
                return None
                
            # 获取Feed标题
            title = parsed.feed.get('title', '')
            if not title and hasattr(parsed.feed, 'link'):
                title = parsed.feed.link
                
            # 获取网站URL
            site_url = parsed.feed.get('link', '')
            
            # 创建Feed对象
            feed = Feed(
                title=title,
                url=url,
                site_url=site_url,
                description=self._create_summary(parsed.feed.get('description', '')),
                last_updated=datetime.now()
            )
            
            return feed
        except Exception as e:
            print(f"解析RSS源时出错: {e}")
            return None
    
    def parse_entries(self, url: str, feed_id: int) -> List[Entry]:
        """
        解析RSS源中的条目，返回Entry对象列表
        
        Args:
            url: RSS源URL
            feed_id: Feed ID
            
        Returns:
            Entry对象列表
        """
        entries = []
        try:
            # 如果是RSSHub的URL，使用特殊处理
            if self._is_rsshub_url(url):
                content = self._fetch_rsshub_content(url)
                if not content:
                    return []
                parsed = parse(content)
            else:
                parsed = parse(url)
            
            if parsed.bozo and not parsed.entries:
                print(f"解析条目失败: {url}")
                return []
                
            for item in parsed.entries:
                # 提取发布日期
                published = None
                if hasattr(item, 'published_parsed') and item.published_parsed:
                    published = datetime(*item.published_parsed[:6])
                elif hasattr(item, 'updated_parsed') and item.updated_parsed:
                    published = datetime(*item.updated_parsed[:6])
                else:
                    published = datetime.now()
                    
                # 提取内容
                content = ''
                if hasattr(item, 'content') and item.content:
                    content = item.content[0].value
                elif hasattr(item, 'description'):
                    content = item.description
                    
                entry = Entry(
                    feed_id=feed_id,
                    title=item.get('title', 'Unnamed Entry'),
                    link=self._normalize_url(item.get('link', ''), url),
                    published_date=published,
                    author=item.get('author', ''),
                    summary=self._create_summary(item.get('summary', '')),
                    content=self._clean_html(content),
                    read_status=False
                )
                
                entries.append(entry)
                
            return entries
        except Exception as e:
            print(f"解析条目时出错: {e}")
            return []
    
if __name__ == "__main__":
    parser = RSSParser()
    feed = parser.parse_feed("https://news.ycombinator.com/rss")
    print(feed)
