"""
RSS数据存储模块，用于存储和检索RSS数据
"""
import sqlite3
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from .models import Feed, Entry

class RSSStorage:
    """RSS数据存储类"""
    
    def __init__(self, db_path: str):
        """
        初始化存储
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """创建数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建feeds表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feeds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            url TEXT UNIQUE NOT NULL,
            site_url TEXT,
            description TEXT,
            last_updated TIMESTAMP,
            category TEXT
        )
        ''')
        
        # 创建entries表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feed_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            link TEXT UNIQUE NOT NULL,
            published_date TIMESTAMP,
            author TEXT,
            summary TEXT,
            content TEXT,
            read_status BOOLEAN DEFAULT 0,
            FOREIGN KEY (feed_id) REFERENCES feeds (id) ON DELETE CASCADE
        )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entries_feed_id ON entries (feed_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entries_published_date ON entries (published_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entries_read_status ON entries (read_status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feeds_category ON feeds (category)')
        
        conn.commit()
        conn.close()
    
    def add_feed(self, feed: Feed) -> int:
        """
        添加RSS源，返回ID
        
        Args:
            feed: Feed对象
            
        Returns:
            新添加的Feed ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 确保分类是字符串或None
            category = feed.category
            if category is not None and not isinstance(category, str):
                category = str(category)
                
            cursor.execute(
                'INSERT INTO feeds (title, url, site_url, description, last_updated, category) VALUES (?, ?, ?, ?, ?, ?)',
                (feed.title, feed.url, feed.site_url, feed.description, feed.last_updated, category)
            )
            
            feed_id = cursor.lastrowid
            conn.commit()
            return feed_id
        except sqlite3.IntegrityError:
            # 如果URL已存在，返回现有Feed的ID
            cursor.execute('SELECT id FROM feeds WHERE url = ?', (feed.url,))
            result = cursor.fetchone()
            return result[0] if result else -1
        finally:
            conn.close()
    
    def update_feed(self, feed_id: int, title: Optional[str] = None, 
                   site_url: Optional[str] = None, description: Optional[str] = None,
                   category: Optional[str] = None) -> bool:
        """
        更新RSS源信息
        
        Args:
            feed_id: Feed ID
            title: 新标题（可选）
            site_url: 新站点URL（可选）
            description: 新描述（可选）
            category: 新分类（可选）
            
        Returns:
            布尔值，表示更新是否成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 构建更新字段和值
        update_fields = []
        values = []
        
        if title is not None:
            update_fields.append("title = ?")
            values.append(title)
        
        if site_url is not None:
            update_fields.append("site_url = ?")
            values.append(site_url)
            
        if description is not None:
            update_fields.append("description = ?")
            values.append(description)
            
        if category is not None:
            update_fields.append("category = ?")
            values.append(category)
        
        if not update_fields:
            conn.close()
            return False
            
        # 添加最后更新时间
        update_fields.append("last_updated = ?")
        values.append(datetime.now())
        
        # 添加feed_id
        values.append(feed_id)
        
        query = f"UPDATE feeds SET {', '.join(update_fields)} WHERE id = ?"
        
        try:
            cursor.execute(query, values)
            conn.commit()
            success = cursor.rowcount > 0
            return success
        except Exception as e:
            print(f"Error updating feed {feed_id}: {e}")
            return False
        finally:
            conn.close()
    
    def delete_feed(self, feed_id: int) -> bool:
        """
        删除RSS源
        
        Args:
            feed_id: Feed ID
            
        Returns:
            布尔值，表示删除是否成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 删除相关条目
            cursor.execute('DELETE FROM entries WHERE feed_id = ?', (feed_id,))
            
            # 删除Feed
            cursor.execute('DELETE FROM feeds WHERE id = ?', (feed_id,))
            
            conn.commit()
            success = cursor.rowcount > 0
            return success
        except Exception as e:
            print(f"Error deleting feed {feed_id}: {e}")
            return False
        finally:
            conn.close()
    
    def add_entries(self, entries: List[Entry]) -> int:
        """
        批量添加条目，返回添加的数量
        
        Args:
            entries: Entry对象列表
            
        Returns:
            新添加的条目数量
        """
        if not entries:
            return 0
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        count = 0
        for entry in entries:
            try:
                cursor.execute(
                    '''INSERT OR IGNORE INTO entries 
                    (feed_id, title, link, published_date, author, summary, content, read_status) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                    (entry.feed_id, entry.title, entry.link, entry.published_date, 
                     entry.author, entry.summary, entry.content, entry.read_status)
                )
                if cursor.rowcount > 0:
                    count += 1
            except sqlite3.IntegrityError:
                # 忽略重复条目
                pass
        
        conn.commit()
        conn.close()
        
        return count
    
    def get_feeds(self, category: Optional[str] = None) -> List[Feed]:
        """
        获取所有RSS源
        
        Args:
            category: 可选的分类过滤
            
        Returns:
            Feed对象列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = 'SELECT * FROM feeds'
        params = []
        
        if category:
            query += ' WHERE category = ?'
            params.append(category)
            
        query += ' ORDER BY title'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        feeds = []
        for row in rows:
            feed = Feed(
                id=row['id'],
                title=row['title'],
                url=row['url'],
                site_url=row['site_url'],
                description=row['description'],
                last_updated=datetime.fromisoformat(row['last_updated']) if row['last_updated'] else datetime.now(),
                category=row['category'] if row['category'] else ""
            )
            feeds.append(feed)
        
        conn.close()
        return feeds
    
    def get_feed_by_id(self, feed_id: int) -> Optional[Feed]:
        """
        根据ID获取Feed
        
        Args:
            feed_id: Feed ID
            
        Returns:
            Feed对象，如果不存在则返回None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM feeds WHERE id = ?', (feed_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
            
        feed = Feed(
            id=row['id'],
            title=row['title'],
            url=row['url'],
            site_url=row['site_url'],
            description=row['description'],
            last_updated=datetime.fromisoformat(row['last_updated']) if row['last_updated'] else None,
            category=row['category']
        )
        
        conn.close()
        return feed
    
    def get_feeds_by_url(self, url: str) -> List[Feed]:
        """
        根据URL获取Feed
        
        Args:
            url: Feed URL
            
        Returns:
            Feed对象列表，如果不存在则返回空列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM feeds WHERE url = ?', (url,))
        rows = cursor.fetchall()
        
        feeds = []
        for row in rows:
            feed = Feed(
                id=row['id'],
                title=row['title'],
                url=row['url'],
                site_url=row['site_url'],
                description=row['description'],
                last_updated=datetime.fromisoformat(row['last_updated']) if row['last_updated'] else None,
                category=row['category']
            )
            feeds.append(feed)
        
        conn.close()
        return feeds
    
    def get_entries(self, feed_id: Optional[int] = None, limit: int = 100, 
                   offset: int = 0, unread_only: bool = False) -> List[Entry]:
        """
        获取条目，可按feed_id过滤
        
        Args:
            feed_id: Feed ID（可选）
            limit: 返回条目数量限制
            offset: 分页偏移量
            unread_only: 是否只返回未读条目
            
        Returns:
            Entry对象列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = 'SELECT * FROM entries'
        params = []
        
        conditions = []
        if feed_id is not None:
            conditions.append('feed_id = ?')
            params.append(feed_id)
        
        if unread_only:
            conditions.append('read_status = 0')
        
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        
        query += ' ORDER BY published_date DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        entries = []
        for row in rows:
            entry = Entry(
                id=row['id'],
                feed_id=row['feed_id'],
                title=row['title'],
                link=row['link'],
                published_date=datetime.fromisoformat(row['published_date']) if row['published_date'] else datetime.now(),
                author=row['author'],
                summary=row['summary'],
                content=row['content'],
                read_status=bool(row['read_status'])
            )
            entries.append(entry)
        
        conn.close()
        return entries
    
    def get_entry_by_id(self, entry_id: int) -> Optional[Entry]:
        """
        通过ID获取条目
        
        Args:
            entry_id: 条目ID
            
        Returns:
            Entry对象，如果不存在则返回None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM entries WHERE id = ?', (entry_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
            
        entry = Entry(
            id=row['id'],
            feed_id=row['feed_id'],
            title=row['title'],
            link=row['link'],
            published_date=datetime.fromisoformat(row['published_date']) if row['published_date'] else datetime.now(),
            author=row['author'],
            summary=row['summary'],
            content=row['content'],
            read_status=bool(row['read_status'])
        )
        
        conn.close()
        return entry
    
    def update_feed_timestamp(self, feed_id: int) -> bool:
        """
        更新RSS源的最后更新时间
        
        Args:
            feed_id: Feed ID
            
        Returns:
            布尔值，表示更新是否成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'UPDATE feeds SET last_updated = ? WHERE id = ?',
                (datetime.now(), feed_id)
            )
            
            conn.commit()
            success = cursor.rowcount > 0
            return success
        except Exception as e:
            print(f"Error updating feed timestamp {feed_id}: {e}")
            return False
        finally:
            conn.close()
    
    def mark_entry_as_read(self, entry_id: int) -> bool:
        """
        将条目标记为已读
        
        Args:
            entry_id: 条目ID
            
        Returns:
            布尔值，表示更新是否成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'UPDATE entries SET read_status = 1 WHERE id = ?',
                (entry_id,)
            )
            
            conn.commit()
            success = cursor.rowcount > 0
            return success
        except Exception as e:
            print(f"Error marking entry as read {entry_id}: {e}")
            return False
        finally:
            conn.close()
    
    def mark_all_entries_as_read(self, feed_id: Optional[int] = None) -> int:
        """
        将所有条目标记为已读
        
        Args:
            feed_id: 可选的Feed ID，如果提供则只标记该Feed的条目
            
        Returns:
            标记为已读的条目数量
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'UPDATE entries SET read_status = 1 WHERE read_status = 0'
        params = []
        
        if feed_id is not None:
            query += ' AND feed_id = ?'
            params.append(feed_id)
            
        try:
            cursor.execute(query, params)
            conn.commit()
            count = cursor.rowcount
            return count
        except Exception as e:
            print(f"Error marking entries as read: {e}")
            return 0
        finally:
            conn.close()
    
    def get_unread_count(self, feed_id: Optional[int] = None) -> int:
        """
        获取未读条目数量
        
        Args:
            feed_id: 可选的Feed ID，如果提供则只计算该Feed的未读条目
            
        Returns:
            未读条目数量
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT COUNT(*) FROM entries WHERE read_status = 0'
        params = []
        
        if feed_id is not None:
            query += ' AND feed_id = ?'
            params.append(feed_id)
            
        cursor.execute(query, params)
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
        
    def get_entry_count(self, feed_id: Optional[int] = None) -> int:
        """
        获取条目总数
        
        Args:
            feed_id: 可选的Feed ID，如果提供则只计算该Feed的条目
            
        Returns:
            条目总数
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT COUNT(*) FROM entries'
        params = []
        
        if feed_id is not None:
            query += ' WHERE feed_id = ?'
            params.append(feed_id)
            
        cursor.execute(query, params)
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
        
    def get_feed_count(self) -> int:
        """
        获取RSS源总数
        
        Returns:
            RSS源总数
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM feeds')
        count = cursor.fetchone()[0]
        
        conn.close()
        return count 