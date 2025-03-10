from typing import Dict, List, Callable, Any
from loguru import logger

class EventSystem:
    """事件系统，用于组件间通信"""
    
    def __init__(self):
        """初始化事件系统"""
        self._subscribers: Dict[str, List[Callable]] = {}
        logger.debug("事件系统初始化")
    
    def subscribe(self, event_name: str, callback: Callable) -> None:
        """订阅事件
        
        Args:
            event_name: 事件名称
            callback: 回调函数，当事件触发时调用
        """
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        
        self._subscribers[event_name].append(callback)
        logger.debug(f"订阅事件: {event_name}")
    
    def unsubscribe(self, event_name: str, callback: Callable) -> None:
        """取消订阅事件
        
        Args:
            event_name: 事件名称
            callback: 回调函数
        """
        if event_name in self._subscribers and callback in self._subscribers[event_name]:
            self._subscribers[event_name].remove(callback)
            logger.debug(f"取消订阅事件: {event_name}")
    
    def publish(self, event_name: str, **kwargs) -> None:
        """发布事件
        
        Args:
            event_name: 事件名称
            **kwargs: 事件参数
        """
        logger.debug(f"发布事件: {event_name}")
        if event_name in self._subscribers:
            for callback in self._subscribers[event_name]:
                try:
                    callback(**kwargs)
                except Exception as e:
                    logger.error(f"事件处理器出错: {e}")
    
    def clear(self) -> None:
        """清除所有订阅"""
        self._subscribers.clear()
        logger.debug("清除所有事件订阅")

# 全局事件系统实例
event_system = EventSystem() 