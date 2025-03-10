from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from loguru import logger

from ...utils.event_system import event_system
from ...utils.dependency_container import container

class Component(ABC):
    """RAG系统基础组件接口"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化组件
        
        Args:
            config: 组件配置
        """
        self.config = config
        self.event_system = event_system
        self.container = container
        logger.debug(f"{self.__class__.__name__} 组件初始化")
    
    @abstractmethod
    def initialize(self) -> None:
        """初始化组件，加载资源等"""
        pass
    
    def get_dependency(self, name: str, default: Optional[Any] = None) -> Any:
        """获取依赖
        
        Args:
            name: 依赖名称
            default: 默认值
            
        Returns:
            依赖实例
        """
        return self.container.get_or_default(name, default)
    
    def register_event_handlers(self) -> None:
        """注册事件处理器，子类可重写此方法"""
        pass
    
    def cleanup(self) -> None:
        """清理资源，子类可重写此方法"""
        pass 