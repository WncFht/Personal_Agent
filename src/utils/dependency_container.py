from typing import Dict, Any, Callable, Type, TypeVar, Optional
from loguru import logger

T = TypeVar('T')

class DependencyContainer:
    """依赖注入容器，用于管理组件依赖"""
    
    def __init__(self):
        """初始化依赖容器"""
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[..., Any]] = {}
        logger.debug("依赖容器初始化")
    
    def register(self, name: str, instance: Any) -> None:
        """注册一个实例
        
        Args:
            name: 实例名称
            instance: 实例对象
        """
        self._instances[name] = instance
        logger.debug(f"注册实例: {name}")
    
    def register_factory(self, name: str, factory: Callable[..., Any]) -> None:
        """注册一个工厂函数
        
        Args:
            name: 工厂名称
            factory: 工厂函数，用于创建实例
        """
        self._factories[name] = factory
        logger.debug(f"注册工厂: {name}")
    
    def get(self, name: str) -> Any:
        """获取实例
        
        Args:
            name: 实例名称
            
        Returns:
            实例对象
            
        Raises:
            KeyError: 如果实例不存在
        """
        if name in self._instances:
            return self._instances[name]
        
        if name in self._factories:
            instance = self._factories[name]()
            self._instances[name] = instance
            return instance
        
        raise KeyError(f"依赖 '{name}' 未注册")
    
    def get_or_default(self, name: str, default: Any = None) -> Any:
        """获取实例，如果不存在则返回默认值
        
        Args:
            name: 实例名称
            default: 默认值
            
        Returns:
            实例对象或默认值
        """
        try:
            return self.get(name)
        except KeyError:
            return default
    
    def clear(self) -> None:
        """清除所有注册的实例和工厂"""
        self._instances.clear()
        self._factories.clear()
        logger.debug("清除所有依赖")

# 全局依赖容器实例
container = DependencyContainer() 