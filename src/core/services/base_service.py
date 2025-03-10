"""
服务基类模块
定义所有服务的基类和接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type


class Service(ABC):
    """服务基类"""
    
    @abstractmethod
    def initialize(self, service_manager) -> None:
        """
        初始化服务
        
        Args:
            service_manager: 服务管理器实例
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """关闭服务"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """服务名称"""
        pass
    
    @property
    def dependencies(self) -> List[Type['Service']]:
        """
        服务依赖列表
        
        Returns:
            List[Type[Service]]: 依赖的服务类型列表
        """
        return []
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取服务状态
        
        Returns:
            Dict[str, Any]: 服务状态信息
        """
        return {
            "name": self.name,
            "status": "running"
        } 