"""
服务管理器模块
负责管理所有服务的生命周期
"""

import logging
from typing import Dict, Type, TypeVar, Optional, List, Any

from ..exceptions import ServiceNotFoundError, ServiceInitializationError
from .base_service import Service

# 类型变量，用于泛型服务类型
T = TypeVar('T', bound=Service)

logger = logging.getLogger(__name__)


class ServiceManager:
    """服务管理器，负责管理所有服务的生命周期"""
    
    def __init__(self):
        """初始化服务管理器"""
        self._services: Dict[Type[Service], Service] = {}
        self._initialized = False
        self._initializing = False
    
    def register(self, service_type: Type[Service], service_instance: Service) -> 'ServiceManager':
        """
        注册服务
        
        Args:
            service_type (Type[Service]): 服务类型
            service_instance (Service): 服务实例
            
        Returns:
            ServiceManager: 服务管理器实例，用于链式调用
        """
        if self._initialized:
            logger.warning(f"Registering service {service_type.__name__} after initialization")
            
        self._services[service_type] = service_instance
        return self
    
    def get(self, service_type: Type[T]) -> T:
        """
        获取服务实例
        
        Args:
            service_type (Type[T]): 服务类型
            
        Returns:
            T: 服务实例
            
        Raises:
            ServiceNotFoundError: 服务未注册时抛出
        """
        if service_type not in self._services:
            raise ServiceNotFoundError(service_type.__name__)
        return self._services[service_type]
    
    def initialize(self) -> None:
        """
        初始化所有服务
        
        Raises:
            ServiceInitializationError: 初始化失败时抛出
        """
        if self._initialized:
            logger.info("Services already initialized")
            return
            
        if self._initializing:
            logger.warning("Service initialization already in progress")
            return
            
        self._initializing = True
        
        try:
            # 按依赖顺序初始化服务
            initialized_services = set()
            
            while len(initialized_services) < len(self._services):
                progress_made = False
                
                for service_type, service in self._services.items():
                    if service_type in initialized_services:
                        continue
                        
                    # 检查依赖是否已初始化
                    dependencies_met = True
                    for dependency in service.dependencies:
                        if dependency not in self._services:
                            raise ServiceInitializationError(
                                service.name,
                                f"Dependency {dependency.__name__} not registered"
                            )
                        if dependency not in initialized_services:
                            dependencies_met = False
                            break
                            
                    if dependencies_met:
                        logger.info(f"Initializing service: {service.name}")
                        try:
                            service.initialize(self)
                            initialized_services.add(service_type)
                            progress_made = True
                        except Exception as e:
                            raise ServiceInitializationError(
                                service.name,
                                str(e)
                            ) from e
                
                if not progress_made and len(initialized_services) < len(self._services):
                    # 如果没有进展，可能存在循环依赖
                    uninitialized = [s.__class__.__name__ for s in self._services.values() 
                                    if s.__class__ not in initialized_services]
                    raise ServiceInitializationError(
                        "multiple services",
                        f"Possible circular dependency detected: {', '.join(uninitialized)}"
                    )
            
            self._initialized = True
            logger.info("All services initialized successfully")
            
        finally:
            self._initializing = False
    
    def shutdown(self) -> None:
        """关闭所有服务"""
        if not self._initialized:
            logger.info("No services to shutdown")
            return
            
        # 按依赖顺序的逆序关闭服务
        # 首先构建依赖图
        dependency_graph = {s_type: set(s.dependencies) for s_type, s in self._services.items()}
        
        # 拓扑排序
        sorted_services = self._topological_sort(dependency_graph)
        
        # 逆序关闭
        for service_type in reversed(sorted_services):
            if service_type in self._services:
                service = self._services[service_type]
                logger.info(f"Shutting down service: {service.name}")
                try:
                    service.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down service {service.name}: {e}")
                    
        self._initialized = False
        logger.info("All services shut down")
    
    def _topological_sort(self, graph: Dict[Type[Service], set]) -> List[Type[Service]]:
        """
        对服务进行拓扑排序
        
        Args:
            graph (Dict[Type[Service], set]): 依赖图
            
        Returns:
            List[Type[Service]]: 排序后的服务类型列表
        """
        result = []
        visited = set()
        temp_visited = set()
        
        def visit(node):
            if node in temp_visited:
                # 检测到循环依赖，但此时我们已经注册了服务，所以只记录警告
                logger.warning(f"Circular dependency detected involving {node.__name__}")
                return
            if node in visited:
                return
                
            temp_visited.add(node)
            
            for neighbor in graph.get(node, set()):
                if neighbor in self._services:  # 只访问已注册的服务
                    visit(neighbor)
                    
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)
        
        for node in graph:
            if node not in visited:
                visit(node)
                
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取所有服务的状态
        
        Returns:
            Dict[str, Any]: 服务状态信息
        """
        return {
            "initialized": self._initialized,
            "services": [service.get_status() for service in self._services.values()]
        } 