"""
异常基类模块
定义系统中使用的所有异常的基类
"""

class RSSRAGException(Exception):
    """RSS-RAG 异常基类"""
    
    def __init__(self, message, code=None, details=None):
        """
        初始化异常
        
        Args:
            message (str): 错误消息
            code (str, optional): 错误代码. Defaults to None.
            details (dict, optional): 错误详情. Defaults to None.
        """
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)
    
    def __str__(self):
        """字符串表示"""
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message
    
    def to_dict(self):
        """转换为字典表示"""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details
        }


class ServiceNotFoundError(RSSRAGException):
    """服务未找到异常"""
    
    def __init__(self, service_name):
        """
        初始化异常
        
        Args:
            service_name (str): 服务名称
        """
        super().__init__(
            message=f"Service not found: {service_name}",
            code="SERVICE_NOT_FOUND",
            details={"service_name": service_name}
        )


class ServiceInitializationError(RSSRAGException):
    """服务初始化异常"""
    
    def __init__(self, service_name, reason=None):
        """
        初始化异常
        
        Args:
            service_name (str): 服务名称
            reason (str, optional): 失败原因. Defaults to None.
        """
        details = {"service_name": service_name}
        if reason:
            details["reason"] = reason
            
        super().__init__(
            message=f"Failed to initialize service: {service_name}" + (f" - {reason}" if reason else ""),
            code="SERVICE_INIT_ERROR",
            details=details
        ) 