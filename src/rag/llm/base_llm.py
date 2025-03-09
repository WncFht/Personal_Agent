from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

class BaseLLM(ABC):
    """LLM基础接口类"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> str:
        """流式生成文本
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Returns:
            迭代器，每次返回生成的一部分文本
        """
        pass 