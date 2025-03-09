import os
from typing import Dict, List, Optional, Union, Iterator
import openai
from loguru import logger

from .base_llm import BaseLLM

class OpenAILLM(BaseLLM):
    """OpenAI LLM实现类"""
    
    def __init__(self, 
                model_name: str = "gpt-3.5-turbo", 
                api_key: Optional[str] = None,
                api_base: Optional[str] = None,
                temperature: float = 0.7,
                max_tokens: int = 1024):
        """初始化OpenAI LLM
        
        Args:
            model_name: 模型名称
            api_key: OpenAI API密钥，默认从环境变量获取
            api_base: OpenAI API基础URL，默认使用官方API
            temperature: 温度参数，控制生成的随机性
            max_tokens: 最大生成token数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 设置API密钥
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            
        # 设置API基础URL
        if api_base:
            openai.api_base = api_base
            
        logger.info(f"Initialized OpenAI LLM with model: {model_name}")
        
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        try:
            # 合并参数
            params = {
                "model": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            params.update(kwargs)
            
            # 创建消息
            messages = [{"role": "user", "content": prompt}]
            
            # 调用API
            response = openai.ChatCompletion.create(
                messages=messages,
                **params
            )
            
            # 提取生成的文本
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            return f"生成失败: {str(e)}"
            
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """流式生成文本
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Returns:
            迭代器，每次返回生成的一部分文本
        """
        try:
            # 合并参数
            params = {
                "model": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True
            }
            params.update(kwargs)
            
            # 创建消息
            messages = [{"role": "user", "content": prompt}]
            
            # 调用API
            response = openai.ChatCompletion.create(
                messages=messages,
                **params
            )
            
            # 流式返回生成的文本
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.get("content"):
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error streaming text with OpenAI: {e}")
            yield f"生成失败: {str(e)}" 