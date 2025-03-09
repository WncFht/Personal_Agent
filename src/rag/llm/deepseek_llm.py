import os
import json
from typing import Dict, List, Optional, Union, Iterator
import requests
from loguru import logger

from .base_llm import BaseLLM

class DeepSeekLLM(BaseLLM):
    """DeepSeek API实现类，支持调用DeepSeek的API服务"""
    
    def __init__(self, 
                api_key: str,
                model: str = "deepseek-chat",
                base_url: str = "https://api.deepseek.com",
                max_tokens: int = 2048,
                temperature: float = 0.7,
                top_p: float = 0.9,
                system_prompt: str = "你是一个有用的AI助手。"):
        """初始化DeepSeekLLM
        
        Args:
            api_key: DeepSeek API密钥
            model: 模型名称，可选值: "deepseek-chat"(DeepSeek-V3), "deepseek-reasoner"(DeepSeek-R1)
            base_url: API基础URL
            max_tokens: 最大生成token数
            temperature: 温度参数，控制生成的随机性
            top_p: top-p采样参数
            system_prompt: 系统提示词
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        
        # 检查API密钥
        if not self.api_key:
            logger.warning("未提供DeepSeek API密钥，请确保在环境变量或配置中设置")
            self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        
        logger.info(f"初始化DeepSeekLLM: 模型={model}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        # 更新参数
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        
        # 构建消息
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # 构建请求数据
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False
        }
        
        # 设置请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 发送请求
        try:
            url = f"{self.base_url}/chat/completions"
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"DeepSeek API请求失败: 状态码={response.status_code}, 响应={response.text}"
                logger.error(error_msg)
                return f"API请求失败: {error_msg}"
            
            # 解析响应
            result = response.json()
            
            # 提取生成的文本
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"DeepSeek API响应格式异常: {result}")
                return "API响应格式异常"
            
        except Exception as e:
            logger.error(f"调用DeepSeek API时出错: {e}")
            return f"API调用出错: {str(e)}"
    
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """流式生成文本
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Returns:
            迭代器，每次返回生成的一部分文本
        """
        # 更新参数
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        
        # 构建消息
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # 构建请求数据
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True
        }
        
        # 设置请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 发送请求
        try:
            url = f"{self.base_url}/chat/completions"
            response = requests.post(url, headers=headers, json=data, stream=True, timeout=60)
            
            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"DeepSeek API请求失败: 状态码={response.status_code}, 响应={response.text}"
                logger.error(error_msg)
                yield f"API请求失败: {error_msg}"
                return
            
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    # 移除"data: "前缀
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        line = line[6:]
                    
                    # 跳过心跳消息
                    if line == "[DONE]":
                        break
                    
                    try:
                        # 解析JSON
                        chunk = json.loads(line)
                        
                        # 提取生成的文本
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析流式响应行: {line}")
                        continue
                    except Exception as e:
                        logger.error(f"处理流式响应时出错: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"调用DeepSeek API流式生成时出错: {e}")
            yield f"API调用出错: {str(e)}" 