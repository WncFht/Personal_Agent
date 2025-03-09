import os
from typing import Dict, List, Optional, Union, Iterator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from loguru import logger

from .base_llm import BaseLLM

class HuggingFaceLLM(BaseLLM):
    """HuggingFace LLM实现类"""
    
    def __init__(self, 
                model_path: str,
                device: str = "cuda",
                max_length: int = 2048,
                temperature: float = 0.7,
                top_p: float = 0.9):
        """初始化HuggingFace LLM
        
        Args:
            model_path: 模型路径或huggingface模型ID
            device: 设备类型，'cpu'或'cuda'
            max_length: 最大生成长度
            temperature: 温度参数，控制生成的随机性
            top_p: top-p采样参数
        """
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
        # 加载模型和分词器
        logger.info(f"Loading LLM from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"Initialized HuggingFace LLM with model: {model_path}")
        
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
                "max_length": self.max_length,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": True
            }
            params.update(kwargs)
            
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **params
                )
                
            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 移除输入提示
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
                
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error generating text with HuggingFace: {e}")
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
                "max_length": self.max_length,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": True
            }
            params.update(kwargs)
            
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # 创建流式生成器
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
            
            # 在新线程中生成文本
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                **params
            )
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # 流式返回生成的文本
            for text in streamer:
                yield text
                
        except Exception as e:
            logger.error(f"Error streaming text with HuggingFace: {e}")
            yield f"生成失败: {str(e)}" 