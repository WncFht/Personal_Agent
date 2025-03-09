import os
from typing import Dict, List, Optional, Union, Iterator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextIteratorStreamer
from threading import Thread
from loguru import logger

from .base_llm import BaseLLM

class TinyLLM(BaseLLM):
    """TinyLLM实现类，适用于小型本地模型"""
    
    def __init__(self, 
                model_path: str,
                device: str = "cuda",
                max_length: int = 2048,
                temperature: float = 0.7,
                top_p: float = 0.9,
                system_prompt: str = "你是一个有用的AI助手。"):
        """初始化TinyLLM
        
        Args:
            model_path: 模型路径或huggingface模型ID
            device: 设备类型，'cpu'或'cuda'
            max_length: 最大生成长度
            temperature: 温度参数，控制生成的随机性
            top_p: top-p采样参数
            system_prompt: 系统提示词
        """
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        
        # 加载模型和分词器
        logger.info(f"正在加载TinyLLM: {model_path}")
        
        # 从预训练模型加载因果语言模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,  # 模型标识符
            torch_dtype="auto",  # 自动选择张量类型
            device_map=self.device,  # 分布到特定设备上
            trust_remote_code=True  # 允许加载远程代码
        )
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,  # 分词器标识符
            use_fast=False,
            trust_remote_code=True
        )
        
        # 加载配置文件
        self.config = AutoConfig.from_pretrained(
            self.model_path,  # 配置文件标识符
            trust_remote_code=True  # 允许加载远程代码
        )

        # 如果使用CPU，转换为float精度
        if self.device == "cpu":
            self.model.float()
        
        # 设置模型为评估模式
        self.model.eval()
        
        logger.info(f"TinyLLM初始化成功: {model_path}")
        
    def _format_prompt(self, prompt: str) -> str:
        """格式化提示词，添加系统提示和对话格式
        
        Args:
            prompt: 用户输入的提示词
            
        Returns:
            str: 格式化后的提示词
        """
        # 尝试检测模型类型并使用合适的格式
        model_name = self.model_path.lower()
        
        # Qwen/ChatGLM格式
        if "qwen" in model_name or "chatglm" in model_name:
            input_txt = "\n".join(["<|system|>", self.system_prompt.strip(), 
                                "<|user|>", prompt.strip(), 
                                "<|assistant|>"]).strip() + "\n"
        # Llama格式
        elif "llama" in model_name:
            input_txt = f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        # Baichuan格式
        elif "baichuan" in model_name:
            input_txt = f"<reserved_106>{self.system_prompt}<reserved_107>{prompt}<reserved_108>"
        # 默认格式
        else:
            input_txt = f"系统: {self.system_prompt}\n\n用户: {prompt}\n\n助手: "
            
        return input_txt
        
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        # 更新参数
        max_length = kwargs.get("max_length", self.max_length)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        
        # 格式化提示词
        input_txt = self._format_prompt(prompt)
        
        # 编码输入
        model_inputs = self.tokenizer(input_txt, return_tensors="pt").to(self.model.device)
        
        # 生成文本
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        
        # 只保留新生成的部分
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # 解码输出
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
    
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """流式生成文本
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Returns:
            迭代器，每次返回生成的一部分文本
        """
        # 更新参数
        max_length = kwargs.get("max_length", self.max_length)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        
        # 格式化提示词
        input_txt = self._format_prompt(prompt)
        
        # 编码输入
        inputs = self.tokenizer(input_txt, return_tensors="pt").to(self.model.device)
        
        # 创建流式生成器
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        
        # 生成参数
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "max_new_tokens": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        }
        
        # 在后台线程中运行生成
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 流式返回生成的文本
        for text in streamer:
            yield text 