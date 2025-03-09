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
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # 明确指定数据类型
            device_map=self.device,  # 分布到特定设备上
            trust_remote_code=True  # 允许加载远程代码
        )
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,  # 分词器标识符
            trust_remote_code=True,
            padding_side="left"  # 设置填充方向为左侧
        )
        
        # 确保tokenizer有正确的特殊token
        self._setup_tokenizer()
        
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
        
        # 检测模型类型
        self.model_type = self._detect_model_type()
        logger.info(f"检测到模型类型: {self.model_type}")
        
        logger.info(f"TinyLLM初始化成功: {model_path}")
    
    def _setup_tokenizer(self):
        """设置tokenizer的特殊token"""
        # 确保有pad_token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                # 如果没有pad_token但有eos_token，使用一个不同的token作为pad_token
                if hasattr(self.tokenizer, 'add_special_tokens'):
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                else:
                    # 如果无法添加特殊token，使用一个现有的token
                    logger.warning("无法添加PAD token，使用EOS token作为替代")
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # 如果连eos_token都没有，使用一个常见的token
                logger.warning("模型没有EOS token，使用默认token")
                self.tokenizer.pad_token = self.tokenizer.eos_token = '</s>'
    
    def _detect_model_type(self) -> str:
        """检测模型类型"""
        model_name = self.model_path.lower()
        
        if "qwen" in model_name:
            return "qwen"
        elif "chatglm" in model_name:
            return "chatglm"
        elif "llama" in model_name or "mistral" in model_name:
            return "llama"
        elif "baichuan" in model_name:
            return "baichuan"
        elif "tiny" in model_name:
            return "tiny"
        else:
            return "default"
        
    def _format_prompt(self, prompt: str) -> str:
        """格式化提示词，添加系统提示和对话格式
        
        Args:
            prompt: 用户输入的提示词
            
        Returns:
            str: 格式化后的提示词
        """
        # 根据检测到的模型类型使用合适的格式
        if self.model_type == "qwen":
            input_txt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif self.model_type == "chatglm":
            input_txt = f"[gMASK]sop<assistant>\n{prompt}"
        elif self.model_type == "llama":
            input_txt = f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        elif self.model_type == "baichuan":
            input_txt = f"<reserved_106>{self.system_prompt}<reserved_107>{prompt}<reserved_108>"
        elif self.model_type == "tiny":
            # TinyLLM特定格式
            input_txt = f"<|system|>\n{self.system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
        else:
            # 默认格式，适用于大多数模型
            input_txt = f"系统: {self.system_prompt}\n\n用户: {prompt}\n\n助手: "
            
        logger.debug(f"格式化后的提示词: {input_txt}...")
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
        inputs = self.tokenizer(input_txt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.model.device)
        
        # 确保有attention_mask
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"].to(self.model.device)
        else:
            attention_mask = torch.ones_like(input_ids).to(self.model.device)
        
        # 生成文本
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # 只保留新生成的部分
            generated_tokens = outputs[0, len(input_ids[0]):]
            
            # 解码输出
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
        except Exception as e:
            logger.error(f"生成文本时出错: {e}")
            # 如果生成失败，返回一个简单的回复
            return "抱歉，我无法回答这个问题。"
    
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
        inputs = self.tokenizer(input_txt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.model.device)
        
        # 确保有attention_mask
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"].to(self.model.device)
        else:
            attention_mask = torch.ones_like(input_ids).to(self.model.device)
        
        try:
            # 创建流式生成器
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)
            
            # 生成参数
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "streamer": streamer,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True
            }
            
            # 在后台线程中运行生成
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # 流式返回生成的文本
            for text in streamer:
                yield text
        except Exception as e:
            logger.error(f"流式生成文本时出错: {e}")
            yield "抱歉，我无法回答这个问题。" 