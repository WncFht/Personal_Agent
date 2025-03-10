from typing import Dict, Any, Iterator, Optional
from loguru import logger

from .component import Component
from ..llm.base_llm import BaseLLM
from ..llm.openai_llm import OpenAILLM
from ..llm.huggingface_llm import HuggingFaceLLM
from ..llm.tiny_llm import TinyLLM
from ..llm.deepseek_llm import DeepSeekLLM

class LLMManager(Component):
    """LLM管理器，负责管理和使用大语言模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化LLM管理器
        
        Args:
            config: 组件配置
        """
        super().__init__(config)
        self.llm = None
    
    def initialize(self) -> None:
        """初始化LLM管理器"""
        # 创建LLM
        self._create_llm()
        
        # 注册到依赖容器
        self.container.register("llm_manager", self)
        
        # 注册事件处理器
        self.register_event_handlers()
        
        logger.info("LLM管理器初始化完成")
    
    def _create_llm(self) -> None:
        """创建LLM实例"""
        llm_type = self.config.get("llm_type", "tiny")
        model_id = self.config.get("llm_model_id", "models/tiny_llm_sft_92m")
        
        # 根据LLM类型选择合适的LLM实现
        if llm_type == "tiny":
            self.llm = TinyLLM(
                model_path=model_id,
                device=self.config.get("device", "cpu"),
                temperature=0.7,
                system_prompt=self.config.get("system_prompt", "你是一个有用的AI助手。")
            )
        elif llm_type == "openai":
            self.llm = OpenAILLM(
                model_name=self.config.get("openai_model", "gpt-3.5-turbo"),
                api_key=self.config.get("openai_api_key", ""),
                temperature=0.7
            )
        elif llm_type == "deepseek":
            self.llm = DeepSeekLLM(
                api_key=self.config.get("deepseek_api_key", ""),
                model=self.config.get("deepseek_model", "deepseek-chat"),
                base_url=self.config.get("deepseek_base_url", "https://api.deepseek.com"),
                temperature=0.7,
                system_prompt=self.config.get("system_prompt", "你是一个有用的AI助手。")
            )
        elif llm_type == "huggingface":
            # 使用HuggingFace模型
            self.llm = HuggingFaceLLM(
                model_path=model_id,
                device=self.config.get("device", "cpu"),
                temperature=0.7,
                system_prompt=self.config.get("system_prompt", "你是一个有用的AI助手。")
            )
        else:
            # 默认使用TinyLLM
            logger.warning(f"未知的LLM类型: {llm_type}，使用TinyLLM作为默认值")
            self.llm = TinyLLM(
                model_path=model_id,
                device=self.config.get("device", "cpu"),
                temperature=0.7,
                system_prompt=self.config.get("system_prompt", "你是一个有用的AI助手。")
            )
            
        logger.info(f"初始化LLM: {model_id}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本
        
        Args:
            prompt: 提示词
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        return self.llm.generate(prompt, **kwargs)
    
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """流式生成文本
        
        Args:
            prompt: 提示词
            **kwargs: 其他参数
            
        Returns:
            生成的文本流
        """
        return self.llm.generate_stream(prompt, **kwargs)
    
    def register_event_handlers(self) -> None:
        """注册事件处理器"""
        # 监听配置更新事件
        self.event_system.subscribe("config_updated", self._on_config_updated)
    
    def _on_config_updated(self, config: Dict[str, Any]) -> None:
        """配置更新事件处理器
        
        Args:
            config: 更新后的配置
        """
        # 检查是否需要更新LLM
        if (config.get("llm_type") != self.config.get("llm_type") or
            config.get("llm_model_id") != self.config.get("llm_model_id") or
            config.get("device") != self.config.get("device") or
            config.get("system_prompt") != self.config.get("system_prompt") or
            config.get("openai_api_key") != self.config.get("openai_api_key") or
            config.get("openai_model") != self.config.get("openai_model") or
            config.get("deepseek_api_key") != self.config.get("deepseek_api_key") or
            config.get("deepseek_model") != self.config.get("deepseek_model") or
            config.get("deepseek_base_url") != self.config.get("deepseek_base_url")):
            
            logger.info("配置已更改，重新初始化LLM")
            
            # 更新配置
            self.config.update(config)
            
            # 清理旧的LLM资源
            if hasattr(self.llm, 'cleanup') and callable(self.llm.cleanup):
                self.llm.cleanup()
            
            # 重新创建LLM
            self._create_llm()
            
            # 发布LLM更新事件
            self.event_system.publish("llm_updated", llm=self.llm)
    
    def cleanup(self) -> None:
        """清理资源"""
        if hasattr(self.llm, 'cleanup') and callable(self.llm.cleanup):
            self.llm.cleanup() 