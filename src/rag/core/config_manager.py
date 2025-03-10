import os
import json
from typing import Dict, Any, Optional
from loguru import logger

from .component import Component

class ConfigManager(Component):
    """配置管理器，用于管理系统配置"""
    
    def __init__(self, config_path: str = "config/app_config.json"):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.default_config = {
            "base_dir": "data/rag_db",
            "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu",
            "llm_type": "tiny",
            "llm_model_id": "models/tiny_llm_sft_92m",
            "embedding_model_id": "models/bge-base-zh-v1.5",
            "system_prompt": "你是一个有用的AI助手，擅长回答关于科技和人工智能的问题。",
            "chunk_size": 800,
            "chunk_overlap": 100,
            "top_k": 5,
            "use_reranker": True,
            "use_query_enhancement": True,
            "use_model_for_splitting": False,
            "sentence_splitter_model": "damo/nlp_bert_document-segmentation_chinese-base",
            "reranker_model_id": "models/bge-reranker-base",
            "openai_api_key": "",
            "openai_model": "gpt-3.5-turbo",
            "deepseek_api_key": "",
            "deepseek_model": "deepseek-chat",
            "deepseek_base_url": "https://api.deepseek.com",
            "cache_dir": "data/cache",
            "cache_size": 1000,
            "cache_ttl": 3600,
            "max_history_days": 30,
            "update_interval": 3600,
            "use_parallel_processing": True,
            "max_workers": None
        }
        self.config = self.default_config.copy()
        super().__init__(self.config)
    
    def initialize(self) -> None:
        """初始化配置管理器，加载配置文件"""
        self.load_config()
        self.register_event_handlers()
        
        # 注册自身到依赖容器
        self.container.register("config_manager", self)
        self.container.register("config", self.config)
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件
        
        Returns:
            配置字典
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # 更新配置，保留默认值
                    self.config.update(loaded_config)
                    logger.info(f"从 {self.config_path} 加载配置")
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
        else:
            logger.warning(f"配置文件 {self.config_path} 不存在，使用默认配置")
            # 确保配置目录存在
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            # 保存默认配置
            self.save_config()
        
        return self.config
    
    def save_config(self) -> None:
        """保存配置到文件"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            logger.info(f"配置已保存到 {self.config_path}")
            
            # 发布配置更新事件
            self.event_system.publish("config_updated", config=self.config)
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """更新配置
        
        Args:
            new_config: 新配置
        """
        self.config.update(new_config)
        self.save_config()
        logger.info("配置已更新")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项
        
        Args:
            key: 配置项键名
            default: 默认值
            
        Returns:
            配置项值
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """设置配置项
        
        Args:
            key: 配置项键名
            value: 配置项值
        """
        self.config[key] = value
        logger.debug(f"设置配置项: {key}={value}")
    
    def register_event_handlers(self) -> None:
        """注册事件处理器"""
        # 这里可以注册对配置变更的响应 