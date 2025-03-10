import os
import json
import time
import threading
import copy
from typing import Dict, Any, Optional, List, Set, Callable, Union, Literal
from pathlib import Path
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, create_model
from dotenv import load_dotenv

from .component import Component
from ..config import RAGConfig

# 加载环境变量
load_dotenv()

class ConfigValidator(BaseModel):
    """配置验证模型，用于验证配置参数的有效性"""
    
    # 基础配置
    base_dir: str = Field(default="data/rag_db", description="基础数据目录")
    device: str = Field(default="cpu", description="设备类型，可选值: cpu, cuda")
    
    # 模型配置
    llm_type: str = Field(default="tiny", description="LLM类型，可选值: tiny, openai, huggingface, deepseek")
    llm_model_id: str = Field(default="models/tiny_llm_sft_92m", description="LLM模型ID")
    embedding_model_id: str = Field(default="models/bge-base-zh-v1.5", description="Embedding模型ID")
    reranker_model_id: str = Field(default="models/bge-reranker-base", description="重排序模型ID")
    system_prompt: str = Field(default="你是一个有用的AI助手。", description="系统提示词")
    
    # DeepSeek配置
    deepseek_api_key: str = Field(default="", description="DeepSeek API密钥")
    deepseek_model: str = Field(default="deepseek-chat", description="DeepSeek模型名称")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", description="DeepSeek API基础URL")
    
    # OpenAI配置
    openai_api_key: str = Field(default="", description="OpenAI API密钥")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI模型名称")
    
    # 文本分割配置
    chunk_size: int = Field(default=500, ge=50, le=2000, description="文本分块大小")
    chunk_overlap: int = Field(default=50, ge=0, le=500, description="文本分块重叠大小")
    use_model_for_splitting: bool = Field(default=False, description="是否使用模型进行分句")
    sentence_splitter_model: str = Field(default="damo/nlp_bert_document-segmentation_chinese-base", description="分句模型")
    
    # 检索配置
    top_k: int = Field(default=3, ge=1, le=20, description="检索结果数量")
    search_weights: Optional[Dict[str, float]] = Field(default=None, description="混合检索权重")
    use_reranker: bool = Field(default=True, description="是否使用重排序")
    
    # 查询增强配置
    use_query_enhancement: bool = Field(default=False, description="是否使用查询增强")
    
    # 缓存配置
    use_cache: bool = Field(default=True, description="是否使用缓存")
    cache_dir: str = Field(default="data/cache", description="缓存目录")
    cache_size: int = Field(default=1000, ge=100, le=10000, description="缓存大小")
    cache_ttl: int = Field(default=3600, ge=60, le=86400, description="缓存过期时间（秒）")
    
    # RSS特定配置
    max_history_days: int = Field(default=30, ge=1, le=365, description="RSS条目最大保留天数")
    update_interval: int = Field(default=3600, ge=300, le=86400, description="RSS更新间隔（秒）")
    
    # 并行处理配置
    use_parallel_processing: bool = Field(default=True, description="是否使用并行处理")
    max_workers: Optional[int] = Field(default=None, description="最大工作线程数，None表示使用CPU核心数")
    
    @classmethod
    def get_field_descriptions(cls) -> Dict[str, str]:
        """获取所有字段的描述信息"""
        return {
            field_name: field.description
            for field_name, field in cls.model_fields.items()
        }
    
    @classmethod
    def get_field_constraints(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有字段的约束信息"""
        constraints = {}
        for field_name, field in cls.model_fields.items():
            field_constraints = {}
            
            # 获取字段类型
            field_type = field.annotation
            
            # 获取字段约束
            if hasattr(field, 'ge') and field.ge is not None:
                field_constraints["min"] = field.ge
            if hasattr(field, 'le') and field.le is not None:
                field_constraints["max"] = field.le
            if hasattr(field, 'pattern') and field.pattern is not None:
                field_constraints["pattern"] = field.pattern
            
            # 对于枚举类型，获取允许的值
            if hasattr(field_type, '__origin__') and field_type.__origin__ is Literal:
                field_constraints["allowed_values"] = list(field_type.__args__)
            
            constraints[field_name] = field_constraints
        
        return constraints


class ConfigManager(Component):
    """配置管理器，用于管理系统配置
    
    特性:
    1. 分层配置系统: 默认配置 < 配置文件 < 用户配置 < 环境变量
    2. 配置验证: 使用Pydantic验证配置参数的有效性
    3. 热重载配置: 支持在运行时重新加载配置
    4. 配置变更通知: 支持配置变更时的回调函数
    """
    
    def __init__(
        self, 
        config_path: str = "config/app_config.json",
        user_config_path: str = "config/user_config.json",
        auto_reload: bool = True,
        reload_interval: int = 30
    ):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径
            user_config_path: 用户配置文件路径
            auto_reload: 是否自动重新加载配置
            reload_interval: 重新加载配置的间隔时间（秒）
        """
        # 配置文件路径
        self.config_path = config_path
        self.user_config_path = user_config_path
        
        # 默认配置
        self.default_config = self._get_default_config()
        
        # 当前配置
        self.config = self.default_config.copy()
        
        # 配置文件最后修改时间
        self.last_modified_time = 0
        
        # 配置变更回调函数
        self.change_callbacks: List[Callable[[Dict[str, Any], Dict[str, Any]], None]] = []
        
        # 自动重新加载配置
        self.auto_reload = auto_reload
        self.reload_interval = reload_interval
        self.reload_thread = None
        self.stop_event = threading.Event()
        
        # 加载配置
        self.reload_config()
        
        # 启动自动重新加载线程
        if self.auto_reload:
            self._start_auto_reload()
        
        # 初始化组件
        super().__init__(self.config)
    
    def initialize(self) -> None:
        """初始化组件，实现抽象方法
        
        此方法主要用于注册事件处理器和将自身注册到依赖容器
        """
        # 注册事件处理器
        self.register_event_handlers()
        
        # 将自身注册到依赖容器
        self.container.register("config_manager", self)
        self.container.register("config", self.config)
        
        logger.info("配置管理器初始化完成")
    
    def register_event_handlers(self) -> None:
        """注册事件处理器"""
        # 注册配置更新事件处理器
        self.event_system.subscribe("request_config_update", self._handle_config_update_request)
        self.event_system.subscribe("request_config_reset", self._handle_config_reset_request)
    
    def _handle_config_update_request(self, config_updates: Dict[str, Any], save_to_user_config: bool = True) -> None:
        """处理配置更新请求事件
        
        Args:
            config_updates: 要更新的配置
            save_to_user_config: 是否保存到用户配置文件
        """
        success = self.update_config(config_updates, save_to_user_config)
        self.event_system.publish("config_update_result", success=success, config=self.config)
    
    def _handle_config_reset_request(self, keys: Optional[List[str]] = None) -> None:
        """处理配置重置请求事件
        
        Args:
            keys: 要重置的配置项列表，None表示重置所有配置
        """
        success = self.reset_to_default(keys)
        self.event_system.publish("config_reset_result", success=success, config=self.config)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        # 从ConfigValidator中获取默认值
        return {
            field_name: field.default
            for field_name, field in ConfigValidator.__fields__.items()
        }
    
    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(file_path):
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件 {file_path} 失败: {e}")
            return {}
    
    def _get_env_config(self) -> Dict[str, Any]:
        """从环境变量中获取配置"""
        env_config = {}
        
        # 环境变量前缀
        prefix = "RSS_RAG_"
        
        # 遍历所有环境变量
        for key, value in os.environ.items():
            # 检查是否是配置环境变量
            if key.startswith(prefix):
                # 移除前缀并转换为小写
                config_key = key[len(prefix):].lower()
                
                # 转换值类型
                if config_key in ConfigValidator.__fields__:
                    field = ConfigValidator.__fields__[config_key]
                    field_type = field.type_
                    
                    try:
                        # 根据字段类型转换值
                        if field_type == bool:
                            env_config[config_key] = value.lower() in ('true', 'yes', '1', 'y')
                        elif field_type == int:
                            env_config[config_key] = int(value)
                        elif field_type == float:
                            env_config[config_key] = float(value)
                        elif field_type == Dict:
                            env_config[config_key] = json.loads(value)
                        else:
                            env_config[config_key] = value
                    except Exception as e:
                        logger.warning(f"环境变量 {key} 转换失败: {e}")
        
        return env_config
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置"""
        try:
            # 使用Pydantic验证配置
            validated_config = ConfigValidator(**config).dict()
            return validated_config
        except ValidationError as e:
            logger.error(f"配置验证失败: {e}")
            # 返回默认配置
            return self.default_config.copy()
    
    def reload_config(self) -> bool:
        """重新加载配置
        
        Returns:
            bool: 配置是否发生变化
        """
        # 保存旧配置
        old_config = copy.deepcopy(self.config)
        
        # 加载配置文件
        file_config = self._load_config_file(self.config_path)
        
        # 加载用户配置文件
        user_config = self._load_config_file(self.user_config_path)
        
        # 从环境变量中获取配置
        env_config = self._get_env_config()
        
        # 合并配置
        merged_config = {**self.default_config, **file_config, **user_config, **env_config}
        
        # 验证配置
        validated_config = self._validate_config(merged_config)
        
        # 更新配置
        self.config = validated_config
        
        # 更新最后修改时间
        self._update_last_modified_time()
        
        # 检查配置是否发生变化
        config_changed = old_config != self.config
        
        # 如果配置发生变化，触发回调函数
        if config_changed:
            self._trigger_change_callbacks(old_config, self.config)
        
        return config_changed
    
    def _update_last_modified_time(self):
        """更新配置文件最后修改时间"""
        try:
            # 获取配置文件的最后修改时间
            config_mtime = 0
            user_config_mtime = 0
            
            if os.path.exists(self.config_path):
                config_mtime = os.path.getmtime(self.config_path)
            
            if os.path.exists(self.user_config_path):
                user_config_mtime = os.path.getmtime(self.user_config_path)
            
            # 使用两个文件中较新的修改时间
            self.last_modified_time = max(config_mtime, user_config_mtime)
        except Exception as e:
            logger.error(f"获取配置文件修改时间失败: {e}")
    
    def _check_config_file_changed(self) -> bool:
        """检查配置文件是否发生变化"""
        try:
            # 检查应用配置文件是否变化
            if os.path.exists(self.config_path):
                current_mtime = os.path.getmtime(self.config_path)
                if current_mtime > self.last_modified_time:
                    logger.debug(f"应用配置文件已更改: {self.config_path}")
                    return True
            
            # 检查用户配置文件是否变化
            if os.path.exists(self.user_config_path):
                current_mtime = os.path.getmtime(self.user_config_path)
                if current_mtime > self.last_modified_time:
                    logger.debug(f"用户配置文件已更改: {self.user_config_path}")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"检查配置文件变化失败: {e}")
            return False
    
    def _auto_reload_loop(self):
        """自动重新加载配置的循环"""
        logger.info(f"启动配置自动重载线程，检查间隔: {self.reload_interval}秒")
        last_reload_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                # 检查配置文件是否发生变化
                if self._check_config_file_changed():
                    logger.info(f"检测到配置文件变化，重新加载配置 (距上次重载: {current_time - last_reload_time:.1f}秒)")
                    config_changed = self.reload_config()
                    if config_changed:
                        logger.info("配置已成功更新")
                    else:
                        logger.info("配置文件已重新加载，但内容未变化")
                    last_reload_time = current_time
            except Exception as e:
                logger.error(f"自动重新加载配置失败: {e}")
            
            # 等待一段时间
            self.stop_event.wait(self.reload_interval)
    
    def _start_auto_reload(self):
        """启动自动重新加载线程"""
        if self.reload_thread is not None and self.reload_thread.is_alive():
            return
        
        self.stop_event.clear()
        self.reload_thread = threading.Thread(target=self._auto_reload_loop, daemon=True)
        self.reload_thread.start()
    
    def _stop_auto_reload(self):
        """停止自动重新加载线程"""
        if self.reload_thread is not None and self.reload_thread.is_alive():
            self.stop_event.set()
            self.reload_thread.join(timeout=1)
            self.reload_thread = None
    
    def register_change_callback(self, callback: Callable[[Dict[str, Any], Dict[str, Any]], None]):
        """注册配置变更回调函数
        
        Args:
            callback: 回调函数，接收两个参数：旧配置和新配置
        """
        if callback not in self.change_callbacks:
            self.change_callbacks.append(callback)
    
    def unregister_change_callback(self, callback: Callable[[Dict[str, Any], Dict[str, Any]], None]):
        """取消注册配置变更回调函数
        
        Args:
            callback: 回调函数
        """
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
    
    def _trigger_change_callbacks(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """触发配置变更回调函数
        
        Args:
            old_config: 旧配置
            new_config: 新配置
        """
        for callback in self.change_callbacks:
            try:
                callback(old_config, new_config)
            except Exception as e:
                logger.error(f"配置变更回调函数执行失败: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return copy.deepcopy(self.config)
    
    def get_rag_config(self) -> RAGConfig:
        """获取RAG配置对象"""
        return RAGConfig(**self.config)
    
    def update_config(self, config_updates: Dict[str, Any], save_to_user_config: bool = True) -> bool:
        """更新配置
        
        Args:
            config_updates: 要更新的配置
            save_to_user_config: 是否保存到用户配置文件
        
        Returns:
            bool: 配置是否成功更新
        """
        # 保存旧配置
        old_config = copy.deepcopy(self.config)
        
        # 合并配置
        merged_config = {**self.config, **config_updates}
        
        # 验证配置
        try:
            validated_config = self._validate_config(merged_config)
        except Exception as e:
            logger.error(f"配置更新验证失败: {e}")
            return False
        
        # 更新配置
        self.config = validated_config
        
        # 如果需要保存到用户配置文件
        if save_to_user_config:
            self._save_user_config()
        
        # 触发配置变更回调函数
        self._trigger_change_callbacks(old_config, self.config)
        
        return True
    
    def _save_user_config(self):
        """保存用户配置到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.user_config_path), exist_ok=True)
            
            # 加载现有用户配置
            existing_user_config = self._load_config_file(self.user_config_path)
            
            # 找出与默认配置不同的配置项
            user_config = {}
            for key, value in self.config.items():
                if key in self.default_config and value != self.default_config[key]:
                    user_config[key] = value
            
            # 合并现有用户配置和新配置
            merged_user_config = {**existing_user_config, **user_config}
            
            # 保存到文件
            with open(self.user_config_path, 'w', encoding='utf-8') as f:
                json.dump(merged_user_config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"用户配置已保存到 {self.user_config_path}")
            
            # 更新最后修改时间，避免触发自动重载
            self._update_last_modified_time()
        except Exception as e:
            logger.error(f"保存用户配置失败: {e}")
    
    def reset_to_default(self, keys: Optional[List[str]] = None) -> bool:
        """重置配置到默认值
        
        Args:
            keys: 要重置的配置项列表，None表示重置所有配置
        
        Returns:
            bool: 配置是否成功重置
        """
        # 保存旧配置
        old_config = copy.deepcopy(self.config)
        
        # 如果没有指定要重置的配置项，重置所有配置
        if keys is None:
            self.config = self.default_config.copy()
        else:
            # 只重置指定的配置项
            for key in keys:
                if key in self.default_config:
                    self.config[key] = self.default_config[key]
        
        # 保存到用户配置文件
        self._save_user_config()
        
        # 触发配置变更回调函数
        self._trigger_change_callbacks(old_config, self.config)
        
        return True
    
    def get_config_schema(self) -> Dict[str, Any]:
        """获取配置模式信息，用于UI展示
        
        Returns:
            Dict[str, Any]: 配置模式信息
        """
        schema = {}
        
        # 获取字段描述
        descriptions = ConfigValidator.get_field_descriptions()
        
        # 获取字段约束
        constraints = ConfigValidator.get_field_constraints()
        
        # 构建模式信息
        for field_name, field in ConfigValidator.model_fields.items():
            field_schema = {
                "type": str(field.annotation),
                "default": field.default,
                "description": descriptions.get(field_name, ""),
                "constraints": constraints.get(field_name, {})
            }
            schema[field_name] = field_schema
        
        return schema
    
    def __del__(self):
        """析构函数，停止自动重新加载线程"""
        self._stop_auto_reload() 