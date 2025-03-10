"""
Agent 基类模块
定义 Agent 的基本接口和生命周期
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..memory import AgentMemory, Message


logger = logging.getLogger(__name__)


class Agent(ABC):
    """
    Agent 基类，定义所有 Agent 的通用接口
    """
    
    def __init__(
        self,
        name: str = None,
        description: str = None,
        system_prompt: str = None,
        memory: Optional[AgentMemory] = None,
        max_steps: int = 10,
        verbose: bool = False,
    ):
        """
        初始化 Agent
        
        Args:
            name (str, optional): Agent 名称. Defaults to None.
            description (str, optional): Agent 描述. Defaults to None.
            system_prompt (str, optional): 系统提示词. Defaults to None.
            memory (Optional[AgentMemory], optional): 记忆系统. Defaults to None.
            max_steps (int, optional): 最大步数. Defaults to 10.
            verbose (bool, optional): 是否详细输出. Defaults to False.
        """
        self.id = str(uuid.uuid4())
        self.name = name or f"Agent-{self.id[:8]}"
        self.description = description or "一个通用的智能助手"
        self.system_prompt = system_prompt or "你是一个有用的智能助手。"
        self.memory = memory or AgentMemory(system_prompt=self.system_prompt)
        self.max_steps = max_steps
        self.verbose = verbose
        self.step_count = 0
        self._initialized = False
    
    def initialize(self) -> None:
        """初始化 Agent，在首次运行时调用"""
        if self._initialized:
            return
        
        logger.info(f"初始化 Agent: {self.name}")
        self._setup()
        self._initialized = True
    
    @abstractmethod
    def _setup(self) -> None:
        """设置 Agent，子类需要实现"""
        pass
    
    def reset(self) -> None:
        """重置 Agent 状态"""
        logger.info(f"重置 Agent: {self.name}")
        self.memory.reset()
        self.step_count = 0
    
    @abstractmethod
    def step(self, input_data: Any) -> Any:
        """
        执行一步操作
        
        Args:
            input_data (Any): 输入数据
            
        Returns:
            Any: 步骤结果
        """
        pass
    
    def run(
        self, 
        task: str, 
        inputs: Optional[Dict[str, Any]] = None, 
        reset: bool = True,
        stream: bool = False
    ) -> Any:
        """
        运行 Agent 完成任务
        
        Args:
            task (str): 任务描述
            inputs (Optional[Dict[str, Any]], optional): 额外输入. Defaults to None.
            reset (bool, optional): 是否重置状态. Defaults to True.
            stream (bool, optional): 是否流式输出. Defaults to False.
            
        Returns:
            Any: 任务结果
        """
        if reset:
            self.reset()
        
        if not self._initialized:
            self.initialize()
        
        logger.info(f"运行 Agent: {self.name} - 任务: {task}")
        
        # 记录任务
        self.memory.add_task(task, inputs)
        
        # 执行步骤
        result = None
        start_time = time.time()
        
        try:
            while self.step_count < self.max_steps:
                self.step_count += 1
                
                if self.verbose:
                    logger.info(f"执行步骤 {self.step_count}/{self.max_steps}")
                
                step_result = self.step(task)
                
                if stream:
                    yield step_result
                
                # 检查是否完成
                if self._is_task_complete(step_result):
                    result = step_result
                    break
            
            if self.step_count >= self.max_steps:
                logger.warning(f"达到最大步数 {self.max_steps}，任务未完成")
                result = self._handle_max_steps_reached(task)
        
        except Exception as e:
            logger.error(f"Agent 运行出错: {e}")
            result = self._handle_error(e)
        
        finally:
            duration = time.time() - start_time
            logger.info(f"Agent 运行完成，耗时: {duration:.2f}秒，步数: {self.step_count}")
            
            if not stream:
                return result
    
    @abstractmethod
    def _is_task_complete(self, step_result: Any) -> bool:
        """
        判断任务是否完成
        
        Args:
            step_result (Any): 步骤结果
            
        Returns:
            bool: 是否完成
        """
        pass
    
    def _handle_max_steps_reached(self, task: str) -> Any:
        """
        处理达到最大步数的情况
        
        Args:
            task (str): 任务描述
            
        Returns:
            Any: 处理结果
        """
        return {
            "status": "incomplete",
            "reason": f"达到最大步数 {self.max_steps}",
            "partial_result": self.memory.get_summary()
        }
    
    def _handle_error(self, error: Exception) -> Any:
        """
        处理运行错误
        
        Args:
            error (Exception): 错误信息
            
        Returns:
            Any: 处理结果
        """
        return {
            "status": "error",
            "error": str(error),
            "partial_result": self.memory.get_summary()
        }
    
    def add_message(self, role: str, content: str) -> None:
        """
        添加消息到记忆
        
        Args:
            role (str): 消息角色
            content (str): 消息内容
        """
        self.memory.add_message(Message(role=role, content=content))
    
    def get_messages(self) -> List[Dict[str, str]]:
        """
        获取所有消息
        
        Returns:
            List[Dict[str, str]]: 消息列表
        """
        return self.memory.get_messages()
    
    def get_summary(self) -> str:
        """
        获取记忆摘要
        
        Returns:
            str: 记忆摘要
        """
        return self.memory.get_summary()
    
    def __call__(self, task: str, **kwargs) -> Any:
        """
        调用 Agent
        
        Args:
            task (str): 任务描述
            
        Returns:
            Any: 任务结果
        """
        return self.run(task, **kwargs) 