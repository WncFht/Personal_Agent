"""
Agent 记忆系统
管理 Agent 的记忆和状态
"""

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class Message:
    """消息类，表示一条对话消息"""
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }


@dataclass
class MemoryStep:
    """记忆步骤基类"""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class TaskStep(MemoryStep):
    """任务步骤，记录任务信息"""
    task: str
    inputs: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": "task",
            "timestamp": self.timestamp,
            "task": self.task,
            "inputs": self.inputs
        }


@dataclass
class ActionStep(MemoryStep):
    """行动步骤，记录 Agent 的行动"""
    action: str
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Any] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": "action",
            "timestamp": self.timestamp,
            "action": self.action,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error
        }


@dataclass
class ObservationStep(MemoryStep):
    """观察步骤，记录环境观察"""
    observation: Any
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": "observation",
            "timestamp": self.timestamp,
            "observation": self.observation,
            "source": self.source
        }


@dataclass
class ReflectionStep(MemoryStep):
    """反思步骤，记录 Agent 的思考"""
    thoughts: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": "reflection",
            "timestamp": self.timestamp,
            "thoughts": self.thoughts
        }


class AgentMemory:
    """Agent 记忆系统，管理 Agent 的记忆和状态"""
    
    def __init__(self, system_prompt: str = None):
        """
        初始化记忆系统
        
        Args:
            system_prompt (str, optional): 系统提示词. Defaults to None.
        """
        self.system_prompt = system_prompt or "你是一个有用的智能助手。"
        self.messages: List[Message] = []
        self.steps: List[MemoryStep] = []
        self.reset()
    
    def reset(self) -> None:
        """重置记忆"""
        self.messages = []
        self.steps = []
        # 添加系统提示词
        self.add_message(Message(role="system", content=self.system_prompt))
    
    def add_message(self, message: Message) -> None:
        """
        添加消息
        
        Args:
            message (Message): 消息对象
        """
        self.messages.append(message)
    
    def add_step(self, step: MemoryStep) -> None:
        """
        添加步骤
        
        Args:
            step (MemoryStep): 步骤对象
        """
        self.steps.append(step)
    
    def add_task(self, task: str, inputs: Optional[Dict[str, Any]] = None) -> None:
        """
        添加任务
        
        Args:
            task (str): 任务描述
            inputs (Optional[Dict[str, Any]], optional): 任务输入. Defaults to None.
        """
        self.add_step(TaskStep(task=task, inputs=inputs))
        self.add_message(Message(role="user", content=task))
    
    def add_action(
        self, 
        action: str, 
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Any] = None,
        error: Optional[str] = None
    ) -> None:
        """
        添加行动
        
        Args:
            action (str): 行动名称
            inputs (Optional[Dict[str, Any]], optional): 行动输入. Defaults to None.
            outputs (Optional[Any], optional): 行动输出. Defaults to None.
            error (Optional[str], optional): 错误信息. Defaults to None.
        """
        self.add_step(ActionStep(action=action, inputs=inputs, outputs=outputs, error=error))
        
        # 如果有错误，添加错误消息
        if error:
            self.add_message(Message(role="system", content=f"错误: {error}"))
    
    def add_observation(self, observation: Any, source: Optional[str] = None) -> None:
        """
        添加观察
        
        Args:
            observation (Any): 观察内容
            source (Optional[str], optional): 观察来源. Defaults to None.
        """
        self.add_step(ObservationStep(observation=observation, source=source))
        
        # 将观察转换为消息
        if isinstance(observation, str):
            content = observation
        else:
            try:
                content = json.dumps(observation, ensure_ascii=False, indent=2)
            except:
                content = str(observation)
        
        self.add_message(Message(role="system", content=f"观察: {content}"))
    
    def add_reflection(self, thoughts: str) -> None:
        """
        添加反思
        
        Args:
            thoughts (str): 思考内容
        """
        self.add_step(ReflectionStep(thoughts=thoughts))
        self.add_message(Message(role="assistant", content=thoughts))
    
    def get_messages(self) -> List[Dict[str, str]]:
        """
        获取所有消息
        
        Returns:
            List[Dict[str, str]]: 消息列表
        """
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]
    
    def get_recent_messages(self, n: int = 10) -> List[Dict[str, str]]:
        """
        获取最近的 n 条消息
        
        Args:
            n (int, optional): 消息数量. Defaults to 10.
            
        Returns:
            List[Dict[str, str]]: 消息列表
        """
        messages = self.messages[-n:] if n > 0 else self.messages
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    def get_steps(self) -> List[Dict[str, Any]]:
        """
        获取所有步骤
        
        Returns:
            List[Dict[str, Any]]: 步骤列表
        """
        return [step.to_dict() for step in self.steps]
    
    def get_recent_steps(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        获取最近的 n 个步骤
        
        Args:
            n (int, optional): 步骤数量. Defaults to 5.
            
        Returns:
            List[Dict[str, Any]]: 步骤列表
        """
        steps = self.steps[-n:] if n > 0 else self.steps
        return [step.to_dict() for step in steps]
    
    def get_summary(self) -> str:
        """
        获取记忆摘要
        
        Returns:
            str: 记忆摘要
        """
        if not self.steps:
            return "没有记忆。"
        
        # 获取任务
        tasks = [step for step in self.steps if isinstance(step, TaskStep)]
        task_summary = f"任务: {tasks[-1].task}" if tasks else "无任务"
        
        # 获取最近的行动
        actions = [step for step in self.steps if isinstance(step, ActionStep)]
        action_summary = "\n".join([
            f"- 行动: {step.action}" for step in actions[-3:]
        ]) if actions else "无行动"
        
        # 获取最近的观察
        observations = [step for step in self.steps if isinstance(step, ObservationStep)]
        observation_summary = "\n".join([
            f"- 观察: {step.observation}" for step in observations[-3:]
        ]) if observations else "无观察"
        
        return f"{task_summary}\n\n最近行动:\n{action_summary}\n\n最近观察:\n{observation_summary}" 