# RSS-RAG 模块化架构

本目录包含RSS-RAG系统的模块化架构实现，遵循单一职责原则和依赖注入模式，提高代码的可维护性和可扩展性。

## 架构概述

模块化架构由以下几个主要部分组成：

1. **组件系统**：所有功能模块都继承自`Component`基类，提供统一的初始化和清理接口。
2. **依赖注入**：使用`DependencyContainer`管理组件依赖，减少硬编码依赖。
3. **事件系统**：使用`EventSystem`实现组件间的松耦合通信。
4. **系统管理器**：`SystemManager`负责初始化和协调所有组件。

## 主要组件

- **ConfigManager**：配置管理器，负责加载、保存和更新系统配置。
- **TextProcessor**：文本处理器，负责文本分块等操作。
- **EmbeddingManager**：嵌入模型管理器，负责文本向量化。
- **LLMManager**：LLM管理器，负责管理和使用大语言模型。
- **RetrievalManager**：检索管理器，负责文档检索。
- **RAGManager**：RAG管理器，负责协调各组件完成RAG任务。
- **SystemManager**：系统管理器，负责初始化和协调所有组件。

## 使用方法

### 基本使用

```python
from src.rag.core.system_manager import SystemManager

# 创建系统管理器
system = SystemManager()

# 初始化系统
system.initialize()

# 从RSS数据库加载数据
system.load_from_rss_db("data/rss.db", days=30)

# 回答问题
answer = system.answer("Python有什么特点？")
print(answer)

# 流式回答
for token in system.answer_stream("人工智能的发展趋势是什么？"):
    print(token, end="", flush=True)

# 关闭系统
system.shutdown()
```

### 更新配置

```python
# 更新配置
system.update_config({
    "llm_type": "openai",
    "openai_model": "gpt-4",
    "openai_api_key": "your-api-key",
    "use_query_enhancement": True
})
```

### 处理RSS条目

```python
from src.rss.models import Entry
from datetime import datetime

# 创建条目
entry = Entry(
    id=1,
    feed_id=1,
    title="测试标题",
    link="https://example.com",
    published_date=datetime.now(),
    author="测试作者",
    summary="测试摘要",
    content="测试内容",
    read_status=False
)

# 处理条目
system.process_entries([entry])
```

## 事件系统

组件可以通过事件系统进行通信，例如：

```python
from src.utils.event_system import event_system

# 订阅事件
def on_config_updated(config):
    print(f"配置已更新: {config}")

event_system.subscribe("config_updated", on_config_updated)

# 发布事件
event_system.publish("config_updated", config={"key": "value"})
```

## 依赖注入

组件可以通过依赖容器获取其他组件，例如：

```python
from src.utils.dependency_container import container

# 获取组件
llm_manager = container.get("llm_manager")
answer = llm_manager.generate("你好，世界！")
```

## 扩展架构

要添加新的组件，只需继承`Component`基类并实现必要的方法：

```python
from src.rag.core.component import Component

class MyComponent(Component):
    def __init__(self, config):
        super().__init__(config)
        
    def initialize(self):
        # 初始化组件
        self.container.register("my_component", self)
        
    def register_event_handlers(self):
        # 注册事件处理器
        self.event_system.subscribe("some_event", self._on_some_event)
        
    def _on_some_event(self, **kwargs):
        # 处理事件
        pass
        
    def cleanup(self):
        # 清理资源
        pass
```

然后在`SystemManager`中初始化新组件：

```python
def initialize(self):
    # ...
    
    # 初始化自定义组件
    my_component = MyComponent(config)
    my_component.initialize()
    self.components.append(my_component)
    
    # ...
``` 