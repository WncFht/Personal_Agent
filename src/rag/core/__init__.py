# 只导出组件类名，不导入具体实现
__all__ = [
    'Component',
    'ConfigManager',
    'TextProcessor',
    'EmbeddingManager',
    'LLMManager',
    'RetrievalManager',
    'RAGManager',
    'SystemManager'
]

# 延迟导入，避免循环依赖
from .component import Component

# 其他组件可以按需导入 