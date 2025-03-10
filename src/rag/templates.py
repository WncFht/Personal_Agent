"""
RAG提示词模板
"""

# 原始RAG提示词模板
ORIGINAL_RAG_PROMPT_TEMPLATE = """参考信息：
{context}
---
问题：
{question}
---
请根据上述参考信息回答问题。如果参考信息不足以回答问题，请说明无法回答。回答要简洁、准确，并尽可能基于参考信息。
"""

# 增强RAG提示词模板
ENHANCED_RAG_PROMPT_TEMPLATE = """参考信息：
{context}
---
我的问题或指令：
{question}
---
我的回答：
{answer}
---
请根据上述参考信息回答和我的问题或指令，修正我的回答。前面的参考信息和我的回答可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你修正的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复。
你修正的回答:""" 