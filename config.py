# config.py
import os

# 🧠 知识库路径
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
DB_DIR = os.path.join(os.path.dirname(__file__), "vector_store")

# 🦙 模型设置
EMBEDDING_MODEL = "all-minilm"       # 用于文本向量化
LLM_MODEL = "llama3.1:8b"            # Ollama 模型名

# 🪄 分块策略
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# 📝 Prompt 模板
QA_PROMPT = """你是一个英语学习助手。
请根据提供的文档回答用户的问题。
要求：
1. 然后给出相关的原文英文示例
2. 如果问题与文档不相关，礼貌提示"抱歉，我无法回答此问题"。
3. 回答时尽量详细易懂。

参考文档：
{context}

问题：{query}"""
