# app_with_memory.py - 带上下文记忆功能的RAG应用

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.llms import OllamaLLM
from retriever_enhanced import EnhancedRetriever
from conversation_memory import ConversationMemory, get_conversation_memory
import gradio as gr
from config import LLM_MODEL
import time
from typing import Tuple, Dict, Any, List
import json


class ConversationRAGApp:
    """带上下文记忆的RAG应用"""

    def __init__(self):
        # 1️⃣ 检索器
        self.retriever = EnhancedRetriever()

        # 2️⃣ 对话记忆
        self.memory = get_conversation_memory()

        # 3️⃣ LLM
        self.llm = OllamaLLM(model=LLM_MODEL)

        # 4️⃣ 增强的Prompt模板（支持上下文记忆）
        self.prompt_with_context = PromptTemplate(
            template="""你是一个专业的英语学习助手。请根据提供的文档和对话历史，全面回答用户的问题。

**对话历史上下文：**
{conversation_context}

**检索到的学习资料：**
{retrieved_docs}

**当前用户问题：**
{query}

**回答要求：**
1. 仔细分析对话历史，理解用户的背景和之前的讨论
2. 结合检索到的资料给出准确的语法解释
3. 如果用户在追问，请基于之前的回答进行补充说明
4. 提供具体的英文示例和用法说明
5. 保持回答的连贯性和一致性
6. 如果发现了之前可能的错误，请主动纠正和澄清

**回答：**""",
            input_variables=["conversation_context", "retrieved_docs", "query"]
        )

        # 5️⃣ 处理链
        self.qa_chain = (
            {
                "conversation_context": lambda q: self.memory.get_context_for_query(q),
                "retrieved_docs": lambda q: self._get_retrieved_docs(q),
                "query": RunnablePassthrough()
            }
            | self.prompt_with_context
            | self.llm
        )

    def _get_retrieved_docs(self, query: str) -> str:
        """获取检索到的文档"""
        try:
            docs = self.retriever.get_relevant_documents(query, method="enhanced")
            if not docs:
                return "未找到相关的英语学习资料。"

            formatted_docs = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', f'文档片段{i}')
                content = doc.page_content.strip()[:800]  # 限制长度
                formatted_docs.append(f"[资料{i} - {source}]:\n{content}")

            return "\n\n".join(formatted_docs)
        except Exception as e:
            return f"检索文档时出错: {str(e)}"

    def chat_with_memory(self, query: str, use_memory: bool = True) -> Tuple[str, Dict[str, Any]]:
        """带记忆的对话"""
        if not query.strip():
            return "❌ 请输入问题", {}

        try:
            start_time = time.time()

            # 获取对话上下文
            conversation_context = ""
            if use_memory and self.memory.conversation_history:
                conversation_context = self.memory.get_context_for_query(query)

            # 检索相关文档
            docs = self.retriever.get_relevant_documents(query, method="enhanced")
            retrieval_quality = self.retriever.analyze_retrieval_quality(query, docs)

            # 如果启用记忆且有上下文
            if use_memory and conversation_context:
                # 使用带上下文的处理链
                response = self.qa_chain.invoke(query)
            else:
                # 简单回答（无记忆）
                response = self._simple_answer(query, docs)

            response_time = time.time() - start_time

            # 保存对话记录
            if use_memory:
                self.memory.add_conversation_turn(
                    user_query=query,
                    ai_response=response,
                    metadata={
                        "response_time": response_time,
                        "retrieval_quality": retrieval_quality.get("quality_score", 0),
                        "num_docs": len(docs)
                    },
                    retrieved_docs=[doc.metadata.get("source", "unknown") for doc in docs]
                )

            # 添加记忆信息
            memory_info = self._format_memory_info(use_memory, conversation_context)
            final_response = response + "\n\n" + memory_info

            return final_response, {
                "retrieval_quality": retrieval_quality,
                "response_time": response_time,
                "num_retrieved_docs": len(docs),
                "memory_enabled": use_memory,
                "conversation_length": len(self.memory.conversation_history) if use_memory else 0
            }

        except Exception as e:
            return f"❌ 错误: {str(e)}", {}

    def _simple_answer(self, query: str, docs) -> str:
        """简单回答（不使用记忆）"""
        if not docs:
            return "抱歉，没有找到相关的英语学习资料。请尝试其他问题。"

        # 简单的Prompt
        simple_prompt = PromptTemplate(
            template="""请根据提供的英语学习资料回答用户问题：

资料：
{docs}

问题：{query}

请给出详细准确的回答。""",
            input_variables=["docs", "query"]
        )

        # 格式化文档
        docs_text = "\n\n".join([f"资料{i+1}: {doc.page_content[:500]}..." for i, doc in enumerate(docs)])

        chain = simple_prompt | self.llm
        return chain.invoke({"docs": docs_text, "query": query})

    def _format_memory_info(self, use_memory: bool, context: str) -> str:
        """格式化记忆信息"""
        if not use_memory:
            return "💭 **记忆功能**: 已关闭"

        info_parts = ["💭 **记忆功能**: 已启用"]

        if context:
            info_parts.append(f"📝 **对话历史**: 已参考 {len(self.memory.conversation_history)} 轮对话")

        # 显示最近的话题
        if self.memory.conversation_history:
            recent_topics = self.memory.conversation_history[-1].keywords
            if recent_topics:
                info_parts.append(f"🏷️ **相关话题**: {', '.join(recent_topics[:3])}")

        return "\n".join(info_parts)

    def get_conversation_stats(self) -> Dict[str, Any]:
        """获取对话统计信息"""
        return self.memory.get_conversation_stats()

    def clear_memory(self) -> str:
        """清空对话记忆"""
        self.memory.clear_history()
        return "✅ 对话历史已清空"

    def export_conversation(self, format: str = "json") -> str:
        """导出对话历史"""
        try:
            return self.memory.export_conversation(format)
        except Exception as e:
            return f"导出失败: {str(e)}"


# 创建应用实例
app = ConversationRAGApp()


# Gradio界面
with gr.Blocks(title="🧠 智能记忆RAG英语助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧠 智能记忆RAG英语助手

        > 💡 **记住每一次对话** - 具有上下文记忆的智能英语学习助手

        🎯 **核心特性**:
        - 📚 本地知识库检索
        - 🧠 对话上下文记忆
        - 🔄 连贯的多轮对话
        - 📊 智能话题追踪
        - 💾 对话历史管理
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            query = gr.Textbox(
                label="你的问题",
                placeholder="例如：刚才提到的现在完成时，能给更多例子吗？",
                lines=3
            )

            with gr.Row():
                use_memory = gr.Checkbox(label="🧠 启用记忆功能", value=True)
                memory_status = gr.Textbox(
                    label="记忆状态",
                    value="已启用 - 可记住对话历史",
                    interactive=False,
                    scale=2
                )

            with gr.Row():
                submit = gr.Button("💬 对话", variant="primary")
                clear_conversation = gr.Button("🗑️ 清空记忆", variant="secondary")
                export_btn = gr.Button("📥 导出对话", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("### 📈 对话统计")
            with gr.Row():
                total_conversations = gr.Number(label="总对话数", precision=0, interactive=False)
                current_session = gr.Number(label="当前会话", precision=0, interactive=False)

            with gr.Row():
                avg_response_length = gr.Number(label="平均回答长度", precision=0, interactive=False)

            gr.Markdown("### 🏷️ 常见话题")
            topics_display = gr.Textbox(
                label="最近话题",
                lines=3,
                interactive=False,
                placeholder="暂无话题记录"
            )

    # 主对话区域
    with gr.Accordion("💬 智能回答", open=True):
        output = gr.Markdown(label="回答内容")

    # 对话详情区域
    with gr.Accordion("📊 详细信息", open=False):
        with gr.Row():
            quality_score = gr.Number(label="检索质量", precision=1, interactive=False)
            response_time = gr.Number(label="响应时间(秒)", precision=2, interactive=False)
            docs_count = gr.Number(label="检索文档数", precision=0, interactive=False)

    # 导出区域
    with gr.Accordion("📋 对话导出", open=False):
        export_format = gr.Radio(
            choices=[("JSON格式", "json"), ("文本格式", "txt")],
            value="json",
            label="导出格式"
        )
        export_output = gr.Textbox(
            label="导出内容",
            lines=10,
            interactive=False,
            show_copy_button=True
        )

    # 绑定事件
    def process_query(query_text, memory_enabled):
        if not query_text.strip():
            return "❌ 请输入问题", 0, 0, 0, 0, 0, ""

        response, metadata = app.chat_with_memory(query_text, memory_enabled)

        # 更新统计信息
        stats = app.get_conversation_stats()
        total_conv = stats["total_conversations"]
        current_sess = stats["current_session_length"]
        avg_length = stats["avg_response_length"]

        # 格式化话题
        topics = stats["most_discussed_topics"]
        topics_text = ", ".join([f"{topic['topic']}({topic['count']}次)" for topic in topics[:3]])

        # 更新元数据
        quality = metadata.get("retrieval_quality", {}).get("quality_score", 0)
        time_taken = metadata.get("response_time", 0)
        docs_num = metadata.get("num_retrieved_docs", 0)

        # 更新记忆状态
        memory_status_text = "已启用" if memory_enabled else "已关闭"
        if memory_enabled and metadata.get("conversation_length", 0) > 0:
            memory_status_text += f" - {metadata['conversation_length']}轮对话"

        return (
            response,  # output
            total_conv,  # total_conversations
            current_sess,  # current_session
            avg_length,  # avg_response_length
            quality,  # quality_score
            time_taken,  # response_time
            docs_num,  # docs_count
            topics_text,  # topics_display
            memory_status_text  # memory_status
        )

    def clear_conversation_handler():
        message = app.clear_memory()
        stats = app.get_conversation_stats()
        return (
            "✅ 对话历史已清空，开始新的对话吧！",
            stats["total_conversations"],
            stats["current_session_length"],
            stats["avg_response_length"],
            0, 0, 0,
            "暂无话题记录",
            "已启用 - 新会话"
        )

    def export_conversation_handler(export_fmt):
        try:
            content = app.export_conversation(export_fmt)
            return content
        except Exception as e:
            return f"导出失败: {str(e)}"

    def update_memory_checkbox(memory_enabled):
        status = "已启用 - 可记住对话历史" if memory_enabled else "已关闭"
        return status

    # 绑定事件
    submit.click(
        fn=process_query,
        inputs=[query, use_memory],
        outputs=[
            output, total_conversations, current_session, avg_response_length,
            quality_score, response_time, docs_count, topics_display, memory_status
        ]
    )

    clear_conversation.click(
        fn=clear_conversation_handler,
        outputs=[
            output, total_conversations, current_session, avg_response_length,
            quality_score, response_time, docs_count, topics_display, memory_status
        ]
    )

    export_btn.click(
        fn=export_conversation_handler,
        inputs=[export_format],
        outputs=[export_output]
    )

    use_memory.change(
        fn=update_memory_checkbox,
        inputs=[use_memory],
        outputs=[memory_status]
    )

    # 示例问题
    gr.Examples(
        examples=[
            ["现在完成时的用法是什么？"],
            ["能给一些具体例子吗？"],
            ["那它和过去时有什么区别？"],
            ["在什么情况下使用现在完成进行时？"],
            ["帮我总结一下现在完成时的要点"]
        ],
        inputs=[query]
    )

    # 使用说明
    with gr.Accordion("📖 使用说明", open=False):
        gr.Markdown(
            """
            ### 🧠 记忆功能说明

            **启用记忆**：
            - 助手会记住之前的对话内容
            - 可以进行连贯的多轮对话
            - 能够理解追问和上下文关联
            - 自动追踪讨论的话题

            **关闭记忆**：
            - 每次对话都是独立的
            - 不会记住之前的讨论内容
            - 适合询问不相关的问题

            **对话管理**：
            - 使用"清空记忆"按钮重新开始
            - 可以导出对话历史进行复习
            - 统计信息帮助了解学习进度

            💡 **建议**：学习相关语法时保持记忆开启，讨论不同话题时可以关闭记忆。
            """
        )


# 启动应用
if __name__ == "__main__":
    print("🧠 智能记忆RAG英语助手启动中...")
    demo.launch(share=False, server_name="0.0.0.0", server_port=7865)
    print("✅ 应用已启动: http://localhost:7865")