# app_enhanced.py - 增强版RAG应用

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.llms import OllamaLLM
from retriever_enhanced import EnhancedRetriever
import gradio as gr
from config import LLM_MODEL, QA_PROMPT
import time
from typing import Tuple, Dict, Any


class EnhancedRAGApp:
    """增强版RAG应用，支持多种检索策略和质量监控"""

    def __init__(self):
        # 1️⃣ 增强检索器
        self.retriever = EnhancedRetriever()

        # 2️⃣ LLM
        self.llm = OllamaLLM(model=LLM_MODEL)

        # 3️⃣ 增强的Prompt模板
        self.prompt = PromptTemplate(
            template=QA_PROMPT,
            input_variables=["context", "query"]
        )

        # 4️⃣ 创建多个处理链
        self._setup_chains()

    def _setup_chains(self):
        """设置不同的处理链"""
        # 基础QA链
        self.qa_chain = (
            {"context": lambda q: self._get_formatted_docs(q), "query": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )

        # 增强QA链（带检索质量分析）
        self.enhanced_qa_chain = self._create_enhanced_chain()

    def _get_formatted_docs(self, query: str, method: str = "enhanced") -> str:
        """获取并格式化检索到的文档"""
        docs = self.retriever.get_relevant_documents(query, method=method)
        if not docs:
            return "未找到相关文档。"

        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', f'文档片段{i}')
            content = doc.page_content.strip()
            formatted_docs.append(f"[文档{i} - {source}]:\n{content}")

        return "\n\n".join(formatted_docs)

    def _create_enhanced_chain(self):
        """创建增强的处理链"""
        def enhanced_process(query: str) -> Tuple[str, Dict[str, Any]]:
            start_time = time.time()

            # 检索文档
            docs = self.retriever.get_relevant_documents(query, method="enhanced")
            retrieval_quality = self.retriever.analyze_retrieval_quality(query, docs)

            # 如果检索质量太低，给出提示
            if retrieval_quality["quality_score"] < 30:
                return (
                    "❌ 抱歉，没有找到相关的学习资料。请尝试：\n"
                    "1. 使用更具体的关键词\n"
                    "2. 检查问题是否在英语学习范围内\n"
                    "3. 查看知识库中是否包含相关内容",
                    {"retrieval_quality": retrieval_quality, "response_time": time.time() - start_time}
                )

            # 生成回答
            formatted_docs = self._get_formatted_docs(query)
            response = self.qa_chain.invoke(query)

            # 添加检索质量信息
            quality_note = f"\n\n---\n📊 检索质量: {retrieval_quality['quality_score']:.1f}/100"
            if retrieval_quality['recommendations']:
                quality_note += f"\n💡 优化建议: {', '.join(retrieval_quality['recommendations'])}"

            return response + quality_note, {
                "retrieval_quality": retrieval_quality,
                "response_time": time.time() - start_time,
                "num_retrieved_docs": len(docs)
            }

        return enhanced_process

    def chat_with_agent(self, query: str, method: str = "enhanced") -> Tuple[str, Dict[str, Any]]:
        """与智能体对话"""
        if not query.strip():
            return "❌ 请输入问题", {}

        try:
            if method == "enhanced":
                # 使用增强链
                response, metadata = self.enhanced_qa_chain(query)
                return response, metadata
            else:
                # 使用指定方法
                start_time = time.time()
                docs = self.retriever.get_relevant_documents(query, method=method)
                retrieval_quality = self.retriever.analyze_retrieval_quality(query, docs)

                if not docs:
                    return "❌ 未找到相关文档", {"retrieval_quality": retrieval_quality}

                formatted_docs = self._get_formatted_docs(query, method)
                response = self.qa_chain.invoke(query)

                return response, {
                    "retrieval_quality": retrieval_quality,
                    "response_time": time.time() - start_time,
                    "num_retrieved_docs": len(docs)
                }

        except Exception as e:
            return f"❌ 错误: {str(e)}", {}

    def get_retrieval_debug_info(self, query: str, method: str = "enhanced") -> Dict[str, Any]:
        """获取检索调试信息"""
        docs = self.retriever.get_relevant_documents(query, method=method)
        return self.retriever.analyze_retrieval_quality(query, docs)


# 创建应用实例
app = EnhancedRAGApp()


# Gradio界面
with gr.Blocks(title="🎓 英语学习智能体 (增强版)", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎓 英语学习智能体 (增强版)

        > 💡 **RAG 2.0** - 基于增强检索的智能英语学习助手

        🚀 **新功能**:
        - 多种检索策略 (向量、MMR、混合)
        - 检索质量监控和分析
        - 智能文档重排序
        - 实时性能指标
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            query = gr.Textbox(
                label="你的问题",
                placeholder="例如：现在完成时的考点和用法",
                lines=3
            )
            method = gr.Radio(
                choices=[
                    ("🧠 智能增强", "enhanced"),
                    ("📊 向量相似性", "vector"),
                    ("🔄 多样性检索 (MMR)", "mmr"),
                    ("🔍 混合检索", "ensemble")
                ],
                value="enhanced",
                label="检索方法",
                info="选择不同的检索策略"
            )
            with gr.Row():
                submit = gr.Button("🚀 提问", variant="primary")
                debug_btn = gr.Button("🔍 检索调试", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("### 📈 检索统计")
            quality_score = gr.Number(label="质量评分", precision=1, interactive=False)
            response_time = gr.Number(label="响应时间(秒)", precision=2, interactive=False)
            doc_count = gr.Number(label="检索文档数", precision=0, interactive=False)

    # 主回答区域
    output = gr.Markdown(label="智能回答", height=300)

    # 调试信息区域
    with gr.Accordion("🔍 检索调试信息", open=False):
        debug_info = gr.JSON(label="详细检索分析")

    # 绑定事件
    def process_query(query_text, method_choice):
        response, metadata = app.chat_with_agent(query_text, method_choice)

        # 更新统计信息
        quality = metadata.get("retrieval_quality", {}).get("quality_score", 0)
        time_taken = metadata.get("response_time", 0)
        num_docs = metadata.get("num_retrieved_docs", 0)

        return response, quality, time_taken, num_docs

    def get_debug_info(query_text, method_choice):
        debug_data = app.get_retrieval_debug_info(query_text, method_choice)
        return debug_data

    submit.click(
        fn=process_query,
        inputs=[query, method],
        outputs=[output, quality_score, response_time, doc_count]
    )

    debug_btn.click(
        fn=get_debug_info,
        inputs=[query, method],
        outputs=[debug_info]
    )

    # 示例问题
    gr.Examples(
        examples=[
            "现在完成时的考点和用法",
            "如何正确使用冠词 a/an/the",
            "英语阅读理解的解题技巧",
            "虚拟语气的语法规则",
            "定语从句的使用方法"
        ],
        inputs=[query]
    )


# 启动应用
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7863)
    print("🚀 增强版RAG应用已启动: http://localhost:7863")