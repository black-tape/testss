# app_hybrid.py - 混合RAG应用（本地知识库 + 网络检索）

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.llms import OllamaLLM
from retriever_enhanced import EnhancedRetriever
from web_search_integration import create_hybrid_retriever
import gradio as gr
from config import LLM_MODEL
import time
from typing import Tuple, Dict, Any


class HybridRAGApp:
    """混合RAG应用 - 结合本地知识库和网络检索"""

    def __init__(self, enable_web_search: bool = True):
        # 1️⃣ 本地检索器
        self.local_retriever = EnhancedRetriever()

        # 2️⃣ 混合检索器
        self.hybrid_retriever = create_hybrid_retriever(
            self.local_retriever,
            enable_web_search
        )

        # 3️⃣ LLM
        self.llm = OllamaLLM(model=LLM_MODEL)

        # 4️⃣ 增强的Prompt模板（支持网络检索）
        self.prompt = PromptTemplate(
            template="""你是一个专业的英语学习助手。请根据提供的文档内容回答用户的问题。

要求：
1. 优先使用提供的相关文档信息进行回答
2. 如果有网络检索结果，请特别注明这是来自网络的信息
3. 给出具体的英文示例和用法说明
4. 如果文档中找不到相关信息，请诚实地说明
5. 回答要详细、准确、易懂

**检索到的资料：**
{context}

**用户问题：**
{query}

**回答：**""",
            input_variables=["context", "query"]
        )

        # 5️⃣ 处理链
        self.qa_chain = (
            {"context": lambda q: self._get_formatted_docs(q), "query": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )

    def _get_formatted_docs(self, query: str, use_local: bool = True, use_web: bool = True) -> str:
        """获取并格式化检索到的文档"""
        docs = self.hybrid_retriever.search_and_retrieve(query, use_local, use_web)

        if not docs:
            return "未找到相关学习资料。请尝试使用不同的关键词或检查网络连接。"

        formatted_docs = []
        local_count = 0
        web_count = 0

        for doc in docs:
            source_type = doc.metadata.get("source_type", "unknown")
            title = doc.metadata.get("title", "")
            source = doc.metadata.get("source", "")

            if source_type == "local":
                local_count += 1
                source_label = f"[本地文档{local_count}]"
            else:
                web_count += 1
                engine = doc.metadata.get("engine", "网络")
                source_label = f"[网络{web_count} - {engine}]"

            content = doc.page_content.strip()
            if title:
                formatted_docs.append(f"{source_label} **{title}**:\n{content}")
            else:
                formatted_docs.append(f"{source_label}:\n{content}")

        # 添加检索统计信息
        header = f"📚 检索到 {local_count} 个本地文档和 {web_count} 个网络资源\n\n"

        return header + "\n\n".join(formatted_docs)

    def chat_with_agent(self,
                       query: str,
                       use_local: bool = True,
                       use_web: bool = True,
                       search_method: str = "hybrid") -> Tuple[str, Dict[str, Any]]:
        """与智能体对话"""
        if not query.strip():
            return "❌ 请输入问题", {}

        try:
            start_time = time.time()

            # 选择检索方法
            if search_method == "hybrid":
                # 混合检索
                docs = self.hybrid_retriever.search_and_retrieve(query, use_local, use_web)
            elif search_method == "local_only":
                # 仅本地检索
                docs = self.local_retriever.get_relevant_documents(query, method="enhanced")
            elif search_method == "web_only":
                # 仅网络检索
                docs = self.hybrid_retriever._web_search(query)

            response_time = time.time() - start_time

            # 分析检索质量
            retrieval_quality = self._analyze_hybrid_retrieval(docs, use_local, use_web)

            # 如果没有检索到文档
            if not docs:
                return (
                    "❌ 抱歉，没有找到相关的学习资料。\n\n"
                    "💡 **建议：**\n"
                    "1. 尝试使用不同的关键词\n"
                    "2. 检查网络连接状态\n"
                    "3. 确保问题与英语学习相关\n"
                    "4. 尝试更具体的问题描述",
                    {
                        "retrieval_quality": retrieval_quality,
                        "response_time": response_time,
                        "num_retrieved_docs": 0,
                        "search_method": search_method
                    }
                )

            # 生成回答
            formatted_docs = self._get_docs_for_prompt(docs)
            response = self.qa_chain.invoke({"context": formatted_docs, "query": query})

            # 添加检索质量信息
            quality_info = self._format_quality_info(retrieval_quality, docs)
            final_response = response + "\n\n" + quality_info

            return final_response, {
                "retrieval_quality": retrieval_quality,
                "response_time": response_time,
                "num_retrieved_docs": len(docs),
                "search_method": search_method,
                "docs_breakdown": self._get_docs_breakdown(docs)
            }

        except Exception as e:
            return f"❌ 错误: {str(e)}", {}

    def _get_docs_for_prompt(self, docs) -> str:
        """为Prompt准备格式化的文档"""
        formatted_docs = []

        for doc in docs:
            source_type = doc.metadata.get("source_type", "unknown")
            title = doc.metadata.get("title", "")
            source = doc.metadata.get("source", "")

            if source_type == "local":
                source_label = "[本地资料]"
            else:
                engine = doc.metadata.get("engine", "网络")
                source_label = f"[网络资源 - {engine}]"

            content = doc.page_content.strip()[:1000]  # 限制长度

            if title:
                formatted_docs.append(f"{source_label} {title}:\n{content}")
            else:
                formatted_docs.append(f"{source_label}:\n{content}")

        return "\n\n".join(formatted_docs)

    def _analyze_hybrid_retrieval(self, docs, use_local: bool, use_web: bool) -> Dict[str, Any]:
        """分析混合检索质量"""
        if not docs:
            return {
                "quality_score": 0,
                "num_local_docs": 0,
                "num_web_docs": 0,
                "recommendations": ["未检索到文档，请检查查询参数"]
            }

        # 统计文档类型
        local_docs = [doc for doc in docs if doc.metadata.get("source_type") == "local"]
        web_docs = [doc for doc in docs if doc.metadata.get("source_type") == "web"]

        # 计算质量评分
        content_lengths = [len(doc.page_content) for doc in docs]
        avg_length = sum(content_lengths) / len(content_lengths)

        # 综合评分
        base_score = min(50, len(docs) * 10)  # 文档数量分数
        length_score = min(30, avg_length / 50)  # 内容长度分数
        diversity_score = min(20, len(local_docs) * 5 + len(web_docs) * 5)  # 多样性分数

        quality_score = base_score + length_score + diversity_score

        # 生成建议
        recommendations = []
        if len(local_docs) == 0 and use_local:
            recommendations.append("本地知识库未找到相关内容，考虑添加更多学习资料")
        if len(web_docs) == 0 and use_web:
            recommendations.append("网络搜索未返回结果，检查网络连接或尝试不同关键词")
        if quality_score < 60:
            recommendations.append("检索质量偏低，建议优化查询词或调整搜索策略")

        return {
            "quality_score": min(100, quality_score),
            "num_local_docs": len(local_docs),
            "num_web_docs": len(web_docs),
            "avg_content_length": avg_length,
            "recommendations": recommendations
        }

    def _format_quality_info(self, quality: Dict[str, Any], docs) -> str:
        """格式化质量信息"""
        info_parts = [
            "---",
            f"📊 **检索质量**: {quality['quality_score']:.1f}/100",
            f"📚 **本地文档**: {quality['num_local_docs']} 个",
            f"🌐 **网络资源**: {quality['num_web_docs']} 个"
        ]

        if quality['recommendations']:
            info_parts.append(f"💡 **建议**: {', '.join(quality['recommendations'])}")

        return "\n".join(info_parts)

    def _get_docs_breakdown(self, docs) -> Dict[str, int]:
        """获取文档分类统计"""
        breakdown = {"local": 0, "web": 0}
        for doc in docs:
            source_type = doc.metadata.get("source_type", "unknown")
            if source_type in breakdown:
                breakdown[source_type] += 1
        return breakdown


# 创建应用实例
app = HybridRAGApp(enable_web_search=True)


# Gradio界面
with gr.Blocks(title="🎓 混合RAG英语学习助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎓 混合RAG英语学习助手

        > 💡 **智能检索 2.0** - 融合本地知识库与实时网络搜索

        🔥 **核心功能**:
        - 📚 本地知识库检索（英语学习文档）
        - 🌐 实时网络搜索（DuckDuckGo + 维基百科）
        - 🎯 智能文档去重和排序
        - 📊 检索质量分析
        - ⚡ 快速响应时间
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            query = gr.Textbox(
                label="你的问题",
                placeholder="例如：现在完成时的用法和示例",
                lines=3
            )

            with gr.Row():
                use_local = gr.Checkbox(label="📚 本地检索", value=True)
                use_web = gr.Checkbox(label="🌐 网络搜索", value=True)

            with gr.Row():
                search_method = gr.Radio(
                    choices=[
                        ("🎯 智能混合", "hybrid"),
                        ("📚 仅本地", "local_only"),
                        ("🌐 仅网络", "web_only")
                    ],
                    value="hybrid",
                    label="搜索模式",
                    info="选择检索策略"
                )

            with gr.Row():
                submit = gr.Button("🚀 开始搜索", variant="primary")
                clear_btn = gr.Button("🗑️ 清空", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("### 📈 搜索统计")
            with gr.Row():
                local_count = gr.Number(label="本地文档", precision=0, interactive=False)
                web_count = gr.Number(label="网络资源", precision=0, interactive=False)

            quality_score = gr.Number(label="质量评分", precision=1, interactive=False)
            response_time = gr.Number(label="响应时间(秒)", precision=2, interactive=False)

    # 主回答区域
    with gr.Accordion("💬 智能回答", open=True):
        output = gr.Markdown(label="回答内容")

    # 详细信息区域
    with gr.Accordion("🔍 检索详情", open=False):
        search_info = gr.JSON(label="搜索分析", visible=True)

    # 绑定事件
    def process_query(query_text, local_enabled, web_enabled, method):
        if not query_text.strip():
            return "❌ 请输入问题", 0, 0, 0, 0, {}

        response, metadata = app.chat_with_agent(
            query_text, local_enabled, web_enabled, method
        )

        # 更新统计信息
        breakdown = metadata.get("docs_breakdown", {})
        local_num = breakdown.get("local", 0)
        web_num = breakdown.get("web", 0)
        quality = metadata.get("retrieval_quality", {}).get("quality_score", 0)
        time_taken = metadata.get("response_time", 0)

        search_metadata = {
            "搜索模式": method,
            "本地检索": local_enabled,
            "网络搜索": web_enabled,
            "检索文档数": metadata.get("num_retrieved_docs", 0),
            "本地文档数": local_num,
            "网络资源数": web_num,
            "质量评分": quality,
            "响应时间": f"{time_taken:.3f}秒"
        }

        return response, local_num, web_num, quality, time_taken, search_metadata

    submit.click(
        fn=process_query,
        inputs=[query, use_local, use_web, search_method],
        outputs=[output, local_count, web_count, quality_score, response_time, search_info]
    )

    clear_btn.click(
        fn=lambda: ("", "", True, True, "hybrid", 0, 0, 0, 0, {}),
        outputs=[output, query, use_local, use_web, search_method,
                local_count, web_count, quality_score, response_time, search_info]
    )

    # 示例问题
    gr.Examples(
        examples=[
            ["现在完成时的用法和区别"],
            ["冠词 a/an/the 的使用规则"],
            ["虚拟语气的语法结构和例句"],
            ["英语阅读理解解题技巧"],
            ["非谓语动词的用法总结"],
            ["定语从句的引导词选择"]
        ],
        inputs=[query]
    )

    # 使用说明
    with gr.Accordion("📖 使用说明", open=False):
        gr.Markdown(
            """
            ### 🎯 搜索模式说明

            **🎯 智能混合**：结合本地文档和网络搜索，提供最全面的信息

            **📚 仅本地**：只使用本地知识库，速度快，适合基础查询

            **🌐 仅网络**：只进行网络搜索，获取最新信息

            ### 💡 使用建议

            1. **语法学习**：建议使用"智能混合"模式
            2. **快速查询**：可以使用"仅本地"模式
            3. **最新资料**：建议使用"仅网络"模式
            4. **详细解答**：确保同时启用本地和网络检索

            ### ⚠️ 注意事项

            - 网络搜索需要稳定的网络连接
            - 不同搜索引擎的结果可能有差异
            - 建议优先使用本地文档，网络资料作为补充
            """
        )


# 启动应用
if __name__ == "__main__":
    print("🚀 混合RAG英语学习助手启动中...")
    demo.launch(share=False, server_name="0.0.0.0", server_port=7864)
    print("✅ 应用已启动: http://localhost:7864")