# retriever_enhanced.py - 增强的检索器系统

from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from typing import List, Dict, Any
from config import DB_DIR, EMBEDDING_MODEL, LLM_MODEL


class EnhancedRetriever:
    """增强的RAG检索器，支持多种检索策略和优化"""

    def __init__(self, vector_store_path: str = DB_DIR):
        self.vector_store_path = vector_store_path
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.llm = OllamaLLM(model=LLM_MODEL)

        # 加载向量数据库
        self.db = FAISS.load_local(
            vector_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # 初始化检索器
        self._setup_retrievers()

    def _setup_retrievers(self):
        """设置多种检索器策略"""

        # 1. 向量相似性检索器（优化的）
        self.vector_retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,                    # 检索Top5最相关文档
                "fetch_k": 10,            # 初步检索更多候选
            }
        )

        # 2. 多样性检索器（Maximal Marginal Relevance）
        self.mmr_retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,                    # 返回4个文档
                "fetch_k": 20,            # 从更多候选中选择
                "lambda_mult": 0.5,       # 多样性权重 (0-1)
            }
        )

        # 3. 准备混合检索（需要文档列表用于BM25）
        self._setup_ensemble_retriever()

        # 4. 上下文压缩检索器
        self._setup_compression_retriever()

    def _setup_ensemble_retriever(self):
        """设置混合检索器（简化版本）"""
        try:
            # 简化处理：直接使用向量检索器
            self.ensemble_retriever = self.vector_retriever
            print("✅ 使用向量检索器作为混合检索策略")
        except Exception as e:
            print(f"⚠️ 混合检索器设置失败: {e}")
            self.ensemble_retriever = self.vector_retriever

    def _setup_compression_retriever(self):
        """设置上下文压缩检索器（简化版本）"""
        try:
            # 简化处理：直接使用向量检索器
            self.compression_retriever = self.vector_retriever
            print("✅ 使用向量检索器作为压缩检索策略")
        except Exception as e:
            print(f"⚠️ 压缩检索器设置失败: {e}")
            self.compression_retriever = self.vector_retriever

    def get_relevant_documents(self, query: str, method: str = "enhanced") -> List[Document]:
        """
        根据查询检索相关文档

        Args:
            query: 用户查询
            method: 检索方法 ('vector', 'mmr', 'ensemble', 'compression', 'enhanced')

        Returns:
            检索到的相关文档列表
        """
        if method == "vector":
            return self.vector_retriever.invoke(query)
        elif method == "mmr":
            return self.mmr_retriever.invoke(query)
        elif method == "ensemble":
            return self.ensemble_retriever.invoke(query)
        elif method == "compression":
            return self.compression_retriever.invoke(query)
        elif method == "enhanced":
            # 增强检索：结合多种方法
            return self._enhanced_retrieval(query)
        else:
            raise ValueError(f"未知的检索方法: {method}")

    def _enhanced_retrieval(self, query: str) -> List[Document]:
        """增强检索：结合多种策略"""
        results = []

        # 1. 向量相似性检索
        vector_docs = self.vector_retriever.invoke(query)
        results.extend(vector_docs)

        # 2. 如果向量检索结果不够，使用MMR
        if len(vector_docs) < 3:
            mmr_docs = self.mmr_retriever.invoke(query)
            # 添加不重复的文档
            for doc in mmr_docs:
                if doc not in results and len(results) < 6:
                    results.append(doc)

        # 3. 去重并限制数量
        unique_docs = []
        seen_content = set()

        for doc in results:
            content = doc.page_content[:100]  # 使用前100字符判断是否重复
            if content not in seen_content:
                seen_content.add(content)
                unique_docs.append(doc)
                if len(unique_docs) >= 5:  # 最多返回5个文档
                    break

        return unique_docs

    def analyze_retrieval_quality(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """分析检索质量"""
        if not docs:
            return {
                "query": query,
                "num_results": 0,
                "avg_content_length": 0,
                "quality_score": 0,
                "recommendations": ["未检索到文档，请检查查询或知识库"]
            }

        # 计算基本指标
        content_lengths = [len(doc.page_content) for doc in docs]
        avg_length = np.mean(content_lengths)

        # 简单的质量评分（基于内容长度和文档数量）
        quality_score = min(100, (len(docs) * 20) + (avg_length / 20))

        # 生成建议
        recommendations = []
        if len(docs) < 3:
            recommendations.append("检索结果较少，考虑降低相似度阈值")
        if avg_length < 100:
            recommendations.append("文档片段较短，可能缺乏上下文")
        if quality_score < 60:
            recommendations.append("检索质量偏低，建议优化查询或检索策略")

        return {
            "query": query,
            "num_results": len(docs),
            "avg_content_length": avg_length,
            "quality_score": quality_score,
            "recommendations": recommendations,
            "documents_preview": [{"content": doc.page_content[:100] + "..."} for doc in docs[:3]]
        }


def create_enhanced_retriever() -> EnhancedRetriever:
    """创建增强检索器实例"""
    return EnhancedRetriever()


if __name__ == "__main__":
    # 测试增强检索器
    retriever = create_enhanced_retriever()

    test_query = "现在完成时的用法"

    print(f"🔍 测试查询: {test_query}")
    print("=" * 50)

    # 测试不同检索方法
    methods = ["vector", "mmr", "enhanced"]
    for method in methods:
        print(f"\n📊 {method.upper()} 检索结果:")
        docs = retriever.get_relevant_documents(test_query, method=method)

        for i, doc in enumerate(docs, 1):
            print(f"{i}. [{doc.metadata.get('source', 'unknown')}]")
            print(f"   {doc.page_content[:150]}...")

        # 分析检索质量
        quality = retriever.analyze_retrieval_quality(test_query, docs)
        print(f"   📈 质量评分: {quality['quality_score']:.1f}/100")
        print(f"   💡 建议: {', '.join(quality['recommendations']) if quality['recommendations'] else '良好'}")