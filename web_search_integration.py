# web_search_integration.py - 网络检索集成模块

import requests
import re
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse
from langchain_core.documents import Document
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSearchEngine:
    """网络搜索引擎基类"""

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """搜索网络内容"""
        raise NotImplementedError


class DuckDuckGoSearchEngine(WebSearchEngine):
    """DuckDuckGo 搜索引擎 - 免费且无需API密钥"""

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """使用DuckDuckGo进行搜索"""
        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            results = []

            # 主要结果
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", ""),
                    "content": data["Abstract"],
                    "url": data.get("AbstractURL", ""),
                    "source": "DuckDuckGo Abstract"
                })

            # 相关主题
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if "Text" in topic:
                    results.append({
                        "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                        "content": topic["Text"],
                        "url": topic.get("FirstURL", ""),
                        "source": "DuckDuckGo Related"
                    })

            return results[:max_results]

        except Exception as e:
            logger.error(f"DuckDuckGo搜索失败: {e}")
            return []


class WikipediaSearchEngine(WebSearchEngine):
    """维基百科搜索引擎 - 适合学术内容"""

    def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """搜索维基百科内容"""
        try:
            # 搜索维基百科页面
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(query)}"

            response = requests.get(search_url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                return [{
                    "title": data.get("title", ""),
                    "content": data.get("extract", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "source": "Wikipedia"
                }]
            else:
                # 如果直接访问失败，尝试搜索
                return self._search_wikipedia_fallback(query, max_results)

        except Exception as e:
            logger.error(f"维基百科搜索失败: {e}")
            return []

    def _search_wikipedia_fallback(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """维基百科搜索后备方案"""
        try:
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "utf8": 1,
                "format": "json",
                "srlimit": max_results
            }

            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = []

            for item in data.get("query", {}).get("search", []):
                title = item.get("title", "")
                page_id = item.get("pageid", 0)

                if page_id:
                    # 获取页面摘要
                    summary_url = "https://en.wikipedia.org/w/api.php"
                    summary_params = {
                        "action": "query",
                        "prop": "extracts",
                        "exintro": 1,
                        "explaintext": 1,
                        "pageids": page_id,
                        "utf8": 1,
                        "format": "json"
                    }

                    summary_response = requests.get(summary_url, params=summary_params, timeout=5)
                    if summary_response.status_code == 200:
                        summary_data = summary_response.json()
                        pages = summary_data.get("query", {}).get("pages", {})

                        if str(page_id) in pages:
                            page = pages[str(page_id)]
                            content = page.get("extract", "")

                            if content:
                                results.append({
                                    "title": title,
                                    "content": content,
                                    "url": f"https://en.wikipedia.org/wiki/{quote(title)}",
                                    "source": "Wikipedia"
                                })

            return results

        except Exception as e:
            logger.error(f"维基百科后备搜索失败: {e}")
            return []


class WebContentExtractor:
    """网页内容提取器"""

    @staticmethod
    def extract_content(url: str, max_length: int = 2000) -> str:
        """提取网页主要内容"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # 移除不需要的元素
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()

            # 提取主要内容
            content = ""

            # 尝试找到主要内容区域
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|body'))

            if main_content:
                content = main_content.get_text(separator=' ', strip=True)
            else:
                # 提取所有段落
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])

            # 清理内容
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()

            return content[:max_length] + "..." if len(content) > max_length else content

        except Exception as e:
            logger.error(f"网页内容提取失败 {url}: {e}")
            return ""


class HybridRAGSystem:
    """混合RAG系统 - 结合本地知识库和网络检索"""

    def __init__(self, local_retriever, enable_web_search: bool = True):
        self.local_retriever = local_retriever
        self.enable_web_search = enable_web_search

        # 初始化搜索引擎
        self.search_engines = [
            DuckDuckGoSearchEngine(),
            WikipediaSearchEngine()
        ]

        self.content_extractor = WebContentExtractor()

    def search_and_retrieve(self, query: str, use_local: bool = True, use_web: bool = True) -> List[Document]:
        """
        混合检索：结合本地和网络搜索

        Args:
            query: 查询字符串
            use_local: 是否使用本地检索
            use_web: 是否使用网络检索

        Returns:
            合并后的文档列表
        """
        all_docs = []

        # 1. 本地知识库检索
        if use_local and self.local_retriever:
            try:
                local_docs = self.local_retriever.get_relevant_documents(query, method="enhanced")
                for doc in local_docs:
                    doc.metadata.update({
                        "source_type": "local",
                        "retrieval_method": "vector_search"
                    })
                all_docs.extend(local_docs)
                logger.info(f"本地检索到 {len(local_docs)} 个文档")
            except Exception as e:
                logger.error(f"本地检索失败: {e}")

        # 2. 网络搜索
        if use_web and self.enable_web_search:
            web_docs = self._web_search(query)
            all_docs.extend(web_docs)
            logger.info(f"网络搜索到 {len(web_docs)} 个文档")

        # 3. 去重和排序
        return self._deduplicate_and_rank(all_docs)

    def _web_search(self, query: str, max_results_per_engine: int = 2) -> List[Document]:
        """执行网络搜索"""
        web_docs = []

        # 为英语学习添加相关关键词
        enhanced_query = self._enhance_query(query)

        for engine in self.search_engines:
            try:
                results = engine.search(enhanced_query, max_results_per_engine)

                for result in results:
                    content = result.get("content", "")
                    title = result.get("title", "")
                    url = result.get("url", "")
                    source = result.get("source", "web")

                    if content and len(content) > 50:  # 过滤太短的内容
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": url,
                                "title": title,
                                "source_type": "web",
                                "engine": source,
                                "query": enhanced_query
                            }
                        )
                        web_docs.append(doc)

                # 避免请求过于频繁
                time.sleep(1)

            except Exception as e:
                logger.error(f"网络搜索失败 {engine.__class__.__name__}: {e}")

        return web_docs

    def _enhance_query(self, query: str) -> str:
        """为英语学习查询添加相关关键词"""
        english_keywords = [
            "English grammar", "ESL", "English learning",
            "English usage", "grammar rules"
        ]

        # 检查是否是英语学习相关查询
        if any(keyword in query.lower() for keyword in ["英语", "语法", "用法", "时态", "冠词"]):
            # 为简单的中文查询添加英文关键词
            if len(query) < 20:
                return f"{query} English grammar rules examples"

        return query

    def _deduplicate_and_rank(self, docs: List[Document]) -> List[Document]:
        """去重和排序文档"""
        # 简单去重：基于内容前100字符
        unique_docs = []
        seen_content = set()

        # 优先本地文档，然后网络文档
        local_docs = [doc for doc in docs if doc.metadata.get("source_type") == "local"]
        web_docs = [doc for doc in docs if doc.metadata.get("source_type") == "web"]

        # 先添加本地文档
        for doc in local_docs:
            content = doc.page_content[:100]
            if content not in seen_content:
                seen_content.add(content)
                unique_docs.append(doc)

        # 再添加网络文档
        for doc in web_docs:
            content = doc.page_content[:100]
            if content not in seen_content:
                seen_content.add(content)
                unique_docs.append(doc)

        return unique_docs[:8]  # 限制总文档数量


def create_hybrid_retriever(local_retriever, enable_web_search: bool = True) -> HybridRAGSystem:
    """创建混合检索器"""
    return HybridRAGSystem(local_retriever, enable_web_search)


# 示例使用
if __name__ == "__main__":
    # 测试网络搜索功能
    print("🌐 测试网络搜索功能")
    print("=" * 50)

    # 测试搜索引擎
    ddg = DuckDuckGoSearchEngine()
    wiki = WikipediaSearchEngine()

    test_query = "现在完成时 English grammar"

    print(f"\n🔍 测试查询: {test_query}")

    # DuckDuckGo搜索
    print("\n📊 DuckDuckGo搜索结果:")
    ddg_results = ddg.search(test_query)
    for i, result in enumerate(ddg_results[:3], 1):
        print(f"{i}. {result['title']}")
        print(f"   {result['content'][:100]}...")
        print(f"   来源: {result['source']}")
        print()

    # 维基百科搜索
    print("\n📚 维基百科搜索结果:")
    wiki_results = wiki.search(test_query)
    for i, result in enumerate(wiki_results[:2], 1):
        print(f"{i}. {result['title']}")
        print(f"   {result['content'][:100]}...")
        print(f"   来源: {result['source']}")
        print()

    print("✅ 网络搜索功能测试完成！")