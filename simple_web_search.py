# simple_web_search.py - 简化的网络搜索功能

import requests
import json
from typing import List, Dict, Any
from urllib.parse import quote


class SimpleWebSearch:
    """简化的网络搜索功能"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def search_english_grammar(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """搜索英语语法相关内容"""
        try:
            # 使用维基百科API搜索
            enhanced_query = f"{query} English grammar"

            # 维基百科搜索
            wiki_results = self._search_wikipedia(enhanced_query)

            # DuckDuckGo API（简化版）
            ddg_results = self._search_duckduckgo_simple(enhanced_query, max_results)

            return wiki_results + ddg_results

        except Exception as e:
            print(f"搜索失败: {e}")
            return []

    def _search_wikipedia(self, query: str) -> List[Dict[str, Any]]:
        """搜索维基百科"""
        try:
            # 维基百科API
            url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(query.replace(" ", "_"))

            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return [{
                    "title": data.get("title", ""),
                    "content": data.get("extract", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "source": "Wikipedia"
                }]
            else:
                return []

        except Exception as e:
            print(f"维基百科搜索失败: {e}")
            return []

    def _search_duckduckgo_simple(self, query: str, max_results: int = 2) -> List[Dict[str, Any]]:
        """简化的DuckDuckGo搜索"""
        try:
            # 使用DuckDuckGo的即时答案API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = []

            # 主要摘要
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", ""),
                    "content": data["Abstract"],
                    "url": data.get("AbstractURL", ""),
                    "source": "DuckDuckGo"
                })

            # 相关主题（限制数量）
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
            print(f"DuckDuckGo搜索失败: {e}")
            return []


def test_simple_search():
    """测试简化搜索功能"""
    print("🌐 测试简化网络搜索功能")
    print("=" * 50)

    searcher = SimpleWebSearch()

    test_queries = [
        "present perfect tense",
        "articles a an the",
        "conditionals if clauses"
    ]

    for query in test_queries:
        print(f"\n🔍 搜索: {query}")
        print("-" * 30)

        try:
            results = searcher.search_english_grammar(query)
            print(f"📊 检索到 {len(results)} 个结果")

            for i, result in enumerate(results, 1):
                print(f"{i}. 【{result['source']}】{result['title']}")
                print(f"   {result['content'][:100]}...")
                print()

        except Exception as e:
            print(f"❌ 搜索失败: {e}")

    print("\n✅ 简化搜索测试完成！")


if __name__ == "__main__":
    test_simple_search()