# demo_web_search.py - 网络检索功能演示

from web_search_integration import DuckDuckGoSearchEngine, WikipediaSearchEngine, HybridRAGSystem
from retriever_enhanced import EnhancedRetriever
import json
import time


def test_web_search_engines():
    """测试各个搜索引擎"""
    print("🌐 测试网络搜索引擎")
    print("=" * 60)

    # 测试查询
    test_queries = [
        "English present perfect tense",
        "grammar rules articles a an the",
        "English conditionals if clauses"
    ]

    # 测试DuckDuckGo
    print("\n🦆 DuckDuckGo 搜索引擎测试")
    ddg = DuckDuckGoSearchEngine()

    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        try:
            results = ddg.search(query, max_results=3)
            print(f"📊 检索到 {len(results)} 个结果")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['title']}")
                print(f"     {result['content'][:80]}...")
                print(f"     来源: {result['source']}")
        except Exception as e:
            print(f"  ❌ 搜索失败: {e}")

    # 测试维基百科
    print("\n📚 维基百科搜索引擎测试")
    wiki = WikipediaSearchEngine()

    for query in ["present perfect", "english grammar", "conditionals"]:
        print(f"\n🔍 查询: {query}")
        try:
            results = wiki.search(query, max_results=2)
            print(f"📊 检索到 {len(results)} 个结果")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['title']}")
                print(f"     {result['content'][:80]}...")
                print(f"     来源: {result['source']}")
        except Exception as e:
            print(f"  ❌ 搜索失败: {e}")


def test_hybrid_system():
    """测试混合RAG系统"""
    print("\n\n🎯 混合RAG系统测试")
    print("=" * 60)

    # 初始化本地检索器
    try:
        local_retriever = EnhancedRetriever()
        print("✅ 本地检索器初始化成功")
    except Exception as e:
        print(f"❌ 本地检索器初始化失败: {e}")
        local_retriever = None

    # 创建混合系统
    hybrid_system = HybridRAGSystem(local_retriever, enable_web_search=True)

    # 测试查询
    test_queries = [
        "现在完成时的用法",
        "冠词的使用规则",
        "虚拟语气的语法"
    ]

    for query in test_queries:
        print(f"\n🔍 测试查询: {query}")
        print("-" * 40)

        try:
            start_time = time.time()
            docs = hybrid_system.search_and_retrieve(
                query,
                use_local=local_retriever is not None,
                use_web=True
            )
            search_time = time.time() - start_time

            print(f"⏱️  搜索耗时: {search_time:.2f}秒")
            print(f"📊 检索文档数: {len(docs)}")

            # 统计文档类型
            local_docs = [d for d in docs if d.metadata.get("source_type") == "local"]
            web_docs = [d for d in docs if d.metadata.get("source_type") == "web"]

            print(f"📚 本地文档: {len(local_docs)} 个")
            print(f"🌐 网络文档: {len(web_docs)} 个")

            # 显示前几个文档预览
            print("\n📄 文档预览:")
            for i, doc in enumerate(docs[:3], 1):
                source_type = doc.metadata.get("source_type", "unknown")
                source_label = "本地" if source_type == "local" else "网络"
                title = doc.metadata.get("title", "无标题")
                content = doc.page_content[:100]

                print(f"  {i}. [{source_label}] {title}")
                print(f"     {content}...")
                print()

        except Exception as e:
            print(f"❌ 混合检索失败: {e}")


def test_query_enhancement():
    """测试查询增强功能"""
    print("\n\n🔧 查询增强功能测试")
    print("=" * 60)

    local_retriever = None  # 为简化测试
    hybrid_system = HybridRAGSystem(local_retriever, enable_web_search=True)

    # 测试查询增强
    queries = [
        "现在完成时",
        "冠词用法",
        "虚拟语气",
        "English grammar"  # 英文查询
    ]

    for query in queries:
        enhanced = hybrid_system._enhance_query(query)
        print(f"原始: {query}")
        print(f"增强: {enhanced}")
        print(f"变化: {'是' if query != enhanced else '否'}")
        print()


def save_test_results():
    """保存测试结果到文件"""
    print("💾 保存测试结果...")

    # 收集测试数据
    test_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engines_tested": ["DuckDuckGo", "Wikipedia"],
        "features_tested": [
            "网络搜索",
            "混合检索",
            "查询增强",
            "文档去重",
            "质量分析"
        ],
        "recommendations": [
            "DuckDuckGo 适合一般查询",
            "维基百科 适合学术内容",
            "混合检索 提供最全面信息",
            "本地检索 速度快且可靠",
            "网络检索 获取最新资料"
        ]
    }

    # 保存到文件
    with open("web_search_test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print("✅ 测试结果已保存到 web_search_test_results.json")


if __name__ == "__main__":
    print("🚀 开始网络检索功能演示")
    print("=" * 80)

    # 运行所有测试
    test_web_search_engines()
    test_hybrid_system()
    test_query_enhancement()
    save_test_results()

    print("\n🎉 网络检索功能演示完成！")
    print("=" * 80)
    print("💡 接下来可以：")
    print("1. 运行 python app_hybrid.py 启动混合RAG应用")
    print("2. 在Gradio界面中测试网络搜索功能")
    print("3. 查看生成的 web_search_test_results.json 文件")