# test_retriever.py - 检索器测试和性能评估

from retriever_enhanced import EnhancedRetriever
import time
from typing import List, Dict
import json


def test_retrieval_methods():
    """测试不同的检索方法"""
    retriever = EnhancedRetriever()

    # 测试查询
    test_queries = [
        "现在完成时的用法",
        "英语阅读理解技巧",
        "冠词的使用规则",
        "虚拟语气语法",
        "定语从句"
    ]

    print("🔍 RAG检索器性能测试")
    print("=" * 80)

    results = {}

    for query in test_queries:
        print(f"\n📝 查询: {query}")
        print("-" * 60)

        query_results = {}

        # 测试所有检索方法
        methods = ["vector", "mmr", "enhanced"]
        for method in methods:
            start_time = time.time()
            docs = retriever.get_relevant_documents(query, method=method)
            response_time = time.time() - start_time

            # 分析检索质量
            quality = retriever.analyze_retrieval_quality(query, docs)

            query_results[method] = {
                "num_docs": len(docs),
                "response_time": response_time,
                "quality_score": quality["quality_score"],
                "avg_content_length": quality["avg_content_length"],
                "recommendations": quality["recommendations"],
                "docs_preview": [doc.page_content[:100] + "..." for doc in docs[:2]]
            }

            print(f"  {method.upper():10} | 文档: {len(docs):2d} | "
                  f"质量: {quality['quality_score']:5.1f} | "
                  f"时间: {response_time:.3f}s")

            # 显示第一个文档预览
            if docs:
                print(f"             | 预览: {docs[0].page_content[:50]}...")

        results[query] = query_results

    # 保存结果
    with open("retrieval_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n📊 详细测试结果已保存到 retrieval_test_results.json")

    # 性能总结
    print("\n📈 性能总结")
    print("=" * 80)
    for method in ["vector", "mmr", "enhanced"]:
        avg_quality = sum(r[method]["quality_score"] for r in results.values()) / len(results)
        avg_time = sum(r[method]["response_time"] for r in results.values()) / len(results)
        avg_docs = sum(r[method]["num_docs"] for r in results.values()) / len(results)

        print(f"{method.upper():10} | 平均质量: {avg_quality:6.1f} | "
              f"平均时间: {avg_time:.3f}s | 平均文档数: {avg_docs:.1f}")


def compare_retrieval_results():
    """比较不同检索方法的结果差异"""
    retriever = EnhancedRetriever()
    query = "现在完成时的用法"

    print(f"\n🔄 检索结果比较: '{query}'")
    print("=" * 80)

    methods = ["vector", "mmr", "enhanced"]
    all_docs = {}

    for method in methods:
        docs = retriever.get_relevant_documents(query, method=method)
        all_docs[method] = docs

        print(f"\n{method.upper()} 检索结果 ({len(docs)}个文档):")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. [{doc.metadata.get('source', 'unknown')}]")
            print(f"     {doc.page_content[:120]}...")

    # 检查重叠度
    print("\n📊 结果重叠度分析:")
    vector_contents = set(d.page_content[:100] for d in all_docs["vector"])
    mmr_contents = set(d.page_content[:100] for d in all_docs["mmr"])
    enhanced_contents = set(d.page_content[:100] for d in all_docs["enhanced"])

    overlap_vector_mmr = len(vector_contents & mmr_contents)
    overlap_vector_enhanced = len(vector_contents & enhanced_contents)
    overlap_mmr_enhanced = len(mmr_contents & enhanced_contents)

    print(f"  Vector ∩ MMR: {overlap_vector_mmr}/{min(len(vector_contents), len(mmr_contents))}")
    print(f"  Vector ∩ Enhanced: {overlap_vector_enhanced}/{min(len(vector_contents), len(enhanced_contents))}")
    print(f"  MMR ∩ Enhanced: {overlap_mmr_enhanced}/{min(len(mmr_contents), len(enhanced_contents))}")


def test_edge_cases():
    """测试边界情况"""
    retriever = EnhancedRetriever()

    print("\n🧪 边界情况测试")
    print("=" * 80)

    edge_cases = [
        "",  # 空查询
        "x",  # 过短查询
        "这是一个非常长且复杂的查询，包含很多词汇，用来测试系统在处理长查询时的表现",  # 过长查询
        "12345",  # 纯数字
        "不相关的内容xyz",  # 不相关内容
    ]

    for query in edge_cases:
        print(f"\n测试查询: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        try:
            docs = retriever.get_relevant_documents(query, method="enhanced")
            quality = retriever.analyze_retrieval_quality(query, docs)

            print(f"  结果数量: {len(docs)}")
            print(f"  质量评分: {quality['quality_score']:.1f}")
            print(f"  建议: {', '.join(quality['recommendations']) if quality['recommendations'] else '正常'}")

        except Exception as e:
            print(f"  ❌ 错误: {e}")


if __name__ == "__main__":
    print("🚀 开始RAG检索器测试\n")

    # 运行所有测试
    test_retrieval_methods()
    compare_retrieval_results()
    test_edge_cases()

    print("\n✅ 测试完成！")