# conversation_memory.py - 上下文记忆功能模块

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class ConversationTurn:
    """对话轮次数据结构"""
    user_query: str
    ai_response: str
    timestamp: datetime
    session_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieved_docs: List[str] = field(default_factory=list)
    context_summary: str = ""
    keywords: List[str] = field(default_factory=list)


class ConversationMemory:
    """对话记忆管理器"""

    def __init__(self,
                 max_history: int = 10,
                 max_context_length: int = 2000,
                 memory_file: str = "conversation_history.json"):
        self.max_history = max_history
        self.max_context_length = max_context_length
        self.memory_file = memory_file
        self.conversation_history: List[ConversationTurn] = []
        self.current_session_id = self._generate_session_id()

        # 加载历史记录
        self._load_history()

    def _generate_session_id(self) -> str:
        """生成会话ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_conversation_turn(self,
                            user_query: str,
                            ai_response: str,
                            metadata: Dict[str, Any] = None,
                            retrieved_docs: List[str] = None) -> None:
        """添加新的对话轮次"""
        turn = ConversationTurn(
            user_query=user_query,
            ai_response=ai_response,
            timestamp=datetime.now(),
            session_id=self.current_session_id,
            metadata=metadata or {},
            retrieved_docs=retrieved_docs or [],
            keywords=self._extract_keywords(user_query + " " + ai_response),
            context_summary=self._generate_summary(user_query, ai_response)
        )

        self.conversation_history.append(turn)

        # 限制历史记录长度
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        # 保存到文件
        self._save_history()

    def get_context_for_query(self, current_query: str, max_length: int = None) -> str:
        """为当前查询获取上下文"""
        if not self.conversation_history:
            return ""

        max_length = max_length or self.max_context_length

        # 获取相关的历史对话
        relevant_contexts = self._get_relevant_context(current_query)

        # 构建上下文字符串
        context_parts = []
        current_length = 0

        for turn in relevant_contexts:
            context_snippet = self._format_context_turn(turn)

            if current_length + len(context_snippet) > max_length:
                break

            context_parts.append(context_snippet)
            current_length += len(context_snippet)

        if not context_parts:
            return ""

        context_header = f"📝 **对话历史 (最近{len(context_parts)}轮):**\n\n"
        return context_header + "\n\n".join(context_parts)

    def _get_relevant_context(self, current_query: str) -> List[ConversationTurn]:
        """获取与当前查询相关的上下文"""
        if not self.conversation_history:
            return []

        # 简单的相关性计算
        current_keywords = set(self._extract_keywords(current_query))

        scored_turns = []
        for turn in self.conversation_history[-5:]:  # 只考虑最近的5轮对话
            turn_keywords = set(turn.keywords)

            # 计算关键词重叠度
            overlap = len(current_keywords & turn_keywords)
            recency_score = len(self.conversation_history) - self.conversation_history.index(turn)

            total_score = overlap * 2 + recency_score
            scored_turns.append((total_score, turn))

        # 按分数排序并返回
        scored_turns.sort(key=lambda x: x[0], reverse=True)
        return [turn for _, turn in scored_turns]

    def _format_context_turn(self, turn: ConversationTurn) -> str:
        """格式化单个对话轮次"""
        time_str = turn.timestamp.strftime("%H:%M")
        return f"**[{time_str}] 用户:** {turn.user_query}\n**助手:** {turn.ai_response}"

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取（可以替换为更复杂的NLP方法）
        english_grammar_terms = [
            "present perfect", "past tense", "future tense", "conditionals",
            "articles", "prepositions", "conjunctions", "verbs", "nouns",
            "adjectives", "adverbs", "pronouns", "tense", "aspect",
            "grammar", "syntax", "clause", "phrase", "sentence"
        ]

        chinese_grammar_terms = [
            "现在完成时", "过去时", "将来时", "虚拟语气",
            "冠词", "介词", "连词", "动词", "名词",
            "形容词", "副词", "代词", "时态", "体态",
            "语法", "句法", "从句", "短语", "句子"
        ]

        text_lower = text.lower()
        keywords = []

        for term in english_grammar_terms + chinese_grammar_terms:
            if term.lower() in text_lower:
                keywords.append(term)

        return list(set(keywords))  # 去重

    def _generate_summary(self, user_query: str, ai_response: str) -> str:
        """生成对话摘要"""
        # 简单的摘要生成（可以替换为LLM生成）
        return f"用户询问了关于{self._extract_main_topic(user_query)}的问题"

    def _extract_main_topic(self, text: str) -> str:
        """提取主要主题"""
        topics = ["时态", "语法", "冠词", "虚拟语气", "条件句", "从句"]
        text_lower = text.lower()

        for topic in topics:
            if topic in text_lower:
                return topic

        return "英语语法"

    def _save_history(self) -> None:
        """保存历史记录到文件"""
        try:
            history_data = []
            for turn in self.conversation_history:
                turn_dict = {
                    "user_query": turn.user_query,
                    "ai_response": turn.ai_response,
                    "timestamp": turn.timestamp.isoformat(),
                    "session_id": turn.session_id,
                    "metadata": turn.metadata,
                    "retrieved_docs": turn.retrieved_docs,
                    "context_summary": turn.context_summary,
                    "keywords": turn.keywords
                }
                history_data.append(turn_dict)

            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"保存对话历史失败: {e}")

    def _load_history(self) -> None:
        """从文件加载历史记录"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    history_data = json.load(f)

                for turn_dict in history_data[-self.max_history:]:  # 只加载最近的记录
                    turn = ConversationTurn(
                        user_query=turn_dict["user_query"],
                        ai_response=turn_dict["ai_response"],
                        timestamp=datetime.fromisoformat(turn_dict["timestamp"]),
                        session_id=turn_dict["session_id"],
                        metadata=turn_dict.get("metadata", {}),
                        retrieved_docs=turn_dict.get("retrieved_docs", []),
                        context_summary=turn_dict.get("context_summary", ""),
                        keywords=turn_dict.get("keywords", [])
                    )
                    self.conversation_history.append(turn)

        except Exception as e:
            print(f"加载对话历史失败: {e}")

    def clear_history(self) -> None:
        """清空历史记录"""
        self.conversation_history.clear()
        self.current_session_id = self._generate_session_id()

        # 删除历史文件
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)

    def get_conversation_stats(self) -> Dict[str, Any]:
        """获取对话统计信息"""
        if not self.conversation_history:
            return {
                "total_conversations": 0,
                "current_session_length": 0,
                "most_discussed_topics": [],
                "avg_response_length": 0
            }

        # 统计信息
        total_conversations = len(self.conversation_history)
        current_session_conversations = len([
            turn for turn in self.conversation_history
            if turn.session_id == self.current_session_id
        ])

        # 统计话题
        all_keywords = []
        for turn in self.conversation_history:
            all_keywords.extend(turn.keywords)

        topic_counts = {}
        for keyword in all_keywords:
            topic_counts[keyword] = topic_counts.get(keyword, 0) + 1

        most_discussed = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # 平均回答长度
        avg_response_length = sum(len(turn.ai_response) for turn in self.conversation_history) / len(self.conversation_history)

        return {
            "total_conversations": total_conversations,
            "current_session_length": current_session_conversations,
            "most_discussed_topics": [{"topic": topic, "count": count} for topic, count in most_discussed],
            "avg_response_length": round(avg_response_length, 1)
        }

    def export_conversation(self, format: str = "json") -> str:
        """导出对话历史"""
        if format == "json":
            return self._export_as_json()
        elif format == "txt":
            return self._export_as_text()
        else:
            raise ValueError("Unsupported export format")

    def _export_as_json(self) -> str:
        """导出为JSON格式"""
        export_data = []
        for turn in self.conversation_history:
            export_data.append({
                "timestamp": turn.timestamp.isoformat(),
                "user": turn.user_query,
                "assistant": turn.ai_response,
                "session": turn.session_id,
                "keywords": turn.keywords
            })

        return json.dumps(export_data, ensure_ascii=False, indent=2)

    def _export_as_text(self) -> str:
        """导出为文本格式"""
        lines = ["=" * 60, "对话历史导出", "=" * 60]

        for i, turn in enumerate(self.conversation_history, 1):
            lines.append(f"\n[{i}] {turn.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"用户: {turn.user_query}")
            lines.append(f"助手: {turn.ai_response}")
            if turn.keywords:
                lines.append(f"关键词: {', '.join(turn.keywords)}")
            lines.append("-" * 40)

        return "\n".join(lines)


# 简化的LangChain兼容类（移除了依赖）
# class LangChainChatMemory:
#     """兼容LangChain的聊天记忆类"""
#     pass


# 全局记忆实例
_global_memory = None


def get_conversation_memory() -> ConversationMemory:
    """获取全局对话记忆实例"""
    global _global_memory
    if _global_memory is None:
        _global_memory = ConversationMemory()
    return _global_memory


def reset_conversation_memory():
    """重置全局对话记忆"""
    global _global_memory
    if _global_memory:
        _global_memory.clear_history()
    _global_memory = ConversationMemory()


# 示例使用
if __name__ == "__main__":
    # 测试对话记忆功能
    print("🧠 测试对话记忆功能")
    print("=" * 50)

    memory = ConversationMemory(max_history=5)

    # 模拟几轮对话
    conversations = [
        ("现在完成时的用法是什么？", "现在完成时用来表示从过去开始持续到现在的动作..."),
        ("能给一些例子吗？", "当然可以。例如：'I have lived here for 10 years'..."),
        ("那么过去时和现在完成时的区别？", "过去时表示特定时间发生的动作，而现在完成时强调与现在的联系...")
    ]

    for user_query, ai_response in conversations:
        memory.add_conversation_turn(user_query, ai_response)
        print(f"用户: {user_query}")
        print(f"助手: {ai_response[:50]}...")
        print()

    # 测试上下文获取
    current_query = "能再详细解释一下吗？"
    context = memory.get_context_for_query(current_query)
    print("📝 上下文信息:")
    print(context)
    print()

    # 测试统计信息
    stats = memory.get_conversation_stats()
    print("📊 对话统计:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    print("\n✅ 对话记忆功能测试完成！")