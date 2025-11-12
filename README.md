# 🎓 英语学习智能体 - RAG 应用

基于 LangChain + Ollama + FAISS 的智能英语学习问答系统，结合本地文档提供精准的英语学习辅导。

## 📋 项目概述

本项目是一个基于检索增强生成（RAG）技术的英语学习智能助手，通过分析本地英语学习文档，为用户提供个性化的语法解答和学习指导。

### 🎯 核心功能

- **智能问答**：基于上传的英语学习文档回答用户问题
- **本地化部署**：使用 Ollama 本地 LLM，保护数据隐私
- **高性能检索**：基于 FAISS 向量数据库的快速语义搜索
- **友好界面**：使用 Gradio 提供简洁直观的 Web 界面

## 🏗️ 技术架构

```
用户查询 → [Gradio界面] → [LangChain处理链] → [FAISS检索] → [Ollama LLM] → 回答输出
```

### 核心组件

1. **📚 文档处理** (`build_knowledge_base.py`)
   - 支持 PDF、DOCX 等多种文档格式
   - 智能分块策略（800字符/块，100字符重叠）
   - 向量化和索引构建

2. **🧠 RAG 系统** (`app.py`)
   - **向量数据库**：基于 FAISS 的高效语义检索
   - **大语言模型**：Ollama 本地部署的 llama3.1:8b
   - **检索链**：LangChain LCEL 语法的现代化处理流程
   - **提示工程**：针对英语学习优化的 Prompt 模板

3. **⚙️ 配置管理** (`config.py`)
   - 模型配置：all-minilm（嵌入模型）+ llama3.1:8b（生成模型）
   - 向量存储路径和文档路径管理
   - 可定制的 Prompt 模板

## 📁 项目结构

```
langchain_ollama_app/
├── app.py                    # 主应用文件（Gradio界面 + RAG处理）
├── build_knowledge_base.py   # 知识库构建脚本
├── config.py                # 配置文件
├── docs/                    # 英语学习文档目录
│   ├── 2020年考研英语真题第2套.docx
│   ├── 2020年考研英语阅读理解真题第1套【答案解析】.pdf
│   ├── 26考研专业课-政治法律 lesson2.docx
│   └── ...                  # 更多英语学习资料
├── vector_store/            # 向量数据库存储
│   ├── index.faiss
│   └── index.pkl
└── README.md               # 项目说明
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Ollama 服务已安装并运行
- 必要的 Python 包：

```bash
pip install langchain langchain_ollama langchain_community faiss-cpu gradio python-docx PyPDF2
```

### 安装和运行

1. **拉取 Ollama 模型**
   ```bash
   ollama pull all-minilm      # 嵌入模型
   ollama pull llama3.1:8b     # 生成模型
   ```

2. **构建知识库**
   ```bash
   python build_knowledge_base.py
   ```

3. **启动应用**
   ```bash
   python app.py
   ```

4. **访问应用**
   - 打开浏览器访问：`http://localhost:7862`

## 💡 使用示例

**用户提问**：
- "现在完成时的考点"
- "如何正确使用冠词 a/an/the"
- "英语阅读理解的技巧"

**系统回答**：
1. 📖 基于文档的详细语法讲解
2. 🔤 相关的原文英文示例
3. 💬 易懂的学习建议

## ⚙️ 配置说明

### 模型配置
- **嵌入模型**：`all-minilm` - 负责文本向量化
- **生成模型**：`llama3.1:8b` - 负责生成回答

### 分块策略
- **块大小**：800 字符
- **重叠部分**：100 字符
- **平衡点**：既保证语义完整性，又提高检索精度

### 自定义配置
修改 `config.py` 中的参数来适应不同的使用场景：
- 调整模型选择
- 修改分块策略
- 自定义 Prompt 模板

## 🔧 开发计划

- [ ] 支持更多文档格式（TXT、MD、HTML等）
- [ ] 添加对话历史记录功能
- [ ] 集成更多本地化模型选择
- [ ] 添加用户反馈和学习进度跟踪
- [ ] 支持文档实时更新和增量索引

## 📝 注意事项

1. **首次运行**：需要先运行 `build_knowledge_base.py` 构建向量数据库
2. **模型下载**：确保 Ollama 已下载所需模型
3. **内存使用**：FAISS 向量数据库会占用一定内存
4. **数据安全**：所有处理均在本地进行，无需担心数据泄露

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📄 许可证

MIT License
