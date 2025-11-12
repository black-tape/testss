from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import gradio as gr
from config import DB_DIR, EMBEDDING_MODEL, LLM_MODEL, QA_PROMPT

# 1️⃣ 向量数据库
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# 2️⃣ LLM
llm = OllamaLLM(model=LLM_MODEL)

# 3️⃣ Prompt
prompt = PromptTemplate(
    template=QA_PROMPT,
    input_variables=["context", "query"]
)

# 4️⃣ QA Chain (使用现代 LCEL 语法)
qa_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
)

# -----------------------------
# 5️⃣ 定义前端函数
# -----------------------------
def chat_with_agent(query):
    if not query.strip():
        return "❌ 请输入问题"
    try:
        result = qa_chain.invoke(query)
        return result
    except Exception as e:
        return f"❌ 错误: {str(e)}"

# -----------------------------
# 4️⃣ Gradio 前端美化
# -----------------------------
with gr.Blocks(title="🎓 英语学习智能体") as demo:
    gr.Markdown(
        """
        # 🎓 英语学习智能体
        
        > 💡 输入一个英语问题或语法点，我会结合文档为你讲解。
        """
    )
    
    with gr.Row():
        query = gr.Textbox(
            label="你的问题",
            placeholder="例如：现在完成时的考点",
            lines=2
        )
        submit = gr.Button("提问", variant="primary")
    
    output = gr.Markdown(label="回答")
    submit.click(fn=chat_with_agent, inputs=query, outputs=output)

# -----------------------------
# 5️⃣ 启动
# -----------------------------
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7862)
