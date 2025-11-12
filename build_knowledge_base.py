# build_knowledge_base.py

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import os
from config import DOCS_DIR, DB_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def load_and_index_documents():
    docs = []

    for filename in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(path)
        elif filename.endswith(".txt"):
            loader = TextLoader(path)
        else:
            print(f"⚠️ 跳过不支持的文件格式: {filename}")
            continue

        docs.extend(loader.load())

    if not docs:
        print("❌ 没有找到可用文档，请在 docs/ 文件夹中添加教材或笔记。")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    split_docs = splitter.split_documents(docs)
    print(f"📚 已加载 {len(split_docs)} 个文档块")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local(DB_DIR)

    print(f"✅ 知识库构建完成，保存于: {DB_DIR}")


if __name__ == "__main__":
    load_and_index_documents()
