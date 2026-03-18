from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from pathlib import Path

COLLECTION_NAME = "ade_documents"

def init_db():
    print("🔧 Initializing Vector DB...")

    BASE_DIR = Path(__file__).resolve().parent
    PERSIST_DIR = BASE_DIR / "chroma_db"

    os.makedirs(PERSIST_DIR, exist_ok=True)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=str(PERSIST_DIR)
    )

    vectordb.persist()

    print("✅ DB created at:", PERSIST_DIR.resolve())

if __name__ == "__main__":
    init_db()
