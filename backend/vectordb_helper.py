from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os, re, chromadb, json

load_dotenv()

API_KEY = os.getenv("API_KEY")

VECTOR_DB_CACHE = {}
CHROMA_CLIENT = None

def get_vector_db(models, collection_name):
    global VECTOR_DB_CACHE, CHROMA_CLIENT

    # Return if already cached
    if collection_name in VECTOR_DB_CACHE:
        print("⚡ Using cached vector DB")
        return VECTOR_DB_CACHE[collection_name]

    print("🚀 Creating new vector DB...")

    # Initialize client once
    if CHROMA_CLIENT is None:
        BASE_DIR = Path(__file__).resolve().parent
        PERSIST_DIR = BASE_DIR.parent / "vectordb" / "chroma_db"
        CHROMA_CLIENT = chromadb.PersistentClient(path=str(PERSIST_DIR))

    # Create vectordb
    vectordb = Chroma(
        client=CHROMA_CLIENT,
        collection_name=collection_name,
        embedding_function=models["embedding"]
    )

    # Store in cache
    VECTOR_DB_CACHE[collection_name] = vectordb

    return vectordb

def clear_all_documents():
    db_path = Path(__file__).resolve().parent / "chroma_db"

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        collection_name="ade_documents",
        embedding_function=embedding,
        persist_directory=str(db_path)
    )

    # 🔥 Get all data (ids included by default)
    data = vectordb._collection.get()
    ids = data.get("ids", [])

    if ids:
        vectordb._collection.delete(ids=ids)
        print(f"🧹 Deleted {len(ids)} documents, structure intact ✅")
    else:
        print("⚠️ No documents found")


def get_docs_from_DB(results):
    docs = []

    for page_data in results:
        page = page_data["page"]

        # Text regions
        for i, text in enumerate(page_data["text_regions"]):
            content = f"[Page {page}] {text}"

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "page": page,
                        "type": "text",
                        "chunk_id": i
                    }
                )
            )

        # Structured regions
        for i, region in enumerate(page_data["structured_regions"]):
            content = f"[Page {page}] Structured Data:\n{json.dumps(region)}"

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "page": page,
                        "type": "structured",
                        "chunk_id": i
                    }
                )
            )

    return docs


def ingest_document(vectordb, collection_name, docs):

    vectordb.add_documents(docs)

    print(f"✅ Stored {len(docs)} chunks in collection: {collection_name}")


def clean_llm_output(text: str) -> str:
    if not text:
        return text

    # convert \[ ... \] → $ ... $
    text = re.sub(r"\\\\\[(.*?)\\\\\]", r"$\1$", text)

    # convert \(...\) → $...$
    text = re.sub(r"\\\\\((.*?)\\\\\)", r"$\1$", text)

    # remove \text{}
    text = re.sub(r"\\\\text\{(.*?)\}", r"\1", text)

    # remove \mathbf{}
    text = re.sub(r"\\\\mathbf\{(.*?)\}", r"\1", text)

    # remove extra backslashes before numbers / words
    text = text.replace("\\", "")

    return text


def get_answer(retriever, user_query):

    # system_prompt = (
    #     "Use the following pieces of retrieved context to answer the "
    #     "user's question. "
    #     "If you don't know the answer, say that you don't know."
    #     "Output math in $ format instead of \[ \]"
    #     "\n\n"
    #     "{context}"
    # )

    system_prompt = """
        You are a document question-answering assistant.

        The provided context may contain two types of data:

        1. JSON / structured data (from layout / table extraction)
        2. OCR text (raw text extracted from document)

        IMPORTANT:
        Structured JSON data is more accurate than OCR text.

        PRIORITY RULES:

        1. First, search for the answer in JSON / structured data.
        2. If the answer exists in JSON, use ONLY the JSON data.
        3. If the answer is not present in JSON, then use OCR text.
        4. If both exist, prefer JSON.
        5. Never override JSON values using OCR text.

        - Use ONLY the provided context.
        - Do NOT use outside knowledge.
        - Keep the answer consice and accurate 
        - Do not provide any unnecessary data that isn't asked by user.

        Context:
        {context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatGroq(
        api_key=API_KEY,
        model="openai/gpt-oss-120b",
        temperature=1
    )

    rag_chain = create_retrieval_chain(retriever, prompt | llm)

    response = rag_chain.invoke({"input": user_query})

    # always convert to string
    answer = response.get("answer", "")

    if hasattr(answer, "content"):
        answer = answer.content

    answer = clean_llm_output(answer)

    return str(answer)  

