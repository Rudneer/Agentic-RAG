from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os, re, chromadb, json
from langchain_openai import OpenAIEmbeddings

load_dotenv()

API_KEY = os.getenv("API_KEY")

COLLECTION_CACHE = {}
CHROMA_CLIENT = None

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_PATH = BASE_DIR / "vectordb" / "chroma_db"

def get_collection(collection_name):
    global COLLECTION_CACHE, CHROMA_CLIENT

    # Return cached collection
    if collection_name in COLLECTION_CACHE:
        print("⚡ Using cached collection")
        return COLLECTION_CACHE[collection_name]

    print("🚀 Creating / loading collection...")

    # Initialize client once
    if CHROMA_CLIENT is None:

        CHROMA_CLIENT = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH)
        )

    # Get or create collection
    collection = CHROMA_CLIENT.get_or_create_collection(
        name=collection_name
    )

    # Cache collection
    COLLECTION_CACHE[collection_name] = collection

    return collection

RETRIEVER_CACHE = {}

def get_retriever(embedding_model, collection_name):

    global RETRIEVER_CACHE

    if collection_name in RETRIEVER_CACHE:
        print("⚡ Using cached retriever")
        return RETRIEVER_CACHE[collection_name]

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DB_PATH)
    )

    retriever = vectordb.as_retriever()

    RETRIEVER_CACHE[collection_name] = retriever

    return retriever


def clear_all_documents():
    db_path = Path(__file__).resolve().parent / "chroma_db"

    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

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
        

# Generate embeddings from docs
def generate_embeddings(embedding_model, docs):
    texts = [doc.page_content for doc in docs]
    embeddings = embedding_model.embed_documents(texts)
    return embeddings


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
                        "chunk_id": f"{page}_text_{i}"
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
                        "chunk_id": f"{page}_structure_{i}"
                    }
                )
            )

    return docs


def ingest_document(embedding_model, collection, collection_name, docs):

    ids = [
        f"{collection_name}_{doc.metadata['chunk_id']}"
        for doc in docs
    ]

    collection.add(
        documents=[doc.page_content for doc in docs],
        metadatas=[doc.metadata for doc in docs],
        embeddings=generate_embeddings(embedding_model, docs),
        ids=ids
    )

    print(f"✅ Stored {len(docs)} chunks in collection")


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
        The document may contain OCR mistakes. Words may be slightly misspelled or 
        characters may be misread.

        Examples:
        - "IESC" may mean "IFSC"
        - "Acc0unt" may mean "Account"
        - "Brnch" may mean "Branch"

        When answering the user's question, interpret the document intelligently 
        and assume such errors may exist.


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
        temperature=0.5
    )

    rag_chain = create_retrieval_chain(retriever, prompt | llm)

    docs = retriever.invoke(user_query)

    for i, doc in enumerate(docs):
        print(f"Chunk {i+1}")
        print(doc.page_content)
        print("-----")

    response = rag_chain.invoke({"input": user_query})

    # always convert to string
    answer = response.get("answer", "")

    if hasattr(answer, "content"):
        answer = answer.content

    answer = clean_llm_output(answer)

    return str(answer)  

