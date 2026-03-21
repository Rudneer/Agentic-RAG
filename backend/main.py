import os
import time
import uuid
from pydantic import BaseModel
from model_loader import load_models
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Body
from vectordb_helper import get_vector_db, get_answer, get_docs_from_DB, ingest_document
from image_tools import load_pages
from process_page import process_page_parallel


app = FastAPI()

class ChatRequest(BaseModel):
    query: str
    file_id: str | None = None 
    collection_name: str

UPLOAD_DIR = "../uploads/images"

os.makedirs(UPLOAD_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve images
app.mount(
    "/images",
    StaticFiles(directory=UPLOAD_DIR),
    name="images",
)

# Preload all the models on startup
models = {}
vectordb = None

@app.on_event("startup")
async def startup():

    global models
    models = load_models()

    global vectordb
    

def generate_safe_filename(original_filename):
    ext = original_filename.split(".")[-1]
    unique_id = uuid.uuid4().hex
    return f"{unique_id}.{ext}", unique_id 

# API to upload doc in the folder
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    # 🔥 Generate safe filename + collection name
    unique_name, collection_name = generate_safe_filename(file.filename)

    path = os.path.join(UPLOAD_DIR, unique_name)

    # Save file
    with open(path, "wb") as f:
        f.write(await file.read())

    return {
        "filename": unique_name,
        "collection_name": collection_name,  # 🔥 critical for retrieval
        "url": f"http://localhost:8000/images/{unique_name}",
        "file_path": path
    }

# API to parse document and store it in vector DB
@app.post("/parse-document")
async def parse_document(data: dict = Body(...)):
    t8 = time.time()
    
    print("parse document called -----------------------------------------")
    file_path = data["file_path"]
    collection_name = data["collection_name"]
    print("Page processing started ------------------------------------------")
    t1 = time.time()
    pages = load_pages(file_path)
    results = []
    for page_idx, image in pages:
        res = process_page_parallel(models, page_idx, image)
        results.append(res)
    print(f"Page processing took  {time.time() - t1:.2f} s--------------------------------------------")
  
    print("converting result to docs -------------------------------------------")
    t2 = time.time()
    docs = get_docs_from_DB(results)
    print(f"Converting to docs took {time.time() - t2:.2f} s")
    print(docs)

    print(f"Storing docs in vector DB -----------------------------------------")
    t3 = time.time()
    vectordb = get_vector_db(models, collection_name)
    ingest_document(vectordb, collection_name, docs)
    print(f"Storing into vector DB took {time.time() - t3:.2f} s")

    print(f"Total time {time.time() - t8:.2f} s")

    return {
        "status": "success",
        "message": "parse-document endpoint reached",
        "file_path": file_path,
        "collection_name": collection_name
    }


# Call Chat API when message sent to user
@app.post("/chat")
async def chat(req: ChatRequest):

    vectordb = get_vector_db(models, req.collection_name)

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    print(retriever)

    answer = get_answer(retriever, req.query)

    return {
        "answer": answer,
    }
