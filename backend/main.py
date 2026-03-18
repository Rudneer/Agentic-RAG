from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from model_loader import load_models
from image_tools import enhance_image
from ocr_pipeline import run_ocr_pipeline, text_chunks_to_docs
from layout_pipeline import run_layout_pipeline, layout_regions_to_docs
from vectordb_helper import get_vector_db, get_docs_for_DB, get_answer, ingest_document
import os
import asyncio
import time
import uuid
from pydantic import BaseModel

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
    t6 = time.time()

    file_path = data["file_path"]
    collection_name = data["collection_name"]

    print(file_path)

    print("Image processing started--------------------------------------")
    t1 = time.time()
    # run tasks in parallel
    img = enhance_image(file_path)
    print(f"Image processing took {time.time() - t1:.2f} s")
    print("OCR pipeline started ----------------------------------------")

    vectordb_task = asyncio.to_thread(get_vector_db, models, collection_name)

    t2 = time.time()
    ocr_task = asyncio.to_thread(run_ocr_pipeline, img, models)
    print(f"OCR task took  {time.time() - t2:.2f} s")

    print("Layout Region--------------------------------------------------")
    t3 = time.time()
    layout_task = asyncio.to_thread(run_layout_pipeline, img, models)
    print(f"Layout pprocess took {time.time() - t3:.2f} s")

    print("await---------------------------------")
    # Run task in parallel
    t5 = time.time()
    ocr_result, layout_result, vectordb  = await asyncio.gather(
        ocr_task,
        layout_task,
        vectordb_task
    )

    print(f"paralle process took {time.time() - t5:.2f} s")
    print("Docs processed------------------------------")

    t4 = time.time()
    vector_docs = []

    vector_docs.extend(text_chunks_to_docs(ocr_result))
    vector_docs.extend(layout_regions_to_docs(layout_result))

    docs = get_docs_for_DB(vector_docs)
    print(f"Vector docs took {time.time() - t4:.2f} s")
    print(vector_docs)

    t7 = time.time()
    # print(vectordb)
    # vectordb = get_vector_db(models, collection_name)
    ingest_document(vectordb, collection_name, docs)
    print(f"Storing into vector DB took {time.time() - t7:.2f} s")

    print(f"Total time {time.time() - t6:.2f} s")

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
