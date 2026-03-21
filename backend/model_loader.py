import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath("../layoutreader"))

from paddleocr import PaddleOCR
from paddleocr import LayoutDetection
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_models():

    models = {}
    print("model loader called")

    # Loading OCR model
    models["ocr"] = PaddleOCR(use_angle_cls=False,lang='en',det_db_box_thresh=0.3,det_db_unclip_ratio=1.8)

    # Loading Layout Model
    models["layout_engine"] = LayoutDetection()

    # Loading embedding model
    models["embedding"] = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return models
