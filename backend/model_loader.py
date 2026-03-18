import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["GLOG_minloglevel"] = "3"
os.environ["FLAGS_use_mkldnn"] = "False"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

sys.path.append(os.path.abspath("../layoutreader"))

from paddleocr import PaddleOCR
from transformers import LayoutLMv3ForTokenClassification
# from v3.helpers import prepare_inputs, boxes2inputs, parse_logits
from paddleocr import LayoutDetection
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_models():

    models = {}
    print("model loader called")

    # Loading OCR model
    models["ocr"] = PaddleOCR(use_angle_cls=False,lang='en',det_db_box_thresh=0.3,det_db_unclip_ratio=1.8)

    # Loading Ordering layout mode 
    model_slug = "hantian/layoutreader"
    models["layout_model"] = LayoutLMv3ForTokenClassification.from_pretrained(model_slug)

    # Loading Layout Model
    models["layout_engine"] = LayoutDetection()

    # Loading embedding model
    models["embedding"] = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return models
