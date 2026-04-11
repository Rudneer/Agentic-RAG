import cv2
import fitz
import numpy as np
from pathlib import Path

def add_padding(img, pad=40):
    return cv2.copyMakeBorder(
        img,
        pad, pad, pad, pad,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

# Resize for Better Detection
def upscale_image(img, target_width=1400):
    h, w = img.shape[:2]
    if w < target_width:
        scale = target_width / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img

# Gentle Contrast Enhancement
def enhance_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    return cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)

def preprocess_image(img, max_side=1024):
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    img = cv2.resize(img, None, fx=scale, fy=scale)
    return img

def load_pages(file_path):
    ext = Path(file_path).suffix.lower()

    # ---------- PDF ----------
    if ext == ".pdf":
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = np.frombuffer(
                pix.samples,
                dtype=np.uint8
            ).reshape(pix.height, pix.width, pix.n)
            yield i, img

    #---------- IMAGE ----------
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Invalid image")
        yield 0, img
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def enhance_image(img):
    img = preprocess_image(img)
    img = add_padding(img, pad=40)
    img = enhance_contrast(img)
    return img
