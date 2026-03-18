import os
import sys
sys.path.append(os.path.abspath("../layoutreader"))
from v3.helpers import prepare_inputs, boxes2inputs, parse_logits
import time

from dataclasses import dataclass

# Store OCR results in a structured format
@dataclass
class OCRRegion:
    text: str
    bbox: list  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    confidence: float
    
    @property
    def bbox_xyxy(self):
        """Return bbox as [x1, y1, x2, y2] format."""
        x_coords = [p[0] for p in self.bbox]
        y_coords = [p[1] for p in self.bbox]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

# GEt reading order of text in the document
def get_reading_order(ocr_regions, layout_model):
    """
    Use LayoutReader to determine reading order of OCR regions.
    Returns list of reading order positions for each region index.
    """
    # 1. Calculate image dimensions from bounding boxes (with padding)
    max_x = max_y = 0
    for region in ocr_regions:
        x1, y1, x2, y2 = region.bbox_xyxy
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)

    image_width = max_x * 1.1   # Add 10% padding
    image_height = max_y * 1.1

    # 2. Convert bboxes to LayoutReader format (normalized to 0-1000)
    boxes = []
    for region in ocr_regions:
        x1, y1, x2, y2 = region.bbox_xyxy
        # Normalize to 0-1000 range
        left = int((x1 / image_width) * 1000)
        top = int((y1 / image_height) * 1000)
        right = int((x2 / image_width) * 1000)
        bottom = int((y2 / image_height) * 1000)
        boxes.append([left, top, right, bottom])

    # 3. Prepare inputs
    inputs = boxes2inputs(boxes)
    inputs = prepare_inputs(inputs, layout_model)
    
    # 4. Run inference
    logits = layout_model(**inputs).logits.cpu().squeeze(0)
    
    # 5. Parse the model's outputs to get reading order
    reading_order = parse_logits(logits, len(boxes))

    return reading_order

# Get Ordered text from the doc using reading order
def get_ordered_text(ocr_regions, reading_order):
    """
    Return OCR text sorted by reading order
    as a single clean text string.
    """

    indexed_regions = [
        (reading_order[i], i, ocr_regions[i]) 
        for i in range(len(ocr_regions))
    ]

    # sort by reading position
    indexed_regions.sort(key=lambda x: x[0])

    # extract only text
    texts = []
    for _, _, region in indexed_regions:
        if region.text.strip():
            texts.append(region.text.strip())

    # merge into one natural paragraph
    return " ".join(texts)

# function to split text into word chunks
def split_into_word_chunks(text, chunk_size=100):

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

# Convert chunks to docs to embedd 
def text_chunks_to_docs(chunks):
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append({
            "text": chunk,
            "metadata": {
                "type": "text",
                "chunk_id": i
            }
        })
    return docs

def run_ocr_pipeline(img, models):

    ocr = models["ocr"]
    layout_model = models["layout_model"]
    t1 = time.time()
    result = ocr.ocr(img)
    print(f"Ocr took {time.time() - t1:.2f} s")
    print("Result -------------------------------------")
    # print(result)
    
    page = result[0]
    texts = page['rec_texts']     
    scores = page['rec_scores']    
    boxes = page['rec_polys']

    t2 = time.time()
    ocr_regions: List[OCRRegion] = []
    for text, score, box in zip(texts, scores, boxes):
        ocr_regions.append(OCRRegion(
            text=text, 
            bbox=box.astype(int).tolist(), 
            confidence=score
        ))
    print(f"ocr_regions took {time.time() - t2:.2f} s")
    print("OCR Regions-----------------------------------------")
    # print(ocr_regions)

    # Get reading order
    t3 = time.time()
    reading_order = get_reading_order(ocr_regions, layout_model)
    print(f"Reading order took {time.time() - t1:.2f} s")
    # print(f"Reading order determined for {len(reading_order)} regions")
    # print(f"First 20 positions: {reading_order[:20]}")

    t4 = time.time()
    ordered_text = get_ordered_text(ocr_regions, reading_order)
    print(f"Ordered text took {time.time() - t4:.2f} s")

    chunks = split_into_word_chunks(ordered_text)

    print("Chunks------------------------------------------------")
    for c in chunks:
        print("------------------------------------")
        print(c)

    return chunks