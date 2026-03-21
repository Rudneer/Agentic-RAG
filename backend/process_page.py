from typing import List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from charts_table_tools import process_charts_tables

@dataclass
class LayoutRegion:
    region_id: int
    region_type: str
    bbox: list  # [x1, y1, x2, y2]
    confidence: float
    page: int 

# Crop image using bounding box
def crop_region(image, bbox, padding=40):
    x1, y1, x2, y2 = bbox

    h, w = image.shape[:2]

    # Apply padding safely
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    return image[y1:y2, x1:x2]

# EXtract text from ocr result
def extract_text_from_ocr(ocr_result):
    texts = []

    for block in ocr_result:
        texts.extend(block.get("rec_texts", []))

    return " ".join(texts).strip()

# Extract text from text layout regions
def process_text_region(ocr, region, image):
    crop = crop_region(image, region.bbox)

    ocr_result = ocr.ocr(crop)

    text = extract_text_from_ocr(ocr_result)

    return {
        "region_id": region.region_id,
        "page": region.page,
        "type": region.region_type,
        "bbox": region.bbox,
        "content": text,
        "confidence": region.confidence
    }

# Mask layout regions and perform ocr on remaining page
def mask_regions(image, regions, pad=5):
    h, w, _ = image.shape
    masked = image.copy()

    for r in regions:

        x1, y1, x2, y2 = map(int, r.bbox)

        # Add padding 
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        # Fill with white
        masked[y1:y2, x1:x2] = 255

    return masked

# Extract text from the masked image
def process_masked_image(ocr, image, page_idx):
    ocr_result = ocr.ocr(image)
    text = extract_text_from_ocr(ocr_result)
    return {
        "region_id": None,
        "page": page_idx,
        "type": "outside_layout",
        "bbox": None,
        "content": text,
        "confidence": None
    }

# Extract layout regions using layout model
def extract_regions(layout_engine, page_idx, image) -> List[LayoutRegion]:
    """Run layout detection and return structured regions."""

    layout_result = layout_engine.predict(image)
    boxes = layout_result[0]['boxes']

    regions = []

    for i, box in enumerate(boxes):
        regions.append(
            LayoutRegion(
                region_id=i,
                region_type=box['label'],
                bbox=[int(x) for x in box['coordinate']],
                confidence=float(box['score']),
                page=page_idx
            )
        )

    # Sort by confidence (optional)
    regions.sort(key=lambda x: x.confidence, reverse=True)
    return regions

# Perform Parallel tasks 1. OCR, 2. LLM analysis for charts/table
def process_page_parallel(models, page_idx, image):
    ocr = models["ocr"]

    layout_engine = models["layout_engine"]
    regions = extract_regions(layout_engine, page_idx, image)

    txt_list = []
    final_regions = []
    text_regions = []
    llm_regions = []

    for r in regions:
        if r.region_type in ["table", "figure", "chart", "equation"]:
            llm_regions.append(r)
        else:
            text_regions.append(r)

    # Start LLM in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_llm = executor.submit(
            lambda: [process_charts_tables(r, image) for r in llm_regions]
        )

        # OCR sequential 
        for r in text_regions:
            result = process_text_region(ocr, r, image)
            txt_list.append(result)

        # Masked OCR
        masked_img = mask_regions(image, regions)
        m_text = process_masked_image(ocr, masked_img, page_idx)

        if m_text["content"] and m_text["content"].strip():
            txt_list.append(m_text)

        # Wait for LLM
        final_regions = future_llm.result()

    return {
        "page": page_idx,
        "text_regions": txt_list,
        "structured_regions": final_regions
    }