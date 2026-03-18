import base64
from io import BytesIO
from PIL import Image
from typing import List
import cv2
import re
import json
from dataclasses import dataclass
from charts_table_tools import AnalyzeChart, AnalyzeTable
import time

@dataclass
class LayoutRegion:
    region_id: int
    region_type: str
    bbox: list  # [x1, y1, x2, y2]
    confidence: float

def extract_valid_json(text):
    # find first JSON block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found")
    json_str = match.group(0)
    # validate JSON
    parsed_json = json.loads(json_str)
    return parsed_json

def layout_regions_to_docs(regions):
    docs = []
    for r in regions:
        docs.append({
            "text": json.dumps(r["content"]),   # embed structured content
            "metadata": {
                "type": r["region_type"],
                "region_id": r["region_id"],
                "bbox": r["bbox"],
                "confidence": r["confidence"]
            }
        })

    return docs

# Crop and save layout regions for agent tools
def crop_region(image, bbox, padding=10):
    """Crop a region from image with optional padding."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1 - padding))
    y1 = max(0, int(y1 - padding))
    x2 = min(image.width, int(x2 + padding))
    y2 = min(image.height, int(y2 + padding))
    return image.crop((x1, y1, x2, y2))

def image_to_base64(img):
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Process document layout 
def process_document(layout_engine, img):
    """Get layout regions from document."""
    layout_result = layout_engine.predict(img)
    
    regions = []
    for box in layout_result[0]['boxes']:
        regions.append({
            'label': box['label'],
            'score': box['score'],
            'bbox': box['coordinate'],  # [x1, y1, x2, y2]
        })
    
    # Sort by confidence
    regions = sorted(regions, key=lambda x: x['score'], reverse=True)
    return regions

def get_layout_regions(layout_results):
    layout_regions: List[LayoutRegion] = []
    for i, r in enumerate(layout_results):
        layout_regions.append(LayoutRegion(
            region_id=i,
            region_type=r['label'],
            bbox=[int(x) for x in r['bbox']],
            confidence=r['score']
        ))
    return layout_regions

def get_region_images(pil_image, layout_regions):
    
    region_images = {}
    for region in layout_regions:
        cropped = crop_region(pil_image, region.bbox)
        region_images[region.region_id] = {
            "image": cropped,
            "base64": image_to_base64(cropped),
            "type": region.region_type,
            "bbox": region.bbox
        }
    return region_images

def get_final_regions(layout_regions, AnalyzeChart, AnalyzeTable, region_images):
    final_regions = []
    for r in layout_regions:
        if r.region_type not in ["table", "chart", "figure"]:
            continue

        region_data = {
            "region_id": r.region_id,
            "region_type": r.region_type,
            "bbox": r.bbox,
            "confidence": r.confidence,
            "content": None
        }

        if r.region_type == "table":
            txt = AnalyzeTable.invoke({"region_id": r.region_id, "region_images": region_images})
            content = extract_valid_json(txt)

        elif r.region_type in ["chart","figure"]:
            txt = AnalyzeChart.invoke({"region_id": r.region_id, "region_images": region_images})
            content = extract_valid_json(txt)

        region_data["content"] = content
        final_regions.append(region_data)
    return final_regions

    

def run_layout_pipeline(img, models):

    layout_engine = models["layout_engine"]
    print("Get layout Results---------------------")
    t1 = time.time()
    layout_results = process_document(layout_engine, img)
    print(f"Loayout result took - {time.time() - t1:.2f} s")
    # print(layout_results)
    print("Get Layout regions------------------------")
    t2 = time.time()
    layout_regions = get_layout_regions(layout_results)
    print(f"layout regions took - {time.time() - t2:.2f} s")
    # print(layout_regions)
    print("Getting region images cropped ----------------------------------")
    t3 = time.time()
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    region_images = get_region_images(pil_image, layout_regions)
    print(f"region images took {time.time() - t3:.2f} s ")
    # print(region_images)
    print("Getting final region-------------------------------------------")
    t4 = time.time()
    final_regions = get_final_regions(layout_regions, AnalyzeChart, AnalyzeTable, region_images)
    print(f"final regions took - {time.time() - t4:.2f} s")
    print("Layout region done")
    # print(final_regions)
    return final_regions