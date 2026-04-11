import os, re, json, cv2, base64
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize VLM for tools
vlm = ChatGroq(api_key = GROQ_API_KEY, model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

# Extract valid json from llms response
def extract_valid_json(text):

    # Extract JSON inside ```json ``` blocks if present
    match = re.search(r"```json(.*?)```", text, re.DOTALL)

    if match:
        text = match.group(1).strip()

    # Find first JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON found in response")

    json_str = text[start:end+1]

    try:
        return json.loads(json_str)

    except json.JSONDecodeError:
        # fallback: try to fix trailing commas
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)
        return json.loads(json_str)

# Crop image and  convert it to base64
def crop_to_base64(image, bbox):
    x1, y1, x2, y2 = bbox

    # Crop image
    crop = image[y1:y2, x1:x2]

    # Encode to PNG in memory
    _, buffer = cv2.imencode(".png", crop)

    # Convert to base64
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return img_base64

# Tool prompts
CHART_ANALYSIS_PROMPT = """You are a Chart Analysis specialist. 
Analyze this chart/figure image and extract:

1. **Chart Type**: (line, bar, scatter, pie, etc.)
2. **Title**: (if visible)
3. **Axes**: X-axis label, Y-axis label, and tick values
4. **Data Points**: Key values (peaks, troughs, endpoints)
5. **Trends**: Overall pattern description
6. **Legend**: (if present)

Return a JSON object with this structure:
```json
{{
  "chart_type": "...",
  "title": "...",
  "x_axis": {{"label": "...", "ticks": [...]}},
  "y_axis": {{"label": "...", "ticks": [...]}},
  "key_data_points": [...],
  "trends": "...",
  "legend": [...]
}}
```
"""

# Merged cells,
TABLE_ANALYSIS_PROMPT = """You are a Table Extraction specialist. 
Extract structured data from this table image.

1. **Identify Structure**: 
    - Column headers, row labels, data cells
2. **Extract All Data**: 
    - Preserve exact values and alignment
3. **Handle Special Cases**: 
    - empty cells (mark as null), multi-line headers
    - If debit and credit both appear in a single row in a bank statement, it is likely that two rows were merged. Split them into two separate rows.

Return a JSON object with this structure:
```json
{{
  "table_title": "...",
  "column_headers": ["header1", "header2", ...],
  "rows": [
    {{"row_label": "...", "values": [val1, val2, ...]}},
    ...
  ],
  "notes": "any footnotes or source info"
}}
```
"""

def call_vlm_with_image(image_base64, prompt):
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url":{"url": f"data:image/png;base64,{image_base64}"}
            }
        ]
    )

    response = vlm.invoke([message])
    return response.content


@tool
def AnalyzeChart(base_64: str) -> str:
    """Analyze a chart or figure region using VLM. 
    Use this tool when you need to extract data from charts, graphs, or figures.
    
    Args:
        base_64: Base_64 of layout region to analyze
    
    Returns:
        JSON string with chart type, axes, data points, and trends
    """
    
    result = call_vlm_with_image(base_64, CHART_ANALYSIS_PROMPT)
    
    return result



@tool
def AnalyzeTable(base_64: str) -> str:
    """
    Extract structured data from a table region using VLM.
    Use this tool when you need to extract tabular data 
    with headers and rows.
    
    Args:
        base_64: Base_64 of layout region to analyze
    
    Returns:
        JSON string with table headers, rows, and any notes
    """
    
    result = call_vlm_with_image(base_64, TABLE_ANALYSIS_PROMPT)
    return result

print("AnalyzeTable tool defined")

# Extract json data for tables and charts
def process_charts_tables(r, image):
    region_data = {
            "region_id": r.region_id,
            "page": r.page,
            "type": r.region_type,
            "bbox": r.bbox,
            "content": None,
            "confidence": r.confidence
        }

    base_64 = crop_to_base64(image, r.bbox)
    
    if r.region_type == "table":
        txt = AnalyzeTable.invoke({"base_64": base_64})
        content = extract_valid_json(txt)

    elif r.region_type in ["chart","figure","image"]:
        txt = AnalyzeChart.invoke({"base_64": base_64})
        content = extract_valid_json(txt)

    region_data["content"] = content
    return region_data