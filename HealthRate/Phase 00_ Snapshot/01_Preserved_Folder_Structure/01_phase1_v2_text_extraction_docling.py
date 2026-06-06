"""
01_phase1_text_extraction_docling.py

Phase 1: Docling-based text extraction with VLM descriptions for pictures/graphs.

Input:
    project_root/
        Phase 00_ Snapshot/
            02_All_Files_Flat/
                your files...

Output:
    project_root/
        Phase 01_TextExtraction/
            file_name/
                docling_crops/
                    pictures/
                    tables/
                file.md
                file_llm_feed.md
                outlined_file.pdf

What this script does:
1. Reads files from Phase 00_ Snapshot / 02_All_Files_Flat
2. Uses Docling to parse each file
3. Saves Docling crops:
   - pictures/graphs for VLM
   - tables for debugging only
4. Uses a local Ollama text LLM to clean Docling-generated table text
5. Uses a local Ollama VLM to describe pictures/graphs
6. Optionally uses OCR on pictures/graphs for exact text
7. Writes one full debug markdown file: file.md
8. Writes one clean LLM/RAG markdown file: file_llm_feed.md
9. Writes one outlined PDF: outlined_file.pdf

Important:
- Tables are first extracted by Docling.
- Table text is then cleaned by a text LLM.
- VLM is only used for pictures/figures/graphs.
- file.md keeps image crop file paths and placeholder image text for debugging.
- file_llm_feed.md removes crop paths, removes skipped/empty image sections, and uses Image Context/Image Text labels for cleaner LLM/RAG input.

Install:
    pip install docling pillow requests

Optional OCR:
    pip install pytesseract
    Install Tesseract itself if you want OCR:
    https://github.com/UB-Mannheim/tesseract/wiki

Example:
    Run this script from your project folder:

    cd "C:\\Users\\amenx\\Desktop\\all phases re-engineered"
    python 01_phase1_text_extraction_docling_v2.py --vlm-model qwen2.5vl:7b

Test only 3 files:
    python 01_phase1_text_extraction_docling_v2.py --vlm-model qwen2.5vl:7b --limit 3

Skip VLM:
    python 01_phase1_text_extraction_docling_v2.py --skip-vlm --skip-ocr
"""

import argparse
import base64
import json
import re
import subprocess
from pathlib import Path

import requests
from PIL import Image, ImageDraw

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

try:
    from docling_core.types.doc import PictureItem, TableItem
except Exception:
    PictureItem = None
    TableItem = None

try:
    import pytesseract
except Exception:
    pytesseract = None


# ----------------------------
# CONFIG
# ----------------------------

PHASE0_FOLDER_NAME = "Phase 00_ Snapshot"
FLAT_FILES_FOLDER_NAME = "02_All_Files_Flat"

PHASE1_FOLDER_NAME = "Phase 01_TextExtraction"

IMAGE_RESOLUTION_SCALE = 2.0

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".xls",
    ".csv",
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tif",
    ".tiff",
}

COLOR_BY_KIND = {
    "table": "red",
    "picture": "blue",
    "text": "green",
    "section_header": "orange",
    "list_item": "purple",
    "unknown": "gray",
}


# ----------------------------
# BASIC HELPERS
# ----------------------------

def clean_name(name):
    """
    Make a safe folder name from a file name.

    Example:
        NeurAegis_InvestorDeck_2025.pdf
    becomes:
        neuraegis-investordeck-2025
    """

    path = Path(name)
    stem = path.stem.lower().strip()

    cleaned = []

    for ch in stem:
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in {" ", "-", "_", ".", "(", ")"}:
            cleaned.append("-")

    result = "".join(cleaned)

    while "--" in result:
        result = result.replace("--", "-")

    return result.strip("-") or "document"


def safe_mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_text(text):
    """
    Light cleanup for LLM-friendly text.
    """

    if text is None:
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")

    # Remove common HTML line-break noise from parser outputs.
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)

    # Clean extra spaces on each line.
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]{2,}", " ", line).rstrip()
        lines.append(line)

    text = "\n".join(lines)

    # Reduce excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def find_files(input_folder):
    """
    Find all supported files from Phase 0 flat folder.
    """

    files = []

    for path in input_folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    return sorted(files, key=lambda x: str(x).lower())


def obj_to_dict(obj):
    """
    Try to convert pydantic/dataclass-like objects into a normal dict.
    """

    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    for method_name in ["model_dump", "dict"]:
        method = getattr(obj, method_name, None)

        if callable(method):
            try:
                return method()
            except Exception:
                pass

    data = {}

    for name in dir(obj):
        if name.startswith("_"):
            continue

        try:
            value = getattr(obj, name)
        except Exception:
            continue

        if callable(value):
            continue

        if isinstance(value, (str, int, float, bool, type(None))):
            data[name] = value

    return data


# ----------------------------
# DOCLING ITEM HELPERS
# ----------------------------

def get_first_prov(element):
    prov = getattr(element, "prov", None)

    if not prov:
        return None

    try:
        return prov[0]
    except Exception:
        return None


def get_page_no(element):
    prov = get_first_prov(element)

    if prov is None:
        return None

    return getattr(prov, "page_no", None)


def get_element_kind(element):
    class_name = element.__class__.__name__.lower()
    label = str(getattr(element, "label", "")).lower()

    if TableItem is not None and isinstance(element, TableItem):
        return "table"

    if PictureItem is not None and isinstance(element, PictureItem):
        return "picture"

    if "table" in class_name or "table" in label:
        return "table"

    if "picture" in class_name or "image" in class_name or "figure" in class_name:
        return "picture"

    if "section" in class_name or "header" in class_name or "title" in label:
        return "section_header"

    if "list" in class_name or "list" in label:
        return "list_item"

    if "text" in class_name or "text" in label:
        return "text"

    return "unknown"


def get_element_text(element):
    """
    Get readable text from a Docling element.
    """

    for attr in ["text", "caption", "name"]:
        value = getattr(element, attr, None)

        if isinstance(value, str) and value.strip():
            return normalize_text(value)

    return ""


def render_table_markdown(element):
    """
    Render a Docling table as markdown if possible.

    Docling versions differ, so this tries several fallbacks.
    """

    # Best case: table object can export itself.
    for method_name in ["export_to_markdown", "to_markdown"]:
        method = getattr(element, method_name, None)

        if callable(method):
            try:
                return normalize_text(method())
            except Exception:
                pass

    # Common case: table has data object.
    data = getattr(element, "data", None)

    if data is not None:
        for method_name in ["export_to_markdown", "to_markdown"]:
            method = getattr(data, method_name, None)

            if callable(method):
                try:
                    return normalize_text(method())
                except Exception:
                    pass

        # Fallback: turn dict/list-looking data into text.
        data_dict = obj_to_dict(data)

        if data_dict:
            return "```json\n" + json.dumps(data_dict, indent=2, ensure_ascii=False) + "\n```"

    # Last fallback.
    text = get_element_text(element)

    if text:
        return text

    return "[Table detected, but no table text was available from Docling]"


def get_picture_caption(element):
    caption = getattr(element, "caption", None)

    if isinstance(caption, str) and caption.strip():
        return normalize_text(caption)

    captions = getattr(element, "captions", None)

    if captions:
        parts = []
        for cap in captions:
            cap_text = get_element_text(cap)
            if cap_text:
                parts.append(cap_text)

        if parts:
            return normalize_text("\n".join(parts))

    return ""


# ----------------------------
# BBOX / OUTLINE HELPERS
# ----------------------------

def get_bbox_raw(element):
    prov = get_first_prov(element)

    if prov is None:
        return None, None

    page_no = getattr(prov, "page_no", None)
    bbox = getattr(prov, "bbox", None)

    return page_no, bbox


def get_bbox_numbers(bbox):
    bbox_dict = obj_to_dict(bbox)

    left = bbox_dict.get("l") or bbox_dict.get("left") or bbox_dict.get("x0") or bbox_dict.get("x_min")
    right = bbox_dict.get("r") or bbox_dict.get("right") or bbox_dict.get("x1") or bbox_dict.get("x_max")
    top = bbox_dict.get("t") or bbox_dict.get("top") or bbox_dict.get("y0") or bbox_dict.get("y_min")
    bottom = bbox_dict.get("b") or bbox_dict.get("bottom") or bbox_dict.get("y1") or bbox_dict.get("y_max")

    origin = bbox_dict.get("coord_origin") or bbox_dict.get("origin") or getattr(bbox, "coord_origin", None)

    if None in [left, top, right, bottom]:
        for method_name in ["as_tuple", "to_tuple"]:
            method = getattr(bbox, method_name, None)

            if callable(method):
                try:
                    values = method()

                    if len(values) >= 4:
                        left, top, right, bottom = values[:4]
                        break

                except Exception:
                    pass

    if None in [left, top, right, bottom]:
        return None

    return float(left), float(top), float(right), float(bottom), str(origin).lower()


def page_size_to_tuple(page):
    size = getattr(page, "size", None)

    if size is None:
        return None

    width = getattr(size, "width", None)
    height = getattr(size, "height", None)

    if width is None or height is None:
        size_dict = obj_to_dict(size)
        width = size_dict.get("width")
        height = size_dict.get("height")

    if width is None or height is None:
        return None

    return float(width), float(height)


def bbox_to_image_xyxy(bbox, page, page_image):
    nums = get_bbox_numbers(bbox)

    if nums is None:
        return None

    left, top, right, bottom, origin = nums

    page_size = page_size_to_tuple(page)

    if page_size is None:
        return None

    page_w, page_h = page_size
    img_w, img_h = page_image.size

    x_scale = img_w / page_w
    y_scale = img_h / page_h

    x1 = left * x_scale
    x2 = right * x_scale

    if "bottom" in origin:
        y1 = img_h - (top * y_scale)
        y2 = img_h - (bottom * y_scale)
    else:
        y1 = top * y_scale
        y2 = bottom * y_scale

    x_min = max(0, min(x1, x2))
    x_max = min(img_w, max(x1, x2))
    y_min = max(0, min(y1, y2))
    y_max = min(img_h, max(y1, y2))

    if x_max <= x_min or y_max <= y_min:
        return None

    return int(x_min), int(y_min), int(x_max), int(y_max)


def create_outlined_pdf(doc, output_pdf_path):
    """
    Create outlined_file.pdf.

    It renders page images, draws Docling layout boxes, then combines the pages
    into a single PDF.
    """

    outlined_pages = []

    for page_no, page in doc.pages.items():
        page_no = page.page_no

        if not getattr(page, "image", None) or not getattr(page.image, "pil_image", None):
            continue

        page_image = page.image.pil_image.convert("RGB")
        draw = ImageDraw.Draw(page_image)

        for item_index, (element, level) in enumerate(doc.iterate_items(), start=1):
            item_page_no, bbox = get_bbox_raw(element)

            if item_page_no != page_no or bbox is None:
                continue

            box = bbox_to_image_xyxy(bbox, page, page_image)

            if box is None:
                continue

            kind = get_element_kind(element)
            color = COLOR_BY_KIND.get(kind, COLOR_BY_KIND["unknown"])

            draw.rectangle(box, outline=color, width=4)

            label = f"{item_index}: {kind}"
            text_x = box[0]
            text_y = max(0, box[1] - 20)

            try:
                draw.rectangle(
                    [text_x, text_y, text_x + len(label) * 8 + 10, text_y + 20],
                    fill=color,
                )
                draw.text((text_x + 5, text_y + 3), label, fill="white")
            except Exception:
                pass

        outlined_pages.append(page_image)

    if not outlined_pages:
        return False

    first_page = outlined_pages[0]
    remaining_pages = outlined_pages[1:]

    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    first_page.save(
        output_pdf_path,
        save_all=True,
        append_images=remaining_pages,
    )

    return True


# ----------------------------
# DOCLING CONVERSION
# ----------------------------

def build_docling_converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    return converter


def convert_with_docling(source_file):
    converter = build_docling_converter()
    return converter.convert(source_file)


# ----------------------------
# IMAGE / OCR / VLM HELPERS
# ----------------------------

def save_docling_crops(doc, crops_root):
    """
    Save Docling table and picture crops.

    Tables are saved only for debugging.
    Pictures are used for VLM image/graph explanation.
    """

    picture_dir = safe_mkdir(crops_root / "pictures")
    table_dir = safe_mkdir(crops_root / "tables")

    picture_crops = {}
    table_crops = {}

    picture_count = 0
    table_count = 0

    for item_index, (element, level) in enumerate(doc.iterate_items(), start=1):
        kind = get_element_kind(element)
        page_no = get_page_no(element)

        if kind not in {"picture", "table"}:
            continue

        try:
            image = element.get_image(doc)

            if image is None:
                continue

            if kind == "picture":
                picture_count += 1
                out_path = picture_dir / f"page_{page_no:03d}_picture_{picture_count:03d}.png"
                image.save(out_path, "PNG")
                picture_crops[item_index] = out_path

            elif kind == "table":
                table_count += 1
                out_path = table_dir / f"page_{page_no:03d}_table_{table_count:03d}.png"
                image.save(out_path, "PNG")
                table_crops[item_index] = out_path

        except Exception:
            continue

    return picture_crops, table_crops


def run_image_ocr(image_path, tesseract_cmd=None, ocr_lang="eng"):
    """
    Run OCR on a picture/graph crop.

    This is separate from Docling's own OCR.
    It is only for exact text inside images/graphs.
    """

    if pytesseract is None:
        return "[OCR skipped: pytesseract is not installed]"

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        image = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(image, lang=ocr_lang)
        text = normalize_text(text)

        if not text:
            return "[no meaningful text found]"

        return text

    except Exception as e:
        return f"[OCR failed: {e}]"


def call_ollama_vlm(image_path, model, ollama_url, prompt):
    """
    Call a local Ollama vision model using /api/chat.
    """

    image_bytes = Path(image_path).read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
    }

    url = ollama_url.rstrip("/") + "/api/chat"

    response = requests.post(url, json=payload, timeout=240)
    response.raise_for_status()

    data = response.json()

    content = data.get("message", {}).get("content", "")
    content = normalize_text(content)

    if not content:
        return "[no meaningful visual description found]"

    return content


def describe_picture_with_vlm(image_path, vlm_model, ollama_url):
    """
    Ask VLM to describe a picture/graph for LLM ingestion.
    """

    prompt = """
You are helping prepare document images for an LLM/RAG system.

Describe this image, figure, chart, graph, or screenshot in clean factual text.

Rules:
- Preserve important labels, numbers, company names, product names, and chart meaning.
- If it is a graph, explain what the axes/labels/trend appear to show.
- If it is a diagram, explain the parts and relationships.
- If it is just decorative or a logo, say that briefly.
- Do not invent missing facts.
- Keep the output useful for search and embeddings.
""".strip()

    try:
        return call_ollama_vlm(
            image_path=image_path,
            model=vlm_model,
            ollama_url=ollama_url,
            prompt=prompt,
        )

    except Exception as e:
        return f"[image description failed: {e}]"


def build_image_block(description, ocr_text, crop_path=None):
    """
    Build the final image block inserted into file.md.
    """

    description = normalize_text(description)
    ocr_text = normalize_text(ocr_text)

    if not description:
        description = "[no meaningful visual description found]"

    if not ocr_text:
        ocr_text = "[no meaningful text found]"

    parts = []

    parts.append("Image description:")
    parts.append(description)
    parts.append("")
    parts.append("Image OCR text:")
    parts.append(ocr_text)

    if crop_path is not None:
        parts.append("")
        parts.append(f"Image crop file: {crop_path}")

    return "\n".join(parts).strip()



def make_no_crop_llm_markdown(markdown_text):
    """
    Create the second markdown file for cleaner LLM/RAG input.

    Changes:
    - removes "Image crop file:" lines
    - changes "Image description:" to "Image Context:"
    - changes "Image OCR text:" to "Image Text:"
    """

    lines = []

    for line in markdown_text.splitlines():
        stripped = line.strip()

        if stripped.lower().startswith("image crop file:"):
            continue

        if stripped == "Image description:":
            lines.append("Image Context:")
            continue

        if stripped == "Image OCR text:":
            lines.append("Image Text:")
            continue

        lines.append(line)

    return "\n".join(lines).strip() + "\n"



def write_both_markdown_files(output_folder, final_md):
    """
    Write both markdown outputs for Phase 1.

    file.md:
        Keeps Image crop file paths for debugging.

    file_llm_feed.md:
        Removes Image crop file paths and changes image labels to:
            Image Text:
            Image Context:
    """

    markdown_path = output_folder / "file.md"
    markdown_path.write_text(final_md, encoding="utf-8")

    llm_feed_md = make_no_crop_llm_markdown(final_md)
    llm_feed_markdown_path = output_folder / "file_llm_feed.md"
    llm_feed_markdown_path.write_text(llm_feed_md, encoding="utf-8")

    return markdown_path, llm_feed_markdown_path


def call_ollama_text_llm(prompt, model, ollama_url):
    """
    Call a local Ollama text model using /api/chat.

    This is used for cleaning Docling-generated table text.
    It is separate from the VLM image description function.
    """

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    url = ollama_url.rstrip("/") + "/api/chat"

    response = requests.post(url, json=payload, timeout=240)
    response.raise_for_status()

    data = response.json()
    content = data.get("message", {}).get("content", "")
    content = normalize_text(content)

    if not content:
        return "[LLM table cleanup returned no text]"

    return content


def clean_table_with_llm(table_text, table_context="", table_llm_model="qwen3:14b", ollama_url="http://localhost:11434"):
    """
    Clean Docling-generated table text for the final markdown file.

    Main goal:
    Do NOT preserve the table just because it is a table.
    Rewrite the table into semantic, LLM-friendly lines.

    Preferred formats:

    1. For simple category-pair tables:
        Acute: Concussion; Chronic: Epilepsy
        Acute: TBI; Chronic: Parkinson's

    2. For comparison tables:
        Current Therapies — Concussion: Currently no FDA approved medications
        Current Therapies — Status Epilepticus: Currently no FDA approved medications
        Current Therapies — Glaucoma: Currently no FDA approved medications

    3. For record tables:
        Row 1 — Company: ABC; Stage: Seed; Sector: AI; Location: San Diego

    This is better for embeddings because every line becomes self-contained.
    """

    table_text = normalize_text(table_text)
    table_context = normalize_text(table_context)

    if not table_text:
        return "[Table detected, but no table text was available]"

    prompt = f"""
You are converting extracted table text into semantic text for an LLM/RAG pipeline.

IMPORTANT:
Do NOT output a markdown table unless absolutely necessary.
Do NOT try to preserve the visual table format.
Instead, flatten the table into clear self-contained lines.

Your goal:
Make every line understandable by itself for search, embeddings, and LLM retrieval.

Rules:
- Preserve all factual information.
- Preserve numbers, disease names, drug names, company names, funding names, dates, and status terms.
- Do not invent missing information.
- Do not summarize away important details.
- Fix obvious spacing problems, for example:
  "fundedbyDOD" -> "funded by DOD"
  "IND- enabling" -> "IND-enabling"
- Remove extraction noise, empty rows, repeated separators, and awkward line breaks.
- Prefer semantic key-value lines over markdown tables.
- Output only the cleaned table text.
- Do not explain what you did.

Choose the best format:

FORMAT A: Simple category-pair table
Use this when the table has category columns like Acute/Chronic.

Example:
Acute: Concussion; Chronic: Epilepsy
Acute: TBI; Chronic: Parkinson's
Acute: Stroke; Chronic: ALS

FORMAT B: Comparison table
Use this when the table compares columns across row labels.

Example:
Current Therapies — Concussion: Currently no FDA approved medications
Current Therapies — Status Epilepticus: Currently no FDA approved medications
Current Therapies — Glaucoma: Currently no FDA approved medications

Overall Incidence — Concussion: CDC estimates 3.8 million concussions occur in the U.S. annually
Overall Incidence — Status Epilepticus: 40 per 100,000 per year in the U.S. annually
Overall Incidence — Glaucoma: CDC estimates 0.6% of the global population

FORMAT C: Record table
Use this when each row is one entity/record.

Example:
Record 1 — Company: ABC Therapeutics; Stage: Seed; Sector: Biotech; Funding: $2M
Record 2 — Company: XYZ Robotics; Stage: Series A; Sector: Robotics; Funding: $8M

Optional nearby context:
{table_context}

Raw table text:
{table_text}
""".strip()

    try:
        return call_ollama_text_llm(
            prompt=prompt,
            model=table_llm_model,
            ollama_url=ollama_url,
        )

    except Exception as e:
        return (
            "Table cleanup failed. Using original Docling table text.\n\n"
            f"Cleanup error: {e}\n\n"
            f"{table_text}"
        )


# ----------------------------
# MARKDOWN RENDERING
# ----------------------------

def looks_like_placeholder_image_text(text):
    """
    Return True if an image/VLM/OCR string is just placeholder noise.

    These placeholder strings are useful in file.md for debugging,
    but they are not useful in file_llm_feed.md for embeddings/RAG.
    """

    text = normalize_text(text)

    if not text:
        return True

    lowered = text.lower().strip()

    placeholder_fragments = [
        "[vlm skipped]",
        "[ocr skipped]",
        "[ocr skipped:",
        "[no meaningful text found]",
        "[no meaningful visual description found]",
        "[picture/figure detected, but no crop file was available]",
        "[llm table cleanup returned no text]",
        "[image description failed:",
        "[ocr failed:",
        "no meaningful text found",
        "no meaningful visual description found",
    ]

    for fragment in placeholder_fragments:
        if fragment in lowered:
            return True

    # If OCR is mostly symbols or a single character, it usually hurts more than it helps.
    alnum_count = len(re.findall(r"[A-Za-z0-9]", text))
    if alnum_count < 3:
        return True

    return False


def build_clean_image_llm_block(caption="", description="", ocr_text=""):
    """
    Build the image block for file_llm_feed.md.

    Clean output labels:
        Image Context:
        Image Text:

    Main rule:
    Keep image sections only when they contain real information.
    Remove:
        [VLM skipped]
        [no meaningful text found]
        crop paths
        empty image placeholders
    """

    caption = normalize_text(caption)
    description = normalize_text(description)
    ocr_text = normalize_text(ocr_text)

    context_parts = []

    if not looks_like_placeholder_image_text(caption):
        context_parts.append(f"Caption: {caption}")

    if not looks_like_placeholder_image_text(description):
        context_parts.append(description)

    useful_ocr = not looks_like_placeholder_image_text(ocr_text)

    if not context_parts and not useful_ocr:
        return ""

    lines = []

    if context_parts:
        lines.append("Image Context:")
        lines.append("\n\n".join(context_parts))
        lines.append("")

    if useful_ocr:
        lines.append("Image Text:")
        lines.append(ocr_text)
        lines.append("")

    return "\n".join(lines).strip()


def append_to_both(debug_lines, llm_lines, text_value):
    """
    Append normal document text to both markdown versions.
    """

    text_value = normalize_text(text_value)

    if not text_value:
        return

    debug_lines.append(text_value)
    debug_lines.append("")

    llm_lines.append(text_value)
    llm_lines.append("")


def append_page_marker(debug_lines, llm_lines, page_no):
    """
    Add page marker to both markdown versions.
    """

    for lines in [debug_lines, llm_lines]:
        lines.append("")
        lines.append("---")
        lines.append(f"Page {page_no}")
        lines.append("---")
        lines.append("")


def render_docling_to_markdown_versions(
    doc,
    source_file,
    picture_crops,
    skip_vlm=False,
    vlm_model="qwen2.5vl:7b",
    ollama_url="http://localhost:11434",
    skip_ocr=False,
    tesseract_cmd=None,
    ocr_lang="eng",
    skip_table_llm=False,
    table_llm_model="qwen3:14b",
):
    """
    Render the Docling document into TWO markdown files.

    file.md:
        Full debug version.
        Keeps:
            Image caption:
            Image description:
            Image OCR text:
            Image crop file:

    file_llm_feed.md:
        Clean LLM/RAG version.
        Keeps only useful image content.
        Uses:
            Image Context:
            Image Text:
        Removes:
            Image crop file paths
            [VLM skipped]
            [no meaningful text found]
            empty image blocks
    """

    debug_lines = []
    llm_lines = []

    debug_lines.append(f"# {source_file.name}")
    debug_lines.append("")

    llm_lines.append(f"# {source_file.name}")
    llm_lines.append("")

    current_page = None

    for item_index, (element, level) in enumerate(doc.iterate_items(), start=1):
        kind = get_element_kind(element)
        page_no = get_page_no(element)

        if page_no is not None and page_no != current_page:
            current_page = page_no
            append_page_marker(debug_lines, llm_lines, page_no)

        if kind == "section_header":
            text = get_element_text(element)

            if text:
                append_to_both(debug_lines, llm_lines, f"## {text}")

        elif kind == "list_item":
            text = get_element_text(element)

            if text:
                append_to_both(debug_lines, llm_lines, f"- {text}")

        elif kind == "text":
            text = get_element_text(element)

            if text:
                append_to_both(debug_lines, llm_lines, text)

        elif kind == "table":
            table_md = render_table_markdown(element)

            if skip_table_llm:
                cleaned_table = table_md
            else:
                cleaned_table = clean_table_with_llm(
                    table_text=table_md,
                    table_context=f"Document: {source_file.name}\nPage: {page_no}",
                    table_llm_model=table_llm_model,
                    ollama_url=ollama_url,
                )

            append_to_both(debug_lines, llm_lines, "Table:")
            append_to_both(debug_lines, llm_lines, cleaned_table)

        elif kind == "picture":
            crop_path = picture_crops.get(item_index)
            caption = get_picture_caption(element)

            # ----------------------------
            # DEBUG VERSION: file.md
            # ----------------------------

            if caption:
                debug_lines.append("Image caption:")
                debug_lines.append(caption)
                debug_lines.append("")

            if crop_path is None:
                description = "[Picture/figure detected, but no crop file was available]"
                ocr_text = "[no meaningful text found]"

                debug_lines.append("Image description:")
                debug_lines.append(description)
                debug_lines.append("")
                debug_lines.append("Image OCR text:")
                debug_lines.append(ocr_text)
                debug_lines.append("")

                # LLM feed gets nothing unless there is a useful caption.
                clean_block = build_clean_image_llm_block(
                    caption=caption,
                    description=description,
                    ocr_text=ocr_text,
                )

                if clean_block:
                    llm_lines.append(clean_block)
                    llm_lines.append("")

                continue

            if skip_vlm:
                description = "[VLM skipped]"
            else:
                description = describe_picture_with_vlm(
                    image_path=crop_path,
                    vlm_model=vlm_model,
                    ollama_url=ollama_url,
                )

            if skip_ocr:
                ocr_text = "[OCR skipped]"
            else:
                ocr_text = run_image_ocr(
                    image_path=crop_path,
                    tesseract_cmd=tesseract_cmd,
                    ocr_lang=ocr_lang,
                )

            debug_image_block = build_image_block(
                description=description,
                ocr_text=ocr_text,
                crop_path=crop_path,
            )

            debug_lines.append(debug_image_block)
            debug_lines.append("")

            # ----------------------------
            # CLEAN LLM VERSION: file_llm_feed.md
            # ----------------------------

            clean_image_block = build_clean_image_llm_block(
                caption=caption,
                description=description,
                ocr_text=ocr_text,
            )

            if clean_image_block:
                llm_lines.append(clean_image_block)
                llm_lines.append("")

        else:
            text = get_element_text(element)

            if text:
                append_to_both(debug_lines, llm_lines, text)

    debug_md = normalize_text("\n".join(debug_lines)) + "\n"
    llm_feed_md = normalize_text("\n".join(llm_lines)) + "\n"

    return debug_md, llm_feed_md


# ----------------------------
# PROCESS ONE FILE
# ----------------------------

def process_one_file(
    source_file,
    phase1_output_folder,
    skip_vlm=False,
    vlm_model="qwen2.5vl:7b",
    ollama_url="http://localhost:11434",
    skip_ocr=False,
    tesseract_cmd=None,
    ocr_lang="eng",
    skip_table_llm=False,
    table_llm_model="qwen3:14b",
):
    doc_folder_name = clean_name(source_file.name)
    doc_output_folder = safe_mkdir(phase1_output_folder / doc_folder_name)

    crops_root = safe_mkdir(doc_output_folder / "docling_crops")

    markdown_path = doc_output_folder / "file.md"
    llm_feed_markdown_path = doc_output_folder / "file_llm_feed.md"
    outlined_pdf_path = doc_output_folder / "outlined_file.pdf"

    print(f"  Running Docling...")

    conv_res = convert_with_docling(source_file)
    doc = conv_res.document

    print(f"  Saving crops...")

    picture_crops, table_crops = save_docling_crops(
        doc=doc,
        crops_root=crops_root,
    )

    print(f"  Creating outlined PDF...")

    outlined_ok = create_outlined_pdf(
        doc=doc,
        output_pdf_path=outlined_pdf_path,
    )

    print(f"  Building file.md and file_llm_feed.md...")

    debug_md, llm_feed_md = render_docling_to_markdown_versions(
        doc=doc,
        source_file=source_file,
        picture_crops=picture_crops,
        skip_vlm=skip_vlm,
        vlm_model=vlm_model,
        ollama_url=ollama_url,
        skip_ocr=skip_ocr,
        tesseract_cmd=tesseract_cmd,
        ocr_lang=ocr_lang,
        skip_table_llm=skip_table_llm,
        table_llm_model=table_llm_model,
    )

    markdown_path.write_text(debug_md, encoding="utf-8")
    llm_feed_markdown_path.write_text(llm_feed_md, encoding="utf-8")

    return {
        "file_name": source_file.name,
        "status": "ok",
        "output_folder": str(doc_output_folder),
        "markdown_file": str(markdown_path),
        "llm_feed_markdown_file": str(llm_feed_markdown_path),
        "outlined_file": str(outlined_pdf_path) if outlined_ok else None,
        "picture_crops": len(picture_crops),
        "table_crops": len(table_crops),
    }


# ----------------------------
# MAIN PHASE 1 LOGIC
# ----------------------------

def run_phase1(
    skip_vlm=False,
    vlm_model="qwen2.5vl:7b",
    ollama_url="http://localhost:11434",
    skip_ocr=False,
    tesseract_cmd=None,
    ocr_lang="eng",
    skip_table_llm=False,
    table_llm_model="qwen3:14b",
    limit=None,
):
    """
    Run Phase 1 using fixed project-relative folders.

    This script assumes you run it from the project folder.

    Input is always:
        ./Phase 00_ Snapshot/02_All_Files_Flat

    Output is always:
        ./Phase 01_TextExtraction

    Example:
        cd "C:\\Users\\amenx\\Desktop\\all phases re-engineered"
        python 01_phase1_text_extraction_docling_v2.py --limit 2
    """

    project_root = Path.cwd().resolve()

    phase0_flat_folder = project_root / PHASE0_FOLDER_NAME / FLAT_FILES_FOLDER_NAME
    phase1_output_folder = safe_mkdir(project_root / PHASE1_FOLDER_NAME)

    if not phase0_flat_folder.exists():
        raise FileNotFoundError(
            "Could not find the fixed Phase 0 flat folder:\n"
            f"{phase0_flat_folder}\n\n"
            "Make sure you are running this script from the project folder that contains:\n"
            "./Phase 00_ Snapshot/02_All_Files_Flat"
        )

    files = find_files(phase0_flat_folder)

    if limit is not None:
        files = files[:limit]

    print("=" * 80)
    print("PHASE 1 TEXT EXTRACTION START")
    print("=" * 80)
    print(f"Working folder:      {project_root}")
    print(f"Input folder:        {phase0_flat_folder}")
    print(f"Output folder:       {phase1_output_folder}")
    print(f"Files to process:    {len(files)}")
    print(f"Use VLM:             {not skip_vlm}")
    print(f"VLM model:           {vlm_model if not skip_vlm else '[skipped]'}")
    print(f"Use OCR:             {not skip_ocr}")
    print(f"Use table LLM:       {not skip_table_llm}")
    print(f"Table LLM model:     {table_llm_model if not skip_table_llm else '[skipped]'}")
    print("=" * 80)

    results = []

    for index, source_file in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {source_file.name}")

        try:
            result = process_one_file(
                source_file=source_file,
                phase1_output_folder=phase1_output_folder,
                skip_vlm=skip_vlm,
                vlm_model=vlm_model,
                ollama_url=ollama_url,
                skip_ocr=skip_ocr,
                tesseract_cmd=tesseract_cmd,
                ocr_lang=ocr_lang,
                skip_table_llm=skip_table_llm,
                table_llm_model=table_llm_model,
            )

            results.append(result)
            print("  Status: ok")

        except Exception as e:
            error_result = {
                "file_name": source_file.name,
                "status": "error",
                "error": repr(e),
            }

            results.append(error_result)
            print("  Status: error")
            print(f"  Reason: {e}")

    ok_count = sum(1 for item in results if item["status"] == "ok")
    error_count = len(results) - ok_count

    print("=" * 80)
    print("PHASE 1 TEXT EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Successful:          {ok_count}")
    print(f"Errors:              {error_count}")
    print(f"Output folder:       {phase1_output_folder}")
    print("=" * 80)

    return phase1_output_folder

# ----------------------------
# CLI
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1: Docling extraction + VLM/OCR image enrichment + clean LLM feed. "
            "Uses fixed project-relative paths."
        )
    )

    parser.add_argument(
        "--vlm-model",
        type=str,
        default="qwen2.5vl:7b",
        help="Ollama vision model to use for image/graph descriptions.",
    )

    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL.",
    )

    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip VLM image descriptions.",
    )

    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip OCR on pictures/graphs.",
    )

    parser.add_argument(
        "--tesseract-cmd",
        type=str,
        default=None,
        help="Optional full path to tesseract.exe.",
    )

    parser.add_argument(
        "--ocr-lang",
        type=str,
        default="eng",
        help="OCR language code.",
    )

    parser.add_argument(
        "--skip-table-llm",
        action="store_true",
        help="Skip LLM cleanup for Docling-generated tables.",
    )

    parser.add_argument(
        "--table-llm-model",
        type=str,
        default="qwen3:14b",
        help="Ollama text model to use for table cleanup.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of files to process for testing.",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    run_phase1(
        skip_vlm=args.skip_vlm,
        vlm_model=args.vlm_model,
        ollama_url=args.ollama_url,
        skip_ocr=args.skip_ocr,
        tesseract_cmd=args.tesseract_cmd,
        ocr_lang=args.ocr_lang,
        skip_table_llm=args.skip_table_llm,
        table_llm_model=args.table_llm_model,
        limit=args.limit,
    )

if __name__ == "__main__":
    main()
