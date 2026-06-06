from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import requests

from .utils import write_json


TABLE_CLEANUP_SYSTEM_PROMPT = """
You clean messy table extractions from document parser output.

Rules:
- Use the cropped table image and any provided text preview.
- Do not invent values.
- If a cell is unreadable, use null.
- Preserve row/column meaning as best as possible.
- Output valid JSON only. No markdown fences.
"""


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def extract_json_from_text(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    if not raw:
        return None

    try:
        return json.loads(raw)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except Exception:
            pass

    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(raw[first:last + 1])
        except Exception:
            pass

    return None


def is_table_crop(item: dict[str, Any]) -> bool:
    group = str(item.get("group", "")).lower()
    label = str(item.get("label", "")).lower()

    if group == "tables":
        return True

    return "table" in label


def selected_table_crops(selected_manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item for item in selected_manifest
        if is_table_crop(item)
        and item.get("crop_path")
        and item.get("status") in {"keep", "needs_review"}
    ]


def build_table_cleanup_prompt(crop_item: dict[str, Any]) -> str:
    crop_path = Path(str(crop_item.get("crop_path", "")))

    metadata = {
        "table_id": crop_item.get("crop_id"),
        "doc_id": crop_item.get("doc_id"),
        "file_name": crop_item.get("file_name"),
        "page_no": crop_item.get("page_no"),
        "crop_id": crop_item.get("crop_id"),
        "label": crop_item.get("label"),
        "group": crop_item.get("group"),
        "crop_path": str(crop_path),
    }

    expected_shape = {
        "table_id": "...",
        "doc_id": "...",
        "file_name": "...",
        "page_no": 1,
        "crop_id": "...",
        "columns": ["column 1", "column 2"],
        "rows": [
            {
                "column 1": "value",
                "column 2": "value",
            }
        ],
        "cleaned_markdown": "| column 1 | column 2 |\\n|---|---|\\n| value | value |",
        "issues": ["any uncertainty or cleanup note"],
        "source": "table_crop",
    }

    return f"""
You are given a cropped table from a document.

Return a JSON object with exactly this general structure:

{json.dumps(expected_shape, ensure_ascii=False, indent=2)}

Table metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

Text preview from parser:
{str(crop_item.get("text_preview", ""))}

Important:
- If the crop is not actually a table, return rows as [] and explain in issues.
- If the table is visually unclear, still return the best cleaned structure and list issues.
- Output valid JSON only.
""".strip()


def run_ollama_table_cleanup_on_crop(
    crop_item: dict[str, Any],
    model: str,
    ollama_url: str,
    timeout: int = 240,
) -> dict[str, Any]:
    crop_path = Path(str(crop_item.get("crop_path", "")))

    prompt = build_table_cleanup_prompt(crop_item)

    url = f"{ollama_url.rstrip('/')}/api/chat"

    message: dict[str, Any] = {
        "role": "user",
        "content": prompt,
    }

    if crop_path.exists():
        message["images"] = [image_to_base64(crop_path)]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TABLE_CLEANUP_SYSTEM_PROMPT},
            message,
        ],
        "stream": False,
    }

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    raw_text = str(data.get("message", {}).get("content", "")).strip()
    parsed = extract_json_from_text(raw_text)

    if parsed is None:
        parsed = {
            "table_id": crop_item.get("crop_id"),
            "doc_id": crop_item.get("doc_id"),
            "file_name": crop_item.get("file_name"),
            "page_no": crop_item.get("page_no"),
            "crop_id": crop_item.get("crop_id"),
            "columns": [],
            "rows": [],
            "cleaned_markdown": "",
            "issues": ["Model did not return valid JSON."],
            "source": "table_crop",
            "raw_model_response": raw_text,
        }

    parsed.setdefault("table_id", crop_item.get("crop_id"))
    parsed.setdefault("doc_id", crop_item.get("doc_id"))
    parsed.setdefault("file_name", crop_item.get("file_name"))
    parsed.setdefault("page_no", crop_item.get("page_no"))
    parsed.setdefault("crop_id", crop_item.get("crop_id"))
    parsed.setdefault("crop_path", str(crop_path))
    parsed.setdefault("columns", [])
    parsed.setdefault("rows", [])
    parsed.setdefault("cleaned_markdown", "")
    parsed.setdefault("issues", [])
    parsed.setdefault("source", "table_crop")

    return parsed


def run_table_cleanup_for_selected_crops(
    selected_manifest: list[dict[str, Any]],
    output_dir: Path,
    model: str,
    ollama_url: str,
) -> list[dict[str, Any]]:
    table_crops = selected_table_crops(selected_manifest)
    results: list[dict[str, Any]] = []

    for crop_item in table_crops:
        result = {
            "table_id": crop_item.get("crop_id"),
            "doc_id": crop_item.get("doc_id"),
            "file_name": crop_item.get("file_name"),
            "page_no": crop_item.get("page_no"),
            "crop_id": crop_item.get("crop_id"),
            "crop_path": crop_item.get("crop_path"),
            "model": model,
            "status": "pending",
            "columns": [],
            "rows": [],
            "cleaned_markdown": "",
            "issues": [],
            "error": None,
        }

        try:
            cleaned = run_ollama_table_cleanup_on_crop(
                crop_item=crop_item,
                model=model,
                ollama_url=ollama_url,
            )
            result.update(cleaned)
            result["model"] = model
            result["status"] = "complete"
            result["error"] = None
        except Exception as error:
            result["status"] = "error"
            result["error"] = str(error)
            result["issues"] = [str(error)]

        results.append(result)

    write_json(output_dir / "cleaned_tables.json", results)
    return results
