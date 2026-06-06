from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import requests

from .utils import write_json


DEFAULT_VLM_PROMPT = """
You are reviewing a cropped region from a document.

Describe only useful information visible in the crop.

Return a concise structured summary with:
- crop_type
- visible_text
- useful_information
- likely_document_role
- should_keep_for_later_extraction: true/false

If the crop is decorative or not useful, say so clearly.
"""


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def run_ollama_vlm_on_image(image_path: Path, model: str, ollama_url: str, prompt: str = DEFAULT_VLM_PROMPT, timeout: int = 180) -> str:
    url = f"{ollama_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_to_base64(image_path)]}],
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return str(data.get("message", {}).get("content", "")).strip()


def run_vlm_for_selected_crops(selected_manifest: list[dict[str, Any]], output_dir: Path, model: str, ollama_url: str) -> list[dict[str, Any]]:
    results = []

    for item in selected_manifest:
        crop_path_value = item.get("crop_path")
        if not crop_path_value:
            continue

        crop_path = Path(crop_path_value)
        result = {
            "crop_id": item.get("crop_id"),
            "doc_id": item.get("doc_id"),
            "file_name": item.get("file_name"),
            "source_box_id": item.get("source_box_id"),
            "label": item.get("label"),
            "group": item.get("group"),
            "page_no": item.get("page_no"),
            "crop_path": str(crop_path),
            "model": model,
            "status": "pending",
            "summary": "",
            "error": None,
        }

        try:
            result["summary"] = run_ollama_vlm_on_image(crop_path, model=model, ollama_url=ollama_url)
            result["status"] = "complete"
        except Exception as error:
            result["status"] = "error"
            result["error"] = str(error)

        results.append(result)

    write_json(output_dir / "vlm_results.json", results)
    return results
