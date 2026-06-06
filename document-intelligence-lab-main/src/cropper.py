from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz

from .layout_boxes import bbox_to_pymupdf_rect
from .utils import ensure_dir, write_json


USEFUL_GROUPS = {"tables", "pictures"}
USEFUL_LABEL_KEYWORDS = ["table", "picture", "image", "figure", "chart", "diagram"]


def is_crop_candidate(box: dict[str, Any]) -> bool:
    group = str(box.get("group", "")).lower()
    label = str(box.get("label", "")).lower()

    if group in USEFUL_GROUPS:
        return True

    return any(keyword in label for keyword in USEFUL_LABEL_KEYWORDS)


def safe_label(label: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9._-]+", "_", label.strip().lower())
    return label or "region"


def crop_pdf_regions(input_pdf_path: Path, boxes: list[dict[str, Any]], output_dir: Path, zoom: float = 2.0) -> list[dict[str, Any]]:
    crops_dir = ensure_dir(output_dir / "crops")
    manifest: list[dict[str, Any]] = []

    doc = fitz.open(str(input_pdf_path))
    matrix = fitz.Matrix(zoom, zoom)

    candidates = [box for box in boxes if is_crop_candidate(box)]

    for idx, box in enumerate(candidates, start=1):
        page_no = int(box["page_no"])
        page_index = page_no - 1

        if page_index < 0 or page_index >= len(doc):
            continue

        page = doc[page_index]
        rect = bbox_to_pymupdf_rect(box["bbox"], page.rect.height)

        if rect.width < 5 or rect.height < 5:
            continue

        label = safe_label(str(box.get("label", "region")))
        crop_name = f"{box.get('doc_id', 'doc')}_crop_{idx:05d}_{label}_p{page_no}.png"
        crop_path = crops_dir / crop_name

        item = {
            "crop_id": f"{box.get('doc_id', 'doc')}_crop_{idx:05d}",
            "doc_id": box.get("doc_id", ""),
            "file_name": box.get("file_name", ""),
            "source_box_id": box.get("box_id"),
            "label": box.get("label"),
            "group": box.get("group"),
            "page_no": page_no,
            "crop_path": str(crop_path),
            "status": "needs_review",
            "error": None,
            "text_preview": box.get("text_preview", ""),
        }

        try:
            pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
            pix.save(str(crop_path))
        except Exception as error:
            item["crop_path"] = None
            item["status"] = "error"
            item["error"] = str(error)

        manifest.append(item)

    doc.close()

    write_json(output_dir / "crop_manifest.json", manifest)
    write_selected_manifest(output_dir, manifest)
    return manifest


def write_selected_manifest(output_dir: Path, manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        item for item in manifest
        if item.get("status") in {"keep", "needs_review"} and item.get("crop_path")
    ]
    write_json(output_dir / "selected_crop_manifest.json", selected)
    return selected


def update_crop_status(output_dir: Path, manifest: list[dict[str, Any]], crop_id: str, status: str) -> list[dict[str, Any]]:
    for item in manifest:
        if item.get("crop_id") == crop_id:
            item["status"] = status

    write_json(output_dir / "crop_manifest.json", manifest)
    write_selected_manifest(output_dir, manifest)
    return manifest
