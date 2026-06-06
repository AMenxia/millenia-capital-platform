from __future__ import annotations

from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter

from .config import CROP_ZOOM, OUTLINE_BOX_WIDTH
from .cropper import crop_pdf_regions
from .layout_boxes import create_outlined_pdf, extract_layout_boxes
from .utils import ensure_dir, write_json, write_text


def docling_to_dict(docling_document: Any) -> dict[str, Any]:
    if hasattr(docling_document, "export_to_dict"):
        return docling_document.export_to_dict()
    if hasattr(docling_document, "model_dump"):
        return docling_document.model_dump(mode="json")
    if hasattr(docling_document, "dict"):
        return docling_document.dict()
    raise TypeError("Could not convert Docling document to dictionary.")


def docling_to_markdown(docling_document: Any) -> str:
    if hasattr(docling_document, "export_to_markdown"):
        return docling_document.export_to_markdown()
    if hasattr(docling_document, "export_to_text"):
        return docling_document.export_to_text()
    return str(docling_document)


def run_docling_for_file(input_path: Path, output_dir: Path, doc_id: str, original_file_name: str) -> dict[str, Any]:
    ensure_dir(output_dir)

    converter = DocumentConverter()
    result = converter.convert(str(input_path))
    doc = result.document

    markdown = docling_to_markdown(doc)
    doc_dict = docling_to_dict(doc)
    boxes = extract_layout_boxes(doc_dict, doc_id=doc_id, file_name=original_file_name)

    markdown_path = output_dir / "file.md"
    json_path = output_dir / "file.json"
    boxes_path = output_dir / "layout_boxes.json"
    outlined_pdf_path = output_dir / "outlined_file.pdf"

    write_text(markdown_path, markdown)
    write_json(json_path, doc_dict)
    write_json(boxes_path, boxes)

    crop_manifest = []

    if input_path.suffix.lower() == ".pdf":
        create_outlined_pdf(input_path, boxes, outlined_pdf_path, line_width=OUTLINE_BOX_WIDTH)
        crop_manifest = crop_pdf_regions(input_path, boxes, output_dir, zoom=CROP_ZOOM)
    else:
        outlined_pdf_path = None

    return {
        "doc_id": doc_id,
        "file_name": original_file_name,
        "input_path": str(input_path),
        "markdown_path": str(markdown_path),
        "json_path": str(json_path),
        "boxes_path": str(boxes_path),
        "outlined_pdf_path": str(outlined_pdf_path) if outlined_pdf_path else None,
        "crop_manifest_path": str(output_dir / "crop_manifest.json"),
        "selected_crop_manifest_path": str(output_dir / "selected_crop_manifest.json"),
        "box_count": len(boxes),
        "crop_count": len(crop_manifest),
        "output_dir": str(output_dir),
    }


def combine_run_outputs(run_output_dir: Path, doc_results: list[dict[str, Any]]) -> dict[str, Any]:
    all_crops = []
    selected_crops = []
    all_boxes = []

    for result in doc_results:
        doc_output_dir = Path(result["output_dir"])

        crop_manifest = []
        selected_manifest = []
        boxes = []

        crop_path = doc_output_dir / "crop_manifest.json"
        selected_path = doc_output_dir / "selected_crop_manifest.json"
        boxes_path = doc_output_dir / "layout_boxes.json"

        if crop_path.exists():
            import json
            crop_manifest = json.loads(crop_path.read_text(encoding="utf-8"))
        if selected_path.exists():
            import json
            selected_manifest = json.loads(selected_path.read_text(encoding="utf-8"))
        if boxes_path.exists():
            import json
            boxes = json.loads(boxes_path.read_text(encoding="utf-8"))

        all_crops.extend(crop_manifest)
        selected_crops.extend(selected_manifest)
        all_boxes.extend(boxes)

    write_json(run_output_dir / "combined_doc_results.json", doc_results)
    write_json(run_output_dir / "layout_boxes.json", all_boxes)
    write_json(run_output_dir / "crop_manifest.json", all_crops)
    write_json(run_output_dir / "selected_crop_manifest.json", selected_crops)

    return {
        "combined_doc_results_path": str(run_output_dir / "combined_doc_results.json"),
        "combined_boxes_path": str(run_output_dir / "layout_boxes.json"),
        "combined_crop_manifest_path": str(run_output_dir / "crop_manifest.json"),
        "combined_selected_crop_manifest_path": str(run_output_dir / "selected_crop_manifest.json"),
        "total_boxes": len(all_boxes),
        "total_crops": len(all_crops),
        "selected_crops": len(selected_crops),
    }
