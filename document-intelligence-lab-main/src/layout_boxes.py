from __future__ import annotations

from typing import Any

import fitz


def get_docling_items(doc_dict: dict[str, Any]) -> list[dict[str, Any]]:
    item_groups = ["texts", "tables", "pictures", "groups", "key_value_items", "form_items"]
    items: list[dict[str, Any]] = []

    for group_name in item_groups:
        group = doc_dict.get(group_name)
        if not isinstance(group, list):
            continue

        for idx, item in enumerate(group):
            if not isinstance(item, dict):
                continue
            copied = dict(item)
            copied["_group"] = group_name
            copied["_index"] = idx
            items.append(copied)

    return items


def get_item_label(item: dict[str, Any]) -> str:
    label = item.get("label")
    group = item.get("_group", "unknown")

    if isinstance(label, str) and label.strip():
        return label
    if group == "texts":
        return "text"
    if group == "tables":
        return "table"
    if group == "pictures":
        return "picture"
    return str(group)


def normalize_bbox_dict(raw_bbox: Any) -> dict[str, float] | None:
    if not isinstance(raw_bbox, dict):
        return None

    left = raw_bbox.get("l", raw_bbox.get("left", raw_bbox.get("x0")))
    top = raw_bbox.get("t", raw_bbox.get("top", raw_bbox.get("y0")))
    right = raw_bbox.get("r", raw_bbox.get("right", raw_bbox.get("x1")))
    bottom = raw_bbox.get("b", raw_bbox.get("bottom", raw_bbox.get("y1")))

    if left is None or top is None or right is None or bottom is None:
        return None

    try:
        return {
            "l": float(left),
            "t": float(top),
            "r": float(right),
            "b": float(bottom),
            "coord_origin": str(raw_bbox.get("coord_origin", "BOTTOMLEFT")).upper(),
        }
    except Exception:
        return None


def extract_layout_boxes(doc_dict: dict[str, Any], doc_id: str = "", file_name: str = "") -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []

    for item in get_docling_items(doc_dict):
        label = get_item_label(item)
        prov = item.get("prov", [])

        if not isinstance(prov, list):
            continue

        for prov_item in prov:
            if not isinstance(prov_item, dict):
                continue

            page_no = prov_item.get("page_no", prov_item.get("page"))
            bbox = normalize_bbox_dict(prov_item.get("bbox"))

            if page_no is None or bbox is None:
                continue

            box_number = len(boxes) + 1
            text = item.get("text") or item.get("orig") or item.get("name") or ""

            boxes.append(
                {
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "box_id": f"{doc_id}_box_{box_number:05d}" if doc_id else f"box_{box_number:05d}",
                    "label": label,
                    "group": item.get("_group", "unknown"),
                    "page_no": int(page_no),
                    "bbox": bbox,
                    "text_preview": str(text)[:250],
                    "source": "docling",
                }
            )

    return boxes


def bbox_to_pymupdf_rect(bbox: dict[str, float], page_height: float) -> fitz.Rect:
    left = bbox["l"]
    right = bbox["r"]
    top = bbox["t"]
    bottom = bbox["b"]
    origin = bbox.get("coord_origin", "BOTTOMLEFT").upper()

    if origin == "BOTTOMLEFT":
        y0 = page_height - top
        y1 = page_height - bottom
    else:
        y0 = top
        y1 = bottom

    x0, x1 = sorted([left, right])
    y0, y1 = sorted([y0, y1])

    return fitz.Rect(x0, y0, x1, y1)


def label_color(label: str, group: str = "") -> tuple[float, float, float]:
    label_lower = label.lower()
    group_lower = group.lower()

    if "table" in label_lower or group_lower == "tables":
        return (0.95, 0.35, 0.05)
    if "picture" in label_lower or "image" in label_lower or "chart" in label_lower or group_lower == "pictures":
        return (0.10, 0.45, 0.95)
    if "title" in label_lower or "section" in label_lower:
        return (0.45, 0.10, 0.75)
    if "caption" in label_lower:
        return (0.10, 0.60, 0.20)
    return (0.20, 0.20, 0.20)


def create_outlined_pdf(input_pdf_path, boxes: list[dict[str, Any]], output_pdf_path, line_width: float = 2.5) -> None:
    doc = fitz.open(str(input_pdf_path))

    for box in boxes:
        page_no = int(box["page_no"])
        page_index = page_no - 1

        if page_index < 0 or page_index >= len(doc):
            continue

        page = doc[page_index]
        rect = bbox_to_pymupdf_rect(box["bbox"], page.rect.height)
        label = str(box.get("label", "unknown"))
        group = str(box.get("group", ""))
        color = label_color(label, group)

        try:
            page.draw_rect(rect, color=color, width=line_width)
            page.insert_text(
                fitz.Point(rect.x0, max(10, rect.y0 - 6)),
                f"{group}:{label}"[:45],
                fontsize=16,
                fontname="hebo",
                color=color,
            )
        except Exception:
            continue

    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_pdf_path))
    doc.close()
