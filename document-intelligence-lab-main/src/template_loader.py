from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_templates(templates_dir: Path) -> dict[str, dict[str, Any]]:
    templates: dict[str, dict[str, Any]] = {}
    for path in sorted(templates_dir.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            name = obj.get("template_name") or path.stem
            templates[name] = obj
        except Exception:
            continue
    return templates


def blank_structured_extraction(template: dict[str, Any], source_files: list[str] | None = None) -> dict[str, Any]:
    source_files = source_files or []

    fields = {}
    for field in template.get("fields", []):
        name = field.get("name")
        if not name:
            continue
        fields[name] = {
            "value": None,
            "answer": "",
            "evidence": [],
        }

    return {
        "template_name": template.get("template_name"),
        "document_type": template.get("document_type", ""),
        "source_files": source_files,
        "short_summary": "",
        "long_summary": "",
        "final_fields": fields,
        "key_people": [],
        "documents": [
            {
                "doc_id": f"doc_{idx:03d}",
                "file_name": file_name,
                "document_category": template.get("document_type", ""),
                "short_summary": "",
            }
            for idx, file_name in enumerate(source_files, start=1)
        ],
        "visual_evidence": [],
        "warnings": [],
    }


def template_fields_dataframe(template: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for field in template.get("fields", []):
        rows.append(
            {
                "field_name": field.get("name", ""),
                "field_label": field.get("label", ""),
                "required": field.get("required", False),
                "description": field.get("description", ""),
            }
        )
    return pd.DataFrame(rows)


def extraction_to_preview_tables(extraction: dict[str, Any]) -> dict[str, pd.DataFrame]:
    overview = pd.DataFrame(
        [
            {
                "template_name": extraction.get("template_name"),
                "document_type": extraction.get("document_type"),
                "source_files": ", ".join(extraction.get("source_files", [])),
                "short_summary": extraction.get("short_summary", ""),
                "long_summary": extraction.get("long_summary", ""),
            }
        ]
    )

    documents = pd.DataFrame(extraction.get("documents", []))

    field_rows = []
    final_fields = extraction.get("final_fields", {})
    if isinstance(final_fields, dict):
        for field_name, field_obj in final_fields.items():
            if isinstance(field_obj, dict):
                field_rows.append(
                    {
                        "field_name": field_name,
                        "value": field_obj.get("value"),
                        "answer": field_obj.get("answer"),
                        "evidence": json.dumps(field_obj.get("evidence", []), ensure_ascii=False),
                    }
                )
            else:
                field_rows.append(
                    {
                        "field_name": field_name,
                        "value": field_obj,
                        "answer": "",
                        "evidence": "[]",
                    }
                )

    return {
        "overview": overview,
        "documents": documents,
        "field_values": pd.DataFrame(field_rows),
        "people_or_entities": pd.DataFrame(extraction.get("key_people", [])),
        "visual_evidence": pd.DataFrame(extraction.get("visual_evidence", [])),
    }
