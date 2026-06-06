from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import read_json


def _df(data: Any) -> pd.DataFrame:
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame()


def create_excel_export(run_dir: Path) -> Path:
    out_path = run_dir / "document_intelligence_export.xlsx"

    documents = read_json(run_dir / "run_manifest.json", default={}).get("documents", [])
    layout_items = read_json(run_dir / "layout_items.json", default=[])
    raw_tables = read_json(run_dir / "raw_tables.json", default=[])
    cleaned_tables = read_json(run_dir / "cleaned_tables.json", default=[])
    visual_items = read_json(run_dir / "visual_items.json", default=[])
    image_summaries = read_json(run_dir / "image_summaries.json", default=[])
    extraction = read_json(run_dir / "consolidated_dossier.json", default=None) or read_json(run_dir / "structured_extraction.json", default={})

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        _df({"run_dir": str(run_dir), "document_count": len(documents)}).to_excel(writer, sheet_name="overview", index=False)
        _df(documents).to_excel(writer, sheet_name="documents", index=False)
        _df(layout_items).to_excel(writer, sheet_name="layout_items", index=False)
        _df(raw_tables).drop(columns=["raw_html", "raw_markdown"], errors="ignore").to_excel(writer, sheet_name="raw_tables", index=False)

        clean_summary = []
        clean_rows = []
        for table in cleaned_tables:
            clean_summary.append({
                "table_id": table.get("table_id"),
                "file_name": table.get("file_name"),
                "page_no": table.get("page_no"),
                "caption": table.get("caption"),
                "row_count": len(table.get("rows", []) or []),
                "columns": ", ".join(table.get("columns", []) or []),
                "issues": "; ".join(map(str, table.get("issues", []) or [])),
            })
            for idx, row in enumerate(table.get("rows", []) or [], start=1):
                clean_rows.append({"table_id": table.get("table_id"), "row_number": idx, **row})
        _df(clean_summary).to_excel(writer, sheet_name="cleaned_tables", index=False)
        _df(clean_rows).to_excel(writer, sheet_name="cleaned_table_rows", index=False)

        _df(visual_items).to_excel(writer, sheet_name="visual_items", index=False)
        _df(image_summaries).to_excel(writer, sheet_name="image_summaries", index=False)
        _df(extraction.get("field_values", []) if isinstance(extraction, dict) else []).to_excel(writer, sheet_name="field_values", index=False)
        _df(extraction.get("people_or_entities", []) if isinstance(extraction, dict) else []).to_excel(writer, sheet_name="people_or_entities", index=False)
        notes = extraction.get("conflicts_or_notes", []) if isinstance(extraction, dict) else []
        _df([{"note": n} for n in notes]).to_excel(writer, sheet_name="conflicts_or_notes", index=False)
    return out_path
