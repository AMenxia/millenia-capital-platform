from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import json_safe, safe_stem, write_json


def evidence_to_cell(value: Any) -> str:
    if value in (None, ""):
        return "[]"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def cleaned_tables_dataframe(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "cleaned_tables.json"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "table_id",
                "doc_id",
                "file_name",
                "page_no",
                "crop_id",
                "status",
                "columns",
                "rows",
                "cleaned_markdown",
                "issues",
                "error",
                "crop_path",
            ]
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = []

    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue

        rows.append(
            {
                "table_id": item.get("table_id", ""),
                "doc_id": item.get("doc_id", ""),
                "file_name": item.get("file_name", ""),
                "page_no": item.get("page_no", ""),
                "crop_id": item.get("crop_id", ""),
                "status": item.get("status", ""),
                "columns": evidence_to_cell(item.get("columns", [])),
                "rows": evidence_to_cell(item.get("rows", [])),
                "cleaned_markdown": item.get("cleaned_markdown", ""),
                "issues": evidence_to_cell(item.get("issues", [])),
                "error": item.get("error", ""),
                "crop_path": item.get("crop_path", ""),
            }
        )

    return pd.DataFrame(rows)


def visual_evidence_dataframe(extraction: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    visual = extraction.get("visual_evidence", [])
    if not isinstance(visual, list):
        visual = []

    if not visual:
        vlm_path = output_dir / "vlm_results.json"
        if vlm_path.exists():
            try:
                visual = json.loads(vlm_path.read_text(encoding="utf-8"))
            except Exception:
                visual = []

    visual_df = pd.DataFrame(visual)
    if visual_df.empty:
        visual_df = pd.DataFrame(columns=["crop_id", "doc_id", "file_name", "page_no", "label", "status", "crop_path", "summary"])

    return visual_df


def field_rows_from_extraction(extraction: dict[str, Any], doc_id: str = "", file_name: str = "") -> list[dict[str, Any]]:
    final_fields = extraction.get("final_fields", {})
    rows: list[dict[str, Any]] = []

    if isinstance(final_fields, dict):
        for field_name, field_obj in final_fields.items():
            if isinstance(field_obj, dict):
                rows.append(
                    {
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "template_name": extraction.get("template_name", ""),
                        "field_name": field_name,
                        "value": json_safe(field_obj.get("value")),
                        "answer": field_obj.get("answer", ""),
                        "evidence": evidence_to_cell(field_obj.get("evidence", [])),
                    }
                )
            else:
                rows.append(
                    {
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "template_name": extraction.get("template_name", ""),
                        "field_name": field_name,
                        "value": json_safe(field_obj),
                        "answer": "",
                        "evidence": "[]",
                    }
                )

    return rows


def people_rows_from_extraction(extraction: dict[str, Any], doc_id: str = "", file_name: str = "") -> list[dict[str, Any]]:
    people = extraction.get("key_people", [])
    if not isinstance(people, list):
        return []

    rows = []
    for item in people:
        if isinstance(item, dict):
            row = dict(item)
            row.setdefault("doc_id", doc_id)
            row.setdefault("file_name", file_name)
            rows.append(row)
    return rows


def conflicts_dataframe(extraction: dict[str, Any]) -> pd.DataFrame:
    conflicts = extraction.get("conflicts_or_notes", [])
    if not isinstance(conflicts, list):
        conflicts = []

    rows = []
    for item in conflicts:
        if isinstance(item, dict):
            rows.append(
                {
                    "field_name": item.get("field_name", ""),
                    "chosen_value": json_safe(item.get("chosen_value", "")),
                    "alternative_values": evidence_to_cell(item.get("alternative_values", [])),
                    "note": item.get("note", ""),
                    "evidence": evidence_to_cell(item.get("evidence", [])),
                }
            )
        else:
            rows.append({"field_name": "", "chosen_value": "", "alternative_values": "[]", "note": str(item), "evidence": "[]"})

    if not rows:
        rows = [{"field_name": "", "chosen_value": "", "alternative_values": "[]", "note": "", "evidence": "[]"}]

    return pd.DataFrame(rows)


def individual_mode_tables(extraction: dict[str, Any], output_dir: Path) -> dict[str, pd.DataFrame]:
    analyses = extraction.get("individual_document_analyses", [])
    if not isinstance(analyses, list):
        analyses = []

    document_rows = []
    field_rows = []
    people_rows = []

    for item in analyses:
        if not isinstance(item, dict):
            continue

        doc_id = item.get("doc_id", "")
        file_name = item.get("file_name", "")
        document_rows.append(
            {
                "doc_id": doc_id,
                "file_name": file_name,
                "document_category": item.get("document_type", ""),
                "short_summary": item.get("short_summary", ""),
                "long_summary": item.get("long_summary", ""),
            }
        )
        field_rows.extend(field_rows_from_extraction(item, doc_id=doc_id, file_name=file_name))
        people_rows.extend(people_rows_from_extraction(item, doc_id=doc_id, file_name=file_name))

    overview = pd.DataFrame(
        [
            {
                "analysis_mode": extraction.get("analysis_mode", "individual_document_analysis"),
                "template_name": extraction.get("template_name", ""),
                "document_type": extraction.get("document_type", ""),
                "source_files": ", ".join(extraction.get("source_files", [])),
                "short_summary": extraction.get("short_summary", ""),
                "long_summary": extraction.get("long_summary", ""),
                "document_count": len(analyses),
                "field_count": len(field_rows),
                "people_or_entities_count": len(people_rows),
                "cleaned_tables_count": len(cleaned_tables_dataframe(output_dir)),
                "warnings": evidence_to_cell(extraction.get("warnings", [])),
            }
        ]
    )

    documents_df = pd.DataFrame(document_rows)
    fields_df = pd.DataFrame(field_rows)
    people_df = pd.DataFrame(people_rows)
    visual_df = visual_evidence_dataframe(extraction, output_dir)
    cleaned_tables_df = cleaned_tables_dataframe(output_dir)

    if documents_df.empty:
        documents_df = pd.DataFrame(columns=["doc_id", "file_name", "document_category", "short_summary", "long_summary"])
    if fields_df.empty:
        fields_df = pd.DataFrame(columns=["doc_id", "file_name", "template_name", "field_name", "value", "answer", "evidence"])
    if people_df.empty:
        people_df = pd.DataFrame(columns=["doc_id", "file_name", "full_name", "role", "organization", "contact_information", "evidence"])

    return {
        "overview": overview,
        "documents": documents_df,
        "field_values": fields_df,
        "people_or_entities": people_df,
        "visual_evidence": visual_df,
        "cleaned_tables": cleaned_tables_df,
    }


def extraction_to_excel_tables(extraction: dict[str, Any], output_dir: Path) -> dict[str, pd.DataFrame]:
    if extraction.get("analysis_mode") == "individual_document_analysis" or "individual_document_analyses" in extraction and extraction.get("analysis_mode") != "consolidated_dossier":
        return individual_mode_tables(extraction, output_dir)

    source_files = extraction.get("source_files", [])
    if not isinstance(source_files, list):
        source_files = [str(source_files)]

    field_rows = field_rows_from_extraction(extraction)
    documents = extraction.get("documents", [])
    if not isinstance(documents, list):
        documents = []

    people = extraction.get("key_people", [])
    if not isinstance(people, list):
        people = []

    visual_df = visual_evidence_dataframe(extraction, output_dir)
    cleaned_tables_df = cleaned_tables_dataframe(output_dir)

    overview = pd.DataFrame(
        [
            {
                "analysis_mode": extraction.get("analysis_mode", "consolidated_dossier"),
                "template_name": extraction.get("template_name", ""),
                "document_type": extraction.get("document_type", ""),
                "source_files": ", ".join(source_files),
                "short_summary": extraction.get("short_summary", ""),
                "long_summary": extraction.get("long_summary", ""),
                "field_count": len(field_rows),
                "people_or_entities_count": len(people),
                "visual_evidence_count": len(visual_df),
                "cleaned_tables_count": len(cleaned_tables_df),
                "warnings": evidence_to_cell(extraction.get("warnings", [])),
            }
        ]
    )

    documents_df = pd.DataFrame(documents)
    fields_df = pd.DataFrame(field_rows)
    people_df = pd.DataFrame(people)
    conflicts_df = conflicts_dataframe(extraction)

    if documents_df.empty:
        documents_df = pd.DataFrame(columns=["doc_id", "file_name", "document_category", "short_summary"])
    if fields_df.empty:
        fields_df = pd.DataFrame(columns=["doc_id", "file_name", "template_name", "field_name", "value", "answer", "evidence"])
    if people_df.empty:
        people_df = pd.DataFrame(columns=["full_name", "role", "organization", "contact_information", "evidence"])

    tables = {
        "overview": overview,
        "documents": documents_df,
        "field_values": fields_df,
        "people_or_entities": people_df,
        "visual_evidence": visual_df,
        "cleaned_tables": cleaned_tables_df,
    }

    if extraction.get("analysis_mode") == "consolidated_dossier":
        tables["conflicts_or_notes"] = conflicts_df

    return tables


def format_worksheet(workbook, worksheet, df: pd.DataFrame) -> None:
    header_format = workbook.add_format(
        {
            "bold": True,
            "bg_color": "#E8EEF7",
            "border": 1,
            "text_wrap": True,
            "valign": "top",
        }
    )
    body_format = workbook.add_format({"text_wrap": True, "valign": "top"})

    worksheet.freeze_panes(1, 0)

    if len(df.columns) > 0:
        worksheet.autofilter(0, 0, max(len(df), 1), len(df.columns) - 1)

    for col_index, col_name in enumerate(df.columns):
        worksheet.write(0, col_index, col_name, header_format)

        if len(df) > 0:
            max_cell_len = df[col_name].astype(str).map(len).max()
        else:
            max_cell_len = 0

        max_len = max(len(str(col_name)), int(max_cell_len))
        width = min(max(max_len + 2, 12), 70)
        worksheet.set_column(col_index, col_index, width, body_format)


def insert_visual_images(worksheet, visual_df: pd.DataFrame, image_col_name: str = "crop_path") -> None:
    if image_col_name not in visual_df.columns:
        return

    image_preview_col = len(visual_df.columns)
    worksheet.write(0, image_preview_col, "image_preview")

    for row_idx, path_value in enumerate(visual_df[image_col_name].tolist(), start=1):
        if not path_value:
            continue

        path = Path(str(path_value))
        if not path.exists():
            continue

        try:
            worksheet.set_row(row_idx, 90)
            worksheet.insert_image(
                row_idx,
                image_preview_col,
                str(path),
                {"x_scale": 0.35, "y_scale": 0.35, "object_position": 1},
            )
        except Exception:
            continue

    worksheet.set_column(image_preview_col, image_preview_col, 22)


def excel_filename_from_sources(source_files: list[str], template_name: str, analysis_mode: str = "") -> str:
    mode_suffix = "dossier" if analysis_mode == "consolidated_dossier" else "analysis"

    if not source_files:
        base = template_name or "document"
    elif len(source_files) == 1:
        base = safe_stem(source_files[0])
    else:
        base = f"{safe_stem(source_files[0])}_multi"

    return f"{base}_{mode_suffix}.xlsx"


def build_excel_export(output_dir: Path, extraction: dict[str, Any]) -> tuple[Path, dict[str, pd.DataFrame]]:
    tables = extraction_to_excel_tables(extraction, output_dir)

    source_files = extraction.get("source_files", [])
    if not isinstance(source_files, list):
        source_files = [str(source_files)]

    filename = excel_filename_from_sources(
        source_files,
        extraction.get("template_name", "document"),
        extraction.get("analysis_mode", ""),
    )
    output_path = output_dir / filename

    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for sheet_name, df in tables.items():
            safe_df = df.copy()
            for col in safe_df.columns:
                safe_df[col] = safe_df[col].map(json_safe)

            safe_df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            workbook = writer.book
            format_worksheet(workbook, worksheet, safe_df)

            if sheet_name in {"visual_evidence", "cleaned_tables"}:
                insert_visual_images(worksheet, safe_df)

    output_path.write_bytes(buffer.getvalue())

    preview_json = {
        name: df.astype(str).to_dict(orient="records")
        for name, df in tables.items()
    }
    write_json(output_dir / "excel_preview_tables.json", preview_json)

    return output_path, tables
