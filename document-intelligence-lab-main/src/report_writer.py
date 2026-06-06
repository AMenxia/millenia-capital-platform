from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import read_json, write_text, json_safe


def field_lines(final_fields: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    if not isinstance(final_fields, dict):
        return lines

    for field_name, field_obj in final_fields.items():
        if isinstance(field_obj, dict):
            lines.extend(
                [
                    f"### {field_name}",
                    "",
                    f"Value: {json_safe(field_obj.get('value'))}",
                    "",
                    f"Answer: {field_obj.get('answer', '')}",
                    "",
                    f"Evidence: {json_safe(field_obj.get('evidence', []))}",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"### {field_name}",
                    "",
                    f"Value: {json_safe(field_obj)}",
                    "",
                ]
            )

    return lines


def build_individual_report(extraction: dict[str, Any]) -> str:
    lines = [
        "# Individual Document Analysis Report",
        "",
        f"Template: `{extraction.get('template_name', '')}`",
        f"Document type: `{extraction.get('document_type', '')}`",
        "",
        "## Overview",
        "",
        extraction.get("short_summary", "") or "Individual document analysis was generated.",
        "",
    ]

    analyses = extraction.get("individual_document_analyses", [])
    if isinstance(analyses, list):
        for item in analyses:
            if not isinstance(item, dict):
                continue

            lines.extend(
                [
                    "---",
                    "",
                    f"## {item.get('doc_id', '')} — {item.get('file_name', '')}",
                    "",
                    "### Summary",
                    "",
                    item.get("short_summary", "") or "No summary.",
                    "",
                    "### Field Values",
                    "",
                ]
            )
            lines.extend(field_lines(item.get("final_fields", {})))

    warnings = extraction.get("warnings", [])
    lines.extend(["## Warnings", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("No warnings.")

    return "\n".join(lines).strip() + "\n"


def build_dossier_report(extraction: dict[str, Any]) -> str:
    lines = [
        "# Consolidated Dossier",
        "",
        f"Template: `{extraction.get('template_name', '')}`",
        f"Document type: `{extraction.get('document_type', '')}`",
        "",
        "## Short Summary",
        "",
        extraction.get("short_summary", "") or "No short summary.",
        "",
        "## Long Summary",
        "",
        extraction.get("long_summary", "") or "No long summary.",
        "",
        "## Field Values",
        "",
    ]

    lines.extend(field_lines(extraction.get("final_fields", {})))

    conflicts = extraction.get("conflicts_or_notes", [])
    lines.extend(["## Conflicts or Notes", ""])
    if isinstance(conflicts, list) and conflicts:
        for item in conflicts:
            if isinstance(item, dict):
                note = item.get("note", "")
                field_name = item.get("field_name", "")
                chosen = json_safe(item.get("chosen_value", ""))
                lines.append(f"- **{field_name}** — chosen: {chosen}. {note}")
            else:
                lines.append(f"- {item}")
    else:
        lines.append("No conflicts or notes recorded.")

    people = extraction.get("key_people", [])
    lines.extend(["", "## People or Entities", ""])
    if isinstance(people, list) and people:
        for person in people:
            if isinstance(person, dict):
                lines.append(f"- {person.get('full_name') or person.get('name') or json_safe(person)}")
            else:
                lines.append(f"- {person}")
    else:
        lines.append("None found.")

    warnings = extraction.get("warnings", [])
    lines.extend(["", "## Warnings", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("No warnings.")

    return "\n".join(lines).strip() + "\n"


def build_default_report(extraction: dict[str, Any]) -> str:
    lines = [
        "# Document Extraction Report",
        "",
        f"Template: `{extraction.get('template_name', '')}`",
        f"Document type: `{extraction.get('document_type', '')}`",
        "",
        "## Short Summary",
        "",
        extraction.get("short_summary", "") or "No short summary.",
        "",
        "## Long Summary",
        "",
        extraction.get("long_summary", "") or "No long summary.",
        "",
        "## Field Values",
        "",
    ]

    lines.extend(field_lines(extraction.get("final_fields", {})))

    warnings = extraction.get("warnings", [])
    lines.extend(["## Warnings", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("No warnings.")

    return "\n".join(lines).strip() + "\n"


def build_report_from_extraction(output_dir: Path) -> str:
    extraction = read_json(output_dir / "structured_extraction.json", fallback={})
    if not extraction:
        report = "# Document Extraction Report\n\nNo structured extraction was found.\n"
        write_text(output_dir / "final_report.md", report)
        return report

    analysis_mode = extraction.get("analysis_mode", "")

    if analysis_mode == "individual_document_analysis":
        report = build_individual_report(extraction)
        write_text(output_dir / "final_report.md", report)
        return report

    if analysis_mode == "consolidated_dossier":
        report = build_dossier_report(extraction)
        write_text(output_dir / "final_report.md", report)
        write_text(output_dir / "consolidated_dossier.md", report)
        return report

    report = build_default_report(extraction)
    write_text(output_dir / "final_report.md", report)
    return report
