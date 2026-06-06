from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .extraction_runner import extract_json_from_text, run_template_extraction
from .llm_runner import run_ollama_llm
from .report_writer import build_report_from_extraction
from .template_loader import blank_structured_extraction
from .utils import read_json, write_json, write_text


INDIVIDUAL_MODE = "Individual Document Analysis"
DOSSIER_MODE = "Consolidated Dossier"

ANALYSIS_MODE_OPTIONS = [INDIVIDUAL_MODE, DOSSIER_MODE]


def analysis_mode_slug(analysis_mode: str) -> str:
    if analysis_mode == DOSSIER_MODE:
        return "consolidated_dossier"
    return "individual_document_analysis"


def blank_analysis_output(
    template: dict[str, Any],
    doc_results: list[dict[str, Any]],
    analysis_mode: str,
) -> dict[str, Any]:
    source_files = [item.get("file_name", "") for item in doc_results]
    slug = analysis_mode_slug(analysis_mode)

    if analysis_mode == DOSSIER_MODE:
        blank = blank_structured_extraction(template, source_files=source_files)
        blank["analysis_mode"] = slug
        blank["conflicts_or_notes"] = []
        return blank

    individual_analyses = []
    for result in doc_results:
        item = blank_structured_extraction(template, source_files=[result.get("file_name", "")])
        item["analysis_mode"] = slug
        item["doc_id"] = result.get("doc_id")
        item["file_name"] = result.get("file_name")
        individual_analyses.append(item)

    return {
        "analysis_mode": slug,
        "template_name": template.get("template_name"),
        "document_type": template.get("document_type", ""),
        "source_files": source_files,
        "short_summary": "",
        "long_summary": "",
        "documents": [
            {
                "doc_id": result.get("doc_id"),
                "file_name": result.get("file_name"),
                "document_category": template.get("document_type", ""),
                "short_summary": "",
            }
            for result in doc_results
        ],
        "individual_document_analyses": individual_analyses,
        "warnings": [],
    }


def save_blank_analysis_output(
    template: dict[str, Any],
    doc_results: list[dict[str, Any]],
    output_dir: Path,
    analysis_mode: str,
) -> dict[str, Any]:
    blank = blank_analysis_output(template, doc_results, analysis_mode)
    write_json(output_dir / "structured_extraction.json", blank)
    write_json(output_dir / "selected_template.json", template)

    if analysis_mode == DOSSIER_MODE:
        write_json(output_dir / "consolidated_dossier.json", blank)
    else:
        write_json(output_dir / "individual_analyses.json", blank.get("individual_document_analyses", []))

    return blank


def filter_items_by_doc_id(items: Any, doc_id: str) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []

    filtered = []
    for item in items:
        if isinstance(item, dict) and str(item.get("doc_id", "")) == str(doc_id):
            filtered.append(item)
    return filtered


def prepare_doc_specific_context(root_output_dir: Path, doc_output_dir: Path, doc_id: str) -> None:
    context_files = [
        "selected_crop_manifest.json",
        "vlm_results.json",
        "cleaned_tables.json",
    ]

    for file_name in context_files:
        source_data = read_json(root_output_dir / file_name, fallback=[])
        filtered = filter_items_by_doc_id(source_data, doc_id)
        write_json(doc_output_dir / file_name, filtered)


def run_individual_document_analysis(
    template: dict[str, Any],
    doc_results: list[dict[str, Any]],
    output_dir: Path,
    model: str,
    ollama_url: str,
) -> dict[str, Any]:
    individual_analyses: list[dict[str, Any]] = []

    for result in doc_results:
        doc_id = result.get("doc_id", "")
        doc_output_dir = Path(result["output_dir"])
        prepare_doc_specific_context(output_dir, doc_output_dir, doc_id)

        extraction = run_template_extraction(
            template=template,
            doc_results=[result],
            output_dir=doc_output_dir,
            model=model,
            ollama_url=ollama_url,
        )

        extraction["analysis_mode"] = "individual_document_analysis"
        extraction["doc_id"] = doc_id
        extraction["file_name"] = result.get("file_name", "")
        write_json(doc_output_dir / "structured_extraction.json", extraction)
        individual_analyses.append(extraction)

    aggregate = {
        "analysis_mode": "individual_document_analysis",
        "template_name": template.get("template_name"),
        "document_type": template.get("document_type", ""),
        "source_files": [result.get("file_name", "") for result in doc_results],
        "short_summary": f"Generated individual analysis for {len(individual_analyses)} document(s).",
        "long_summary": "Each uploaded document was analyzed separately using the selected template.",
        "documents": [
            {
                "doc_id": extraction.get("doc_id", ""),
                "file_name": extraction.get("file_name", ""),
                "document_category": extraction.get("document_type", ""),
                "short_summary": extraction.get("short_summary", ""),
            }
            for extraction in individual_analyses
        ],
        "individual_document_analyses": individual_analyses,
        "warnings": [],
    }

    write_json(output_dir / "individual_analyses.json", individual_analyses)
    write_json(output_dir / "structured_extraction.json", aggregate)
    build_report_from_extraction(output_dir)

    return aggregate


MERGE_SYSTEM_PROMPT = """
You merge multiple per-document analyses into one consolidated dossier.

Rules:
- Use only the individual document analyses, cleaned tables, and VLM summaries provided.
- Do not invent facts.
- If documents disagree, choose the best supported value and record the conflict in conflicts_or_notes.
- Keep evidence whenever possible.
- Output valid JSON only. No markdown fences.
"""


def build_dossier_schema(template: dict[str, Any], doc_results: list[dict[str, Any]]) -> dict[str, Any]:
    schema = blank_structured_extraction(
        template,
        source_files=[result.get("file_name", "") for result in doc_results],
    )
    schema["analysis_mode"] = "consolidated_dossier"
    schema["conflicts_or_notes"] = [
        {
            "field_name": "",
            "chosen_value": "",
            "alternative_values": [],
            "note": "",
            "evidence": [],
        }
    ]
    return schema


def build_dossier_prompt(
    template: dict[str, Any],
    doc_results: list[dict[str, Any]],
    individual_analyses: list[dict[str, Any]],
    output_dir: Path,
) -> str:
    schema = build_dossier_schema(template, doc_results)

    cleaned_tables = read_json(output_dir / "cleaned_tables.json", fallback=[])
    vlm_results = read_json(output_dir / "vlm_results.json", fallback=[])

    return f"""
Selected template:
{json.dumps(template, ensure_ascii=False, indent=2)}

Return JSON in exactly this general shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Individual document analyses:
{json.dumps(individual_analyses, ensure_ascii=False, indent=2)}

Cleaned tables:
{json.dumps(cleaned_tables, ensure_ascii=False, indent=2)}

VLM crop summaries:
{json.dumps(vlm_results, ensure_ascii=False, indent=2)}

Important:
- Produce one consolidated dossier across all uploaded documents.
- final_fields must use the exact field names from the selected template.
- Each final_fields item must contain value, answer, and evidence.
- conflicts_or_notes should list disagreements, weak evidence, or missing important fields.
- No confidence field.
- Output valid JSON only.
""".strip()


def fallback_dossier(
    template: dict[str, Any],
    doc_results: list[dict[str, Any]],
    individual_analyses: list[dict[str, Any]],
    warning: str,
    raw_llm_response: str | None = None,
) -> dict[str, Any]:
    dossier = build_dossier_schema(template, doc_results)
    dossier["short_summary"] = f"Generated a consolidated dossier from {len(individual_analyses)} document analysis result(s)."
    dossier["long_summary"] = "The merge model did not produce valid JSON, so a structured fallback dossier was created from the individual analyses."
    dossier["documents"] = [
        {
            "doc_id": item.get("doc_id", ""),
            "file_name": item.get("file_name", ""),
            "document_category": item.get("document_type", ""),
            "short_summary": item.get("short_summary", ""),
        }
        for item in individual_analyses
    ]
    dossier["individual_document_analyses"] = individual_analyses
    dossier["warnings"] = [warning]
    if raw_llm_response:
        dossier["raw_llm_response"] = raw_llm_response
    return dossier


def run_consolidated_dossier(
    template: dict[str, Any],
    doc_results: list[dict[str, Any]],
    output_dir: Path,
    model: str,
    ollama_url: str,
) -> dict[str, Any]:
    individual_output = run_individual_document_analysis(
        template=template,
        doc_results=doc_results,
        output_dir=output_dir,
        model=model,
        ollama_url=ollama_url,
    )

    individual_analyses = individual_output.get("individual_document_analyses", [])

    prompt = build_dossier_prompt(
        template=template,
        doc_results=doc_results,
        individual_analyses=individual_analyses,
        output_dir=output_dir,
    )

    raw = run_ollama_llm(
        model=model,
        ollama_url=ollama_url,
        user_content=prompt,
        system_prompt=MERGE_SYSTEM_PROMPT,
    )

    parsed = extract_json_from_text(raw)
    if parsed is None:
        parsed = fallback_dossier(
            template=template,
            doc_results=doc_results,
            individual_analyses=individual_analyses,
            warning="Dossier merge model did not return valid JSON. Fallback dossier was saved.",
            raw_llm_response=raw,
        )

    parsed.setdefault("analysis_mode", "consolidated_dossier")
    parsed.setdefault("template_name", template.get("template_name"))
    parsed.setdefault("document_type", template.get("document_type", ""))
    parsed.setdefault("source_files", [result.get("file_name", "") for result in doc_results])
    parsed.setdefault("individual_document_analyses", individual_analyses)
    parsed.setdefault("conflicts_or_notes", [])
    parsed.setdefault("warnings", [])

    write_json(output_dir / "consolidated_dossier.json", parsed)
    write_json(output_dir / "structured_extraction.json", parsed)

    report = build_report_from_extraction(output_dir)
    write_text(output_dir / "consolidated_dossier.md", report)

    return parsed
