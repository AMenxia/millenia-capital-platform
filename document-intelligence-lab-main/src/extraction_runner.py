from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .llm_runner import run_ollama_llm
from .template_loader import blank_structured_extraction
from .utils import read_json, read_text, write_json


SYSTEM_PROMPT = """
You extract structured data from parsed documents.

Rules:
- Use only the provided document text, cleaned tables, selected crop metadata, and VLM crop summaries.
- Do not invent facts.
- If a value is not found, use null.
- Keep the requested JSON shape.
- For each field, include value, answer, and evidence.
- Evidence should be a list of objects with quote, doc_id, page_start, and page_end when possible.
- No confidence score.
- Output valid JSON only. No markdown.
"""


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


def build_extraction_schema(template: dict[str, Any], doc_results: list[dict[str, Any]]) -> dict[str, Any]:
    source_files = [d.get("file_name", "") for d in doc_results]
    return blank_structured_extraction(template, source_files=source_files)


def build_extraction_prompt(template: dict[str, Any], doc_results: list[dict[str, Any]], output_dir: Path) -> str:
    schema = build_extraction_schema(template, doc_results)
    selected_crops = read_json(output_dir / "selected_crop_manifest.json", fallback=[])
    vlm_results = read_json(output_dir / "vlm_results.json", fallback=[])
    cleaned_tables = read_json(output_dir / "cleaned_tables.json", fallback=[])

    docs_text = []
    for result in doc_results:
        md = read_text(Path(result["markdown_path"]), "")
        docs_text.append(
            {
                "doc_id": result.get("doc_id"),
                "file_name": result.get("file_name"),
                "markdown_excerpt": md[:50000],
            }
        )

    return f"""
Selected template:
{json.dumps(template, ensure_ascii=False, indent=2)}

Return JSON in exactly this general shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Parsed documents:
{json.dumps(docs_text, ensure_ascii=False, indent=2)}

Cleaned tables:
{json.dumps(cleaned_tables, ensure_ascii=False, indent=2)}

Selected crop manifest:
{json.dumps(selected_crops, ensure_ascii=False, indent=2)}

VLM results:
{json.dumps(vlm_results, ensure_ascii=False, indent=2)}

Important:
- The output must contain short_summary, long_summary, final_fields, key_people, documents, visual_evidence, and warnings.
- final_fields must use the exact field names from the selected template.
- Each final_fields item must contain value, answer, and evidence.
- Use cleaned_tables when they provide better table structure than raw Markdown.
- No confidence field.
""".strip()


def run_template_extraction(
    template: dict[str, Any],
    doc_results: list[dict[str, Any]],
    output_dir: Path,
    model: str,
    ollama_url: str,
) -> dict[str, Any]:
    prompt = build_extraction_prompt(template, doc_results, output_dir)

    raw = run_ollama_llm(
        model=model,
        ollama_url=ollama_url,
        user_content=prompt,
        system_prompt=SYSTEM_PROMPT,
    )

    parsed = extract_json_from_text(raw)
    if parsed is None:
        parsed = build_extraction_schema(template, doc_results)
        parsed.setdefault("warnings", []).append("LLM did not return valid JSON. Blank template was saved.")
        parsed["raw_llm_response"] = raw

    parsed.setdefault("template_name", template.get("template_name"))
    parsed.setdefault("document_type", template.get("document_type"))
    parsed.setdefault("source_files", [d.get("file_name", "") for d in doc_results])
    parsed.setdefault("warnings", [])

    write_json(output_dir / "structured_extraction.json", parsed)
    write_json(output_dir / "selected_template.json", template)

    return parsed


def save_blank_extraction(template: dict[str, Any], doc_results: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    blank = build_extraction_schema(template, doc_results)
    write_json(output_dir / "structured_extraction.json", blank)
    write_json(output_dir / "selected_template.json", template)
    return blank
