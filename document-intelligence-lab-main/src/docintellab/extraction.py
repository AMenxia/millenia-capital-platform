from __future__ import annotations

from pathlib import Path
from typing import Any

from .ollama_client import chat_json
from .templates import load_template
from .utils import read_json, safe_text, write_json, write_text


def build_extraction_prompt(feed_text: str, template: dict[str, Any], analysis_mode: str) -> str:
    fields = template.get("fields", [])
    return f"""
You are analyzing documents using a structured extraction template.

Analysis mode: {analysis_mode}
Template name: {template.get('template_name')}
Template description: {template.get('description')}

Fields to extract:
{fields}

Return strict JSON only with this schema:
{{
  "template_name": "...",
  "analysis_mode": "Individual Document Analysis or Consolidated Dossier",
  "short_summary": "quick summary",
  "field_values": [
    {{
      "field": "field name",
      "value": "extracted value or empty string",
      "source_name": "file/document name if known",
      "notes": "brief note if needed"
    }}
  ],
  "people_or_entities": ["important names/entities if any"],
  "conflicts_or_notes": ["conflicts, missing info, or uncertainty"]
}}

Rules:
- Use the document feed only.
- Do not invent missing facts.
- Keep values concise.
- If a field is not present, use an empty value and explain in notes.

Document feed:
{feed_text[:30000]}
""".strip()


def run_extraction(
    run_dir: Path,
    template_name: str,
    analysis_mode: str,
    llm_model: str,
    ollama_url: str,
) -> dict[str, Any]:
    feed_path = run_dir / "file_llm_feed.md"
    feed_text = feed_path.read_text(encoding="utf-8") if feed_path.exists() else ""
    template = load_template(template_name)
    try:
        result = chat_json(build_extraction_prompt(feed_text, template, analysis_mode), model=llm_model, ollama_url=ollama_url, temperature=0.1)
        if not isinstance(result, dict):
            raise ValueError("LLM returned non-object JSON")
    except Exception as exc:
        result = {
            "template_name": template_name,
            "analysis_mode": analysis_mode,
            "short_summary": "Extraction did not run successfully. Check Ollama/model settings.",
            "field_values": [{"field": f.get("name", str(f)), "value": "", "source_name": "", "notes": "LLM extraction failed."} for f in template.get("fields", [])],
            "people_or_entities": [],
            "conflicts_or_notes": [str(exc)],
        }

    output_name = "consolidated_dossier" if analysis_mode == "Consolidated Dossier" else "structured_extraction"
    write_json(run_dir / f"{output_name}.json", result)

    md_lines = [f"# {analysis_mode}", "", result.get("short_summary", "")]
    md_lines.append("\n## Field Values\n")
    for row in result.get("field_values", []):
        md_lines.append(f"- **{row.get('field')}**: {row.get('value')}  ")
        if row.get("notes"):
            md_lines.append(f"  - Notes: {row.get('notes')}")
    if result.get("conflicts_or_notes"):
        md_lines.append("\n## Conflicts or Notes\n")
        for note in result.get("conflicts_or_notes", []):
            md_lines.append(f"- {note}")
    write_text(run_dir / f"{output_name}.md", "\n".join(md_lines))
    return result
