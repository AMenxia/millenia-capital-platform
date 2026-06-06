# AGENTS.md — Millenia Dossier Workspace

This workspace contains two source projects and a third merged project being built from them.

## Projects

| Directory | Description |
|-----------|-------------|
| `document-intelligence-lab-main/` | Streamlit UI app for PDF→Docling→Review→Extraction pipeline. Source for the tabbed UI. |
| `HealthRate/` | CLI-only 3-phase pipeline: snapshot → text extraction → company dossier with 56 fields. Source for the field schema + evidence tracking. |
| `Millenia-Dossier/` | **Merged project** — Streamlit UI (from DocIntelLab) + HealthRate's 56-field schema with evidence tracking. |

## Key Architecture

- Both source projects use **Docling** as the core PDF parser and **Ollama** as the LLM server.
- The merged project takes DocIntelLab's Streamlit tabbed pipeline UI and HealthRate's inline `FIELD_DEFINITIONS` dict (56 fields) with per-field `value` + `answer` + `evidence[]` structure (see `HealthRate/03_phase3_llm_only_explicit_fields.py` for the reference schema).
- Output format: `{short_summary, long_summary, final_fields: {field: {value, answer, evidence: [{doc_id, file_name, quote, page_start, page_end}]}}, documents: [...], model_used, ...}`
- Two-phase extraction: per-document JSONs saved to `json_for_each_file/<doc_id>.json`, then merged into `company_dossier_merged.json`.
- Package name: `millenia_dossier` (not `docintellab`). Source lives in `src/millenia_dossier/`.
- The 56 `FIELD_DEFINITIONS` live inline in `src/millenia_dossier/extraction.py` (not in templates).

## Running the Apps

**Millenia Dossier** (merged project):
```
streamlit run Millenia-Dossier/app/streamlit_app.py
```
Config via `.env` in `Millenia-Dossier/`: `OLLAMA_URL`, `LLM_MODEL`, `VLM_MODEL`, `RUNS_DIR`
Tab order: Upload → Docling → Item Review → Table Correction → VLM Review → Build Feed → 56-Field Extraction → Excel
Extraction runs in 2 steps: (1) per-document JSONs, (2) merge into dossier.

**Document Intelligence Lab** (source for UI):
```
streamlit run document-intelligence-lab-main/app/streamlit_app.py
```

**HealthRate Phase 3** (source for field schema + evidence):
```
cd HealthRate
python 03_phase3_llm_only_explicit_fields.py --model <model-name> --timeout 900
```
Expects `Phase 01_TextExtraction/<doc_id>/file_llm_feed.md` as input; outputs to `Phase03_ExplicitFieldsV2/<model>/company_dossier_merged.json`

## Setup Requirements

- Python 3.11+, Ollama running at `http://localhost:11434`
- Install: `pip install streamlit pandas openpyxl python-dotenv Pillow requests pymupdf numpy docling`
- For Docling: may pull large model files on first run
- Millenia-Dossier adds `src/` to `sys.path` at runtime (see `app/streamlit_app.py:13-15`)

## Important Conventions

- The merged project's extraction tab writes per-doc JSONs to `json_for_each_file/` then merges to `company_dossier_merged.json`.
- Field definitions are inline in Python at `src/millenia_dossier/extraction.py:FIELD_DEFINITIONS` (56 fields), following HealthRate's approach.
- Evidence per field: `{doc_id, file_name, quote, page_start, page_end}`.
- Run directories follow the naming pattern `runs/<run_id>/`.
- Excel export is named `millenia_dossier_export.xlsx` (not `document_intelligence_export.xlsx`).
