# Pipeline Efficiency Analysis

Comprehensive architectural overview of the 8-phase Millenia Dossier pipeline, identifying bottlenecks and optimization opportunities.

---

## Phase 1: Upload

**Process:** File picker / text path / Browse UI → files copied to `run_dir/input/`

**Efficiency:** Clean and minimal.

**Concern:** Full file copy duplicates storage. A 50MB PDF deck results in the original plus a copy in `input/`.

---

## Phase 2: Docling (`docling_pipeline.py`)

**Process:** Iterates input PDFs sequentially → Docling OCR/parsing → extracts layout items → generates crop PNGs → outlined PDFs → per-doc markdown/JSON → writes `run_manifest.json`, `layout_items.json`, `raw_tables.json`, `visual_items.json`

### Bottlenecks

1. **No parallelism** — Each PDF is processed one at a time. Docling is CPU/GPU-intensive. With 10 PDFs, total time = sum of all 10, not `max(10)`.

2. **Cropping is synchronous** — For every table and picture, the pipeline crops from the PDF at 2× resolution using PyMuPDF. This blocks the next item.

3. **Quick_skim runs per-item** — `quick_skim_crop()` opens and analyzes each crop PNG immediately, adding ~50–100 ms per table/picture.

4. **Writes are all-or-nothing** — `all_layout_items`, `all_raw_tables`, and `all_visual_items` are accumulated in memory and written as monolithic JSON files at the end. If the process crashes on PDF 8, work from PDFs 1–7 is lost.

5. **Per-doc files are redundant** — Per-doc layout_items, raw_tables, and visual_items are saved in `parsed/doc_NNN/` but **never read back** by later phases. Only the monolithic merged files at the run root are used. This is unnecessary I/O.

### What's actually needed from this phase
- Text content + item metadata (type, page, bbox)
- Crop images for tables and pictures
- Table data structures

---

## Phase 3: Item Review (UI only, no processing)

**Process:** Human reviews layout items via expanders; sets status + notes

**Efficiency:** Fine. No server-side processing.

**Minor:** Limited to the first 100 items per view. For documents with 500+ items, filtering by type/page is required to navigate.

---

## Phase 4: Table Correction (`table_corrector.py`)

**Process:** For each selected table → build prompt (raw markdown + HTML + columns + rows) → LLM call → parse JSON → save to `cleaned_tables.json`

### Bottlenecks

1. **Sequential LLM calls** — Tables are corrected one at a time. Each table is an independent LLM call that could run in parallel.

2. **No cache** — Re-running corrects every table again, even if nothing changed. The prompt is deterministic given the raw table data, so identical inputs produce identical outputs (at the same temperature).

3. **Prompt bloat** — The prompt includes full raw markdown, raw HTML (8k chars), columns, and rows simultaneously. HTML alone at 8k chars adds significant context for a "clean this table" task.

---

## Phase 5: VLM Review (`visual_analyzer.py`)

**Process:** For each selected visual → build prompt + base64-encode crop image → VLM call → parse JSON → save to `image_summaries.json`

### Bottlenecks

1. **Same sequential pattern** as Table Correction.

2. **Image encoding overhead** — Every crop is base64-encoded into the JSON payload. A 500 KB crop becomes ~670 KB of base64 text, transmitted and processed by the LLM for every visual.

3. **`nearby_text` is duplicated** — It is extracted from layout_items during Docling and included in every visual's prompt, but identical nearby text is re-analyzed per visual with no deduplication.

4. **No cache** — Same as Table Correction.

---

## Phase 6: Build Feed (`feed_builder.py`)

**Process:** Reads layout_items, cleaned_tables, image_summaries → assembles markdown → writes `file_llm_feed.md`

**Efficiency:** Excellent. Pure data assembly, no LLM calls. Runs in milliseconds.

---

## Phase 7: 56-Field Extraction (`extraction.py`)

**Process:** Two sub-phases:
- **Step 1:** Split feed by `# Document:` → for each doc → LLM extracts 56 fields → per-doc JSON saved
- **Step 2:** All per-doc results → merge LLM produces `company_dossier_merged.json`

### Bottlenecks

1. **Most expensive phase** — N documents × LLM call per doc + 1 merge call = N + 1 LLM calls.

2. **Feed splitting is fragile** — Uses `# Document:` as the delimiter. If any document text contains `# Document:` naturally (e.g., a document titled "Document: Terms and Conditions"), the split will produce extra fragments.

3. **30k char truncation** — `doc["text"][:30000]`. Documents longer than 30k chars get arbitrarily truncated. Compound documents (merging multiple source files) could easily exceed this.

4. **Merge prompt can be enormous** — All per-doc fields are serialized to JSON in the prompt. With 20 documents × 56 fields, this could reach 50k+ tokens before the LLM starts reasoning.

5. **No incremental resumption** — If the merge fails, per-doc JSONs are preserved but the merge must run from scratch. Same input produces the same output (at same temperature), so caching would save the full time.

6. **Error reporting** — Failure on doc 5 saves docs 1–4 and continues with doc 6 (good pattern), but the UI shows "success" regardless of individual failures. You must inspect individual JSONs to find failures.

---

## Phase 8: Excel (`excel_exporter.py`)

**Process:** Reads all JSON artifacts → writes multi-sheet xlsx via openpyxl

**Efficiency:** Fine. Runs in seconds. Clean many-sheet architecture.

---

## Strategic Observations

### What's actually slow

| Phase | Approximate cost per item | Parallelizable? |
|-------|---------------------------|-----------------|
| Docling | 5–30 s per PDF | **Yes** — per-document |
| Table Correction | 3–10 s per table | **Yes** — per-table |
| VLM Review | 5–20 s per visual | **Yes** — per-visual |
| Extraction (per-doc) | 10–30 s per document | **Yes** — per-document |
| Extraction (merge) | 30–90 s total | No (single output) |

### Biggest potential wins

1. **Parallelism** — Phases 2, 4, 5, and 7 Step 1 are all embarrassingly parallel. Using `ThreadPoolExecutor` could cut total runtime by 3–5× on a multi-document run. The merge phase is the only truly sequential bottleneck.

2. **Incremental checkpointing** — Each phase's loop could check whether output already exists for an item before processing it. Combined with caching, this makes re-runs instant for already-completed items.

3. **System prompt for field definitions** — The 56 field definitions are repeated in every per-document prompt. Ollama's `/api/chat` supports `"role": "system"`. Moving field definitions to a system message would let models cache them rather than re-reading them from scratch each time.

4. **Feed splitting robustness** — Replacing the `# Document:` regex split with a UUID-anchored delimiter (e.g., `# Document::22a1f7d3-…`) would prevent accidental splits on document content.

5. **Docling output redundancy** — Per-doc files in `parsed/doc_NNN/` are written but never read back by later phases. Removing this saves disk I/O and complexity. The monolithic files at run root are the canonical source.

### Architecture verdict

The two-phase extraction (per-doc → merge) is **the right architecture** — it gives debuggable per-document artifacts and a single merge step. The sequential processing is the main performance tax, and that is an easy fix with `concurrent.futures`.
