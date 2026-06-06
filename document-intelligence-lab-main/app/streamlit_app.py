from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Allow running from repo root without installing as package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from docintellab.config import get_config
from docintellab.docling_pipeline import run_docling_on_files
from docintellab.excel_exporter import create_excel_export
from docintellab.extraction import run_extraction
from docintellab.feed_builder import build_llm_feed
from docintellab.table_corrector import correct_tables
from docintellab.templates import list_templates
from docintellab.utils import now_run_id, read_json, write_json
from docintellab.visual_analyzer import analyze_visuals, update_visual_notes

st.set_page_config(page_title="Document Intelligence Lab", layout="wide")


def render_pdf(path: Path, height: int = 700) -> None:
    if not path.exists():
        st.warning("PDF not found.")
        return
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    st.markdown(
        f'<embed src="data:application/pdf;base64,{b64}" width="100%" height="{height}" type="application/pdf">',
        unsafe_allow_html=True,
    )


def show_dataframe(data: Any, empty_message: str = "No data yet.") -> None:
    if not data:
        st.info(empty_message)
        return
    try:
        st.dataframe(pd.DataFrame(data), use_container_width=True)
    except Exception:
        st.json(data)


def init_state():
    if "run_dir" not in st.session_state:
        st.session_state.run_dir = None
    if "uploaded_paths" not in st.session_state:
        st.session_state.uploaded_paths = []


init_state()
config = get_config()

st.title("Document Intelligence Lab")
st.caption("DoclingDocument-centered parsing → LLM table correction → VLM visual review → structured outputs")

with st.sidebar:
    st.header("Settings")
    analysis_mode = st.selectbox("Analysis Mode", ["Individual Document Analysis", "Consolidated Dossier"])
    template_name = st.selectbox("Template", list_templates())
    llm_model = st.text_input("LLM Model", value=config.llm_model)
    vlm_model = st.text_input("VLM Model", value=config.vlm_model)
    ollama_url = st.text_input("Ollama URL", value=config.ollama_url)
    runs_dir = Path(st.text_input("Runs Directory", value=str(config.runs_dir)))
    st.divider()
    st.write("Optional switches")
    use_llm_table_correction = st.toggle("Use LLM table correction", value=True)
    use_vlm_visual_review = st.toggle("Use VLM visual review", value=True)

run_dir = Path(st.session_state.run_dir) if st.session_state.run_dir else None

tabs = st.tabs([
    "1) Upload + Template",
    "2) Run Docling",
    "3) Item Review + Quick Skim",
    "4) LLM Table Correction",
    "5) Image / Chart VLM Review",
    "6) Build LLM Feed",
    "7) Analysis + Extraction",
    "8) Excel Preview + Download",
])

with tabs[0]:
    st.subheader("1) Upload + Template")
    st.write("Upload one or more public-safe PDFs. For GitHub, do not use private/company documents.")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Create new run", type="primary"):
            new_run = runs_dir / now_run_id("docintellab")
            input_dir = new_run / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            for f in uploaded or []:
                out = input_dir / f.name
                out.write_bytes(f.getbuffer())
                saved.append(str(out))
            st.session_state.run_dir = str(new_run)
            st.session_state.uploaded_paths = saved
            st.success(f"Created run: {new_run}")
    with col2:
        st.write("Current run:")
        st.code(str(st.session_state.run_dir or "No run yet"))
        st.write("Template:", template_name)
        st.write("Analysis mode:", analysis_mode)

    if uploaded:
        st.markdown("### Uploaded file preview")
        first = uploaded[0]
        pdf_b64 = base64.b64encode(first.getvalue()).decode("utf-8")
        st.markdown(
            f'<embed src="data:application/pdf;base64,{pdf_b64}" width="100%" height="500" type="application/pdf">',
            unsafe_allow_html=True,
        )

with tabs[1]:
    st.subheader("2) Run Docling")
    st.write("This creates raw Docling outputs, layout_items.json, raw_tables.json, visual_items.json, crops, and outlined PDFs.")
    run_dir = Path(st.session_state.run_dir) if st.session_state.run_dir else None
    if not run_dir:
        st.warning("Create a run first.")
    else:
        input_paths = [Path(p) for p in st.session_state.uploaded_paths]
        if not input_paths:
            st.warning("No uploaded PDF paths found in this run.")
        else:
            st.write("Input files:")
            for p in input_paths:
                st.code(str(p))
            if st.button("Run Docling parser", type="primary"):
                with st.spinner("Running Docling..."):
                    try:
                        manifest = run_docling_on_files(input_paths, run_dir)
                        st.success("Docling parsing completed.")
                        st.json(manifest)
                    except Exception as exc:
                        st.error(str(exc))

        manifest = read_json(run_dir / "run_manifest.json", default=None)
        if manifest:
            st.markdown("### Outlined PDF Preview")
            docs = manifest.get("documents", [])
            labels = [d.get("file_name", d.get("doc_id")) for d in docs]
            if docs:
                choice = st.selectbox("Choose outlined PDF", labels)
                idx = labels.index(choice)
                outlined = docs[idx].get("outlined_pdf_path")
                if outlined:
                    render_pdf(Path(outlined), height=650)
                else:
                    st.info("Outlined PDF was not created. This can happen if bbox extraction failed for this Docling version.")

with tabs[2]:
    st.subheader("3) DoclingDocument Item Review + Quick Skim")
    run_dir = Path(st.session_state.run_dir) if st.session_state.run_dir else None
    if not run_dir or not (run_dir / "layout_items.json").exists():
        st.warning("Run Docling first.")
    else:
        layout_items = read_json(run_dir / "layout_items.json", default=[])
        st.write(f"Detected items: {len(layout_items)}")
        c1, c2, c3 = st.columns(3)
        with c1:
            type_filter = st.multiselect("Item types", sorted(set(i.get("item_type") for i in layout_items)), default=[])
        with c2:
            skim_filter = st.multiselect("Quick skim", sorted(set((i.get("quick_skim") or {}).get("label") for i in layout_items if i.get("quick_skim"))), default=[])
        with c3:
            page_filter = st.text_input("Page contains", value="")
        filtered = layout_items
        if type_filter:
            filtered = [i for i in filtered if i.get("item_type") in type_filter]
        if skim_filter:
            filtered = [i for i in filtered if (i.get("quick_skim") or {}).get("label") in skim_filter]
        if page_filter.strip():
            filtered = [i for i in filtered if str(i.get("page_no")) == page_filter.strip()]

        updated = False
        for item in filtered[:100]:
            with st.expander(f"{item.get('reading_order_index')} | {item.get('item_type')} | {item.get('item_id')} | page {item.get('page_no')}"):
                cols = st.columns([1, 2])
                with cols[0]:
                    crop = item.get("crop_path")
                    if crop and Path(crop).exists():
                        st.image(crop, caption="Crop preview", use_container_width=True)
                    skim = item.get("quick_skim")
                    if skim:
                        st.write("Quick skim:", skim.get("label"))
                        st.caption(skim.get("reason", ""))
                    status_options = ["keep", "discard", "needs_review"]
                    if item.get("item_type") == "table":
                        status_options = ["keep", "discard", "needs_correction", "corrected"]
                    new_status = st.selectbox(
                        "Human status",
                        status_options,
                        index=status_options.index(item.get("human_status")) if item.get("human_status") in status_options else 0,
                        key=f"status_{item.get('item_id')}",
                    )
                    note = st.text_area("Human note", value=item.get("human_note", ""), key=f"note_{item.get('item_id')}")
                    if new_status != item.get("human_status") or note != item.get("human_note", ""):
                        item["human_status"] = new_status
                        item["human_note"] = note
                        updated = True
                with cols[1]:
                    st.write("Text/raw preview")
                    st.text_area("", value=item.get("text", "")[:4000], height=200, key=f"text_{item.get('item_id')}", disabled=True)
                    st.json({k: item.get(k) for k in ["doc_id", "file_name", "page_no", "bbox", "label"]})
        if updated:
            if st.button("Save item review updates"):
                write_json(run_dir / "layout_items.json", layout_items)
                # Mirror visual item notes/status by item_id.
                visual_items = read_json(run_dir / "visual_items.json", default=[])
                by_item = {i.get("item_id"): i for i in layout_items}
                for v in visual_items:
                    src = by_item.get(v.get("item_id"))
                    if src:
                        v["human_status"] = src.get("human_status", v.get("human_status"))
                        v["human_note"] = src.get("human_note", v.get("human_note", ""))
                write_json(run_dir / "visual_items.json", visual_items)
                st.success("Saved review updates.")

with tabs[3]:
    st.subheader("4) LLM Table Correction")
    run_dir = Path(st.session_state.run_dir) if st.session_state.run_dir else None
    if not run_dir or not (run_dir / "raw_tables.json").exists():
        st.warning("Run Docling first.")
    else:
        raw_tables = read_json(run_dir / "raw_tables.json", default=[])
        st.write(f"Raw tables found: {len(raw_tables)}")
        selected_tables = st.multiselect("Tables to correct", [t.get("table_id") for t in raw_tables], default=[t.get("table_id") for t in raw_tables])
        for table in raw_tables:
            with st.expander(f"{table.get('table_id')} | {table.get('file_name')} | page {table.get('page_no')}"):
                cols = st.columns([1, 2])
                with cols[0]:
                    crop = table.get("crop_path")
                    if crop and Path(crop).exists():
                        st.image(crop, caption="Table crop preview", use_container_width=True)
                with cols[1]:
                    st.markdown("Raw Docling table preview")
                    st.text_area("Raw table markdown", value=table.get("raw_markdown", "")[:6000], height=220, key=f"rawtable_{table.get('table_id')}")
        if st.button("Run table correction", type="primary"):
            with st.spinner("Correcting tables..."):
                cleaned = correct_tables(run_dir, llm_model=llm_model, ollama_url=ollama_url, table_ids=selected_tables, use_llm=use_llm_table_correction)
                st.success(f"Created cleaned_tables.json with {len(cleaned)} table(s).")
        cleaned_tables = read_json(run_dir / "cleaned_tables.json", default=[])
        if cleaned_tables:
            st.markdown("### Cleaned table preview")
            for table in cleaned_tables:
                st.write(f"**{table.get('table_id')}** — {table.get('caption','')}")
                show_dataframe(table.get("rows", []), empty_message="No rows in this cleaned table.")
                if table.get("issues"):
                    st.caption("Issues: " + "; ".join(map(str, table.get("issues"))))

with tabs[4]:
    st.subheader("5) Image / Chart VLM Review")
    st.write("Add optional human context notes before sending visual crops to the VLM.")
    run_dir = Path(st.session_state.run_dir) if st.session_state.run_dir else None
    if not run_dir or not (run_dir / "visual_items.json").exists():
        st.warning("Run Docling first.")
    else:
        visual_items = read_json(run_dir / "visual_items.json", default=[])
        st.write(f"Visual items found: {len(visual_items)}")
        updates = {}
        selected_visuals = []
        for item in visual_items:
            with st.expander(f"{item.get('visual_id')} | {item.get('file_name')} | page {item.get('page_no')}"):
                cols = st.columns([1, 2])
                with cols[0]:
                    crop = item.get("crop_path")
                    if crop and Path(crop).exists():
                        st.image(crop, caption="Visual crop", use_container_width=True)
                    st.write("Quick skim:", (item.get("quick_skim") or {}).get("label", "none"))
                    status = st.selectbox(
                        "Status",
                        ["keep", "discard", "needs_review"],
                        index=["keep", "discard", "needs_review"].index(item.get("human_status")) if item.get("human_status") in ["keep", "discard", "needs_review"] else 0,
                        key=f"visual_status_{item.get('visual_id')}",
                    )
                    run_this = st.checkbox("Run VLM on this visual", value=status in {"keep", "needs_review"}, key=f"run_vlm_{item.get('visual_id')}")
                    if run_this:
                        selected_visuals.append(item.get("visual_id"))
                with cols[1]:
                    note = st.text_area("Reviewer note for VLM context", value=item.get("human_note", ""), key=f"visual_note_{item.get('visual_id')}")
                    st.text_area("Nearby text", value=item.get("nearby_text", "")[:3000], height=160, disabled=True, key=f"nearby_{item.get('visual_id')}")
                    updates[item.get("visual_id")] = {"human_status": status, "human_note": note}
        if st.button("Save visual notes/statuses"):
            update_visual_notes(run_dir, updates)
            st.success("Saved visual notes/statuses.")
        if st.button("Run VLM on selected visuals", type="primary", disabled=not use_vlm_visual_review):
            update_visual_notes(run_dir, updates)
            with st.spinner("Running VLM..."):
                summaries = analyze_visuals(run_dir, vlm_model=vlm_model, ollama_url=ollama_url, visual_ids=selected_visuals)
                st.success(f"Created image_summaries.json with {len(summaries)} summaries.")
        summaries = read_json(run_dir / "image_summaries.json", default=[])
        if summaries:
            st.markdown("### Image summary preview")
            show_dataframe(summaries)

with tabs[5]:
    st.subheader("6) Build LLM Feed")
    run_dir = Path(st.session_state.run_dir) if st.session_state.run_dir else None
    if not run_dir or not (run_dir / "layout_items.json").exists():
        st.warning("Run Docling first.")
    else:
        if st.button("Build file_llm_feed.md", type="primary"):
            feed = build_llm_feed(run_dir)
            st.success("Built file_llm_feed.md")
            st.text_area("Feed preview", value=feed[:15000], height=500)
        feed_path = run_dir / "file_llm_feed.md"
        if feed_path.exists():
            feed_text = feed_path.read_text(encoding="utf-8")
            st.download_button("Download file_llm_feed.md", data=feed_text, file_name="file_llm_feed.md")
            st.text_area("Current feed", value=feed_text[:15000], height=500)

with tabs[6]:
    st.subheader("7) Analysis + Extraction")
    run_dir = Path(st.session_state.run_dir) if st.session_state.run_dir else None
    if not run_dir or not (run_dir / "file_llm_feed.md").exists():
        st.warning("Build the LLM feed first.")
    else:
        if st.button("Run analysis/extraction", type="primary"):
            with st.spinner("Running extraction..."):
                result = run_extraction(run_dir, template_name=template_name, analysis_mode=analysis_mode, llm_model=llm_model, ollama_url=ollama_url)
                st.success("Extraction complete.")
                st.json(result)
        dossier = read_json(run_dir / "consolidated_dossier.json", default=None)
        extraction = read_json(run_dir / "structured_extraction.json", default=None)
        result = dossier or extraction
        if result:
            st.markdown("### Extraction preview")
            st.write(result.get("short_summary", ""))
            show_dataframe(result.get("field_values", []))

with tabs[7]:
    st.subheader("8) Excel Preview + Download")
    run_dir = Path(st.session_state.run_dir) if st.session_state.run_dir else None
    if not run_dir:
        st.warning("Create a run first.")
    else:
        if st.button("Generate Excel workbook", type="primary"):
            out = create_excel_export(run_dir)
            st.success(f"Created {out.name}")
        excel_path = run_dir / "document_intelligence_export.xlsx"
        if excel_path.exists():
            st.download_button(
                "Download Excel workbook",
                data=excel_path.read_bytes(),
                file_name="document_intelligence_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        st.markdown("### Preview key output tables")
        output_choice = st.selectbox("Preview", ["cleaned_tables", "field_values", "image_summaries", "layout_items"])
        if output_choice == "cleaned_tables":
            cleaned_tables = read_json(run_dir / "cleaned_tables.json", default=[])
            rows = []
            for table in cleaned_tables:
                for i, row in enumerate(table.get("rows", []) or [], start=1):
                    rows.append({"table_id": table.get("table_id"), "row_number": i, **row})
            show_dataframe(rows)
        elif output_choice == "field_values":
            result = read_json(run_dir / "consolidated_dossier.json", default=None) or read_json(run_dir / "structured_extraction.json", default={})
            show_dataframe(result.get("field_values", []) if isinstance(result, dict) else [])
        elif output_choice == "image_summaries":
            show_dataframe(read_json(run_dir / "image_summaries.json", default=[]))
        elif output_choice == "layout_items":
            show_dataframe(read_json(run_dir / "layout_items.json", default=[]))
