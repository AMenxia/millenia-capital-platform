from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st


def render_file_preview(file_type: str, file_name: str, file_bytes: bytes) -> None:
    if file_type.startswith("image/"):
        st.image(file_bytes, caption=file_name, use_container_width=True)
        return

    if file_type == "application/pdf" or file_name.lower().endswith(".pdf"):
        pdf_b64 = base64.b64encode(file_bytes).decode("utf-8")
        st.markdown(
            f'<embed src="data:application/pdf;base64,{pdf_b64}" width="100%" height="750" type="application/pdf">',
            unsafe_allow_html=True,
        )
        return

    st.info("Preview is available for PDFs and images.")


def render_pdf_path(pdf_path: Path, height: int = 850) -> None:
    if not pdf_path.exists():
        st.warning(f"PDF not found: {pdf_path}")
        return

    pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
    st.markdown(
        f'<embed src="data:application/pdf;base64,{pdf_b64}" width="100%" height="{height}" type="application/pdf">',
        unsafe_allow_html=True,
    )
