from __future__ import annotations

from pathlib import Path

import fitz


def merge_pdfs(pdf_paths: list[Path], output_path: Path) -> Path:
    """
    Merge multiple PDFs into one PDF.

    This is used for Streamlit preview so the app can show one combined
    outlined PDF instead of embedding many individual outlined PDFs.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged = fitz.open()

    for pdf_path in pdf_paths:
        if pdf_path is None:
            continue

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            continue

        try:
            source = fitz.open(str(pdf_path))
            merged.insert_pdf(source)
            source.close()
        except Exception:
            continue

    if len(merged) == 0:
        merged.close()
        raise ValueError("No valid PDFs were found to merge.")

    merged.save(str(output_path))
    merged.close()

    return output_path
