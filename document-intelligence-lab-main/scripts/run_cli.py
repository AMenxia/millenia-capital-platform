from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from docintellab.config import get_config
from docintellab.docling_pipeline import run_docling_on_files
from docintellab.feed_builder import build_llm_feed
from docintellab.table_corrector import correct_tables
from docintellab.extraction import run_extraction
from docintellab.excel_exporter import create_excel_export
from docintellab.utils import now_run_id


def main():
    parser = argparse.ArgumentParser(description="Run Document Intelligence Lab from CLI")
    parser.add_argument("pdfs", nargs="+", help="PDF files to process")
    parser.add_argument("--template", default="generic_document")
    parser.add_argument("--analysis-mode", default="Individual Document Analysis", choices=["Individual Document Analysis", "Consolidated Dossier"])
    parser.add_argument("--skip-table-llm", action="store_true")
    parser.add_argument("--run-extraction", action="store_true")
    args = parser.parse_args()

    cfg = get_config()
    run_dir = cfg.runs_dir / now_run_id("cli")
    run_dir.mkdir(parents=True, exist_ok=True)
    input_paths = [Path(p) for p in args.pdfs]

    print(f"Run dir: {run_dir}")
    run_docling_on_files(input_paths, run_dir)
    correct_tables(run_dir, llm_model=cfg.llm_model, ollama_url=cfg.ollama_url, use_llm=not args.skip_table_llm)
    build_llm_feed(run_dir)
    if args.run_extraction:
        run_extraction(run_dir, template_name=args.template, analysis_mode=args.analysis_mode, llm_model=cfg.llm_model, ollama_url=cfg.ollama_url)
    create_excel_export(run_dir)
    print("Done.")


if __name__ == "__main__":
    main()
