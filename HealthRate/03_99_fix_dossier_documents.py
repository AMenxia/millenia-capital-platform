
"""
03_99_fix_dossier_documents.py

Fix only the "documents" list inside company_dossier_merged.json.

This script does NOT use an LLM.
It rebuilds the documents list from:

    Phase03_ExplicitFieldsV2/<model_folder>/json_for_each_file/*.json

It ignores:
    all_field_candidates.json

Run from project root:
    python 03_99_fix_dossier_documents.py --model-folder deepseek-v3.2_cloud

Or direct path:
    python 03_99_fix_dossier_documents.py --phase3-model-folder "C:\\Users\\amenx\\Desktop\\HealthRate\\Phase03_ExplicitFieldsV2\\deepseek-v3.2_cloud"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PHASE3_FOLDER_NAME = "Phase03_ExplicitFieldsV2"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def clean_doc_id_from_filename(path: Path) -> str:
    name = path.stem.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return name or "document"


def resolve_phase3_model_folder(args: argparse.Namespace) -> Path:
    if args.phase3_model_folder:
        folder = Path(args.phase3_model_folder).expanduser().resolve()

        if not folder.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder}")

        return folder

    project_root = Path.cwd().resolve()
    phase3_root = project_root / PHASE3_FOLDER_NAME

    if not phase3_root.exists():
        raise FileNotFoundError(f"Could not find: {phase3_root}")

    if args.model_folder:
        folder = phase3_root / args.model_folder

        if not folder.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder}")

        return folder

    subfolders = [p for p in phase3_root.iterdir() if p.is_dir()]

    if len(subfolders) == 1:
        print(f"[auto] Using only model folder found: {subfolders[0]}")
        return subfolders[0]

    raise FileNotFoundError(
        "Could not auto-detect the model folder.\n"
        "Use --model-folder or --phase3-model-folder."
    )


def build_document_record(json_path: Path, data: dict[str, Any]) -> dict[str, Any]:
    doc_id = data.get("doc_id") or clean_doc_id_from_filename(json_path)
    file_name = data.get("file_name") or doc_id

    return {
        "doc_id": doc_id,
        "file_name": file_name,
        "document_category": data.get("document_category"),
        "short_summary": data.get("short_summary"),
        "long_summary": data.get("long_summary"),
        "source_markdown_file": data.get("source_markdown_file"),
        "phase1_folder": data.get("phase1_folder"),
        "chunk_count": data.get("chunk_count"),
        "source_json_file": str(json_path),
    }


def load_all_document_records(json_for_each_file_folder: Path) -> list[dict[str, Any]]:
    records = []

    for json_path in sorted(json_for_each_file_folder.glob("*.json"), key=lambda p: p.name.lower()):
        if json_path.name == "all_field_candidates.json":
            continue

        if json_path.name == "company_dossier_merged.json":
            continue

        try:
            data = read_json(json_path)

            if not isinstance(data, dict):
                print(f"[skip] Not a JSON object: {json_path.name}")
                continue

            records.append(build_document_record(json_path, data))

        except Exception as e:
            print(f"[warning] Could not read {json_path.name}: {e}")

    return records


def fix_dossier_documents(args: argparse.Namespace) -> Path:
    model_folder = resolve_phase3_model_folder(args)

    dossier_path = model_folder / "company_dossier_merged.json"
    json_for_each_file_folder = model_folder / "json_for_each_file"

    if not dossier_path.exists():
        raise FileNotFoundError(f"Missing merged dossier: {dossier_path}")

    if not json_for_each_file_folder.exists():
        raise FileNotFoundError(f"Missing json_for_each_file folder: {json_for_each_file_folder}")

    dossier = read_json(dossier_path)

    if not isinstance(dossier, dict):
        raise ValueError("company_dossier_merged.json is not a JSON object.")

    document_records = load_all_document_records(json_for_each_file_folder)

    old_documents_count = len(dossier.get("documents", [])) if isinstance(dossier.get("documents"), list) else 0

    if args.backup:
        backup_path = dossier_path.with_suffix(".backup_before_document_fix.json")
        write_json(backup_path, dossier)
        print(f"Backup saved: {backup_path}")

    dossier["documents"] = document_records
    dossier["source_document_count"] = len(document_records)

    write_json(dossier_path, dossier)

    print("=" * 80)
    print("DOSSIER DOCUMENT LIST FIXED")
    print("=" * 80)
    print(f"Model folder:           {model_folder}")
    print(f"Merged dossier:         {dossier_path}")
    print(f"Old documents count:    {old_documents_count}")
    print(f"New documents count:    {len(document_records)}")
    print("=" * 80)

    print("Documents now listed:")
    for record in document_records:
        print(f"  - {record['doc_id']} | {record.get('document_category')}")

    return dossier_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix company_dossier_merged.json so its documents list matches json_for_each_file/*.json."
    )

    parser.add_argument(
        "--model-folder",
        type=str,
        default=None,
        help="Folder inside Phase03_ExplicitFieldsV2, example: deepseek-v3.2_cloud",
    )

    parser.add_argument(
        "--phase3-model-folder",
        type=str,
        default=None,
        help="Direct path to Phase03_ExplicitFieldsV2/<model_folder>.",
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Save a backup before overwriting company_dossier_merged.json.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fix_dossier_documents(args)


if __name__ == "__main__":
    main()
