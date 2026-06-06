
"""
03_99_fix_dossier_fields.py

Fix company_dossier_merged.json so final_fields contains ALL expected fields
in the same order as the Phase 3 prompt.

This script does NOT use an LLM.
It only patches the merged dossier structure.

What it does:
1. Opens:
   Phase03_ExplicitFieldsV2/<model_folder>/company_dossier_merged.json

2. Rebuilds:
   dossier["final_fields"]

3. Keeps existing field data if already present.

4. Adds missing fields as:
   {
     "value": null,
     "answer": "Not clearly stated in the provided documents.",
     "evidence": []
   }

5. Saves the fixed company_dossier_merged.json.

Run from project root:
    python 03_99_fix_dossier_fields.py --model-folder deepseek-v3.2_cloud

Optional backup:
    python 03_99_fix_dossier_fields.py --model-folder deepseek-v3.2_cloud --backup

Direct path:
    python 03_99_fix_dossier_fields.py --phase3-model-folder "C:\\Users\\amenx\\Desktop\\HealthRate\\Phase03_ExplicitFieldsV2\\deepseek-v3.2_cloud"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PHASE3_FOLDER_NAME = "Phase03_ExplicitFieldsV2"


# Same order as the Phase 3 prompt / FIELD_DEFINITIONS.
FIELD_ORDER = [
    "company_name",
    "legal_company_name",
    "website",
    "one_line_description",
    "industry_or_sector",
    "company_stage_or_company_idea",
    "headquarters_location",
    "incorporation_location",
    "year_founded",
    "legal_entity_structure",
    "company_problem",
    "company_solution",
    "business_model",
    "target_customers",
    "competitive_advantage",
    "product_status",
    "use_of_funds",
    "key_milestones",
    "funding_milestones",
    "cash_on_hand",
    "current_round",
    "amount_raised_to_date",
    "target_raise_amount",
    "minimum_investment",
    "previous_investors",
    "valuation_last_round",
    "valuation_current_ask",
    "pre_money_valuation",
    "post_money_valuation",
    "instrument",
    "revenue_monthly_annual",
    "mrr_arr",
    "growth_rate_mom_yoy",
    "gross_margin",
    "burn_rate",
    "runway_months",
    "ltv",
    "payback_period",
    "churn_rate",
    "users_total_active",
    "dau_mau",
    "sales_pipeline",
    "conversion_rates",
    "average_order_value",
    "retention_rates",
    "engagement_metrics",
    "partnerships",
    "contracts_signed",
    "lois",
    "key_hires",
    "incorporation_documents",
    "ip_ownership",
    "trademarks_patents",
    "regulatory_exposure",
    "risk_disclosures",
    "market_size",
]


EMPTY_FIELD = {
    "value": None,
    "answer": "Not clearly stated in the provided documents.",
    "evidence": [],
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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


def normalize_field_object(field_value: Any) -> dict[str, Any]:
    """
    Make sure every field is shaped like:
        {
          "value": ...,
          "answer": ...,
          "evidence": [...]
        }

    If the current field is a plain string/list/number, preserve it as value.
    """

    if isinstance(field_value, dict):
        fixed = dict(field_value)

        if "value" not in fixed:
            # Sometimes earlier scripts used "final_value" or "clean_value".
            if "final_value" in fixed:
                fixed["value"] = fixed.get("final_value")
            elif "clean_value" in fixed:
                fixed["value"] = fixed.get("clean_value")
            else:
                fixed["value"] = None

        if "answer" not in fixed:
            value = fixed.get("value")
            if value is None or value == "" or value == [] or value == {}:
                fixed["answer"] = EMPTY_FIELD["answer"]
            else:
                fixed["answer"] = str(value)

        if "evidence" not in fixed or fixed.get("evidence") is None:
            fixed["evidence"] = []

        if not isinstance(fixed["evidence"], list):
            fixed["evidence"] = [fixed["evidence"]]

        return fixed

    if field_value is None or field_value == "" or field_value == [] or field_value == {}:
        return dict(EMPTY_FIELD)

    return {
        "value": field_value,
        "answer": str(field_value),
        "evidence": [],
    }


def fix_final_fields(dossier: dict[str, Any], keep_extra_fields: bool) -> tuple[dict[str, Any], list[str], list[str]]:
    original_fields = dossier.get("final_fields", {})

    if not isinstance(original_fields, dict):
        original_fields = {}

    fixed_fields = {}
    added_missing = []

    for field_name in FIELD_ORDER:
        if field_name in original_fields:
            fixed_fields[field_name] = normalize_field_object(original_fields[field_name])
        else:
            fixed_fields[field_name] = dict(EMPTY_FIELD)
            added_missing.append(field_name)

    extra_fields = []

    if keep_extra_fields:
        for field_name, field_value in original_fields.items():
            if field_name not in fixed_fields:
                fixed_fields[field_name] = normalize_field_object(field_value)
                extra_fields.append(field_name)

    dossier["final_fields"] = fixed_fields

    return dossier, added_missing, extra_fields


def fix_dossier_fields(args: argparse.Namespace) -> Path:
    model_folder = resolve_phase3_model_folder(args)
    dossier_path = model_folder / "company_dossier_merged.json"

    if not dossier_path.exists():
        raise FileNotFoundError(f"Missing merged dossier: {dossier_path}")

    dossier = read_json(dossier_path)

    if not isinstance(dossier, dict):
        raise ValueError("company_dossier_merged.json is not a JSON object.")

    original_final_fields = dossier.get("final_fields", {})
    original_count = len(original_final_fields) if isinstance(original_final_fields, dict) else 0

    if args.backup:
        backup_path = dossier_path.with_suffix(".backup_before_field_fix.json")
        write_json(backup_path, dossier)
        print(f"Backup saved: {backup_path}")

    fixed_dossier, added_missing, extra_fields = fix_final_fields(
        dossier=dossier,
        keep_extra_fields=args.keep_extra_fields,
    )

    fixed_dossier["final_fields_count"] = len(fixed_dossier["final_fields"])
    fixed_dossier["final_fields_order_fixed"] = True

    write_json(dossier_path, fixed_dossier)

    print("=" * 80)
    print("DOSSIER FINAL FIELDS FIXED")
    print("=" * 80)
    print(f"Model folder:              {model_folder}")
    print(f"Merged dossier:            {dossier_path}")
    print(f"Original final_fields:     {original_count}")
    print(f"Required fields:           {len(FIELD_ORDER)}")
    print(f"Missing fields added:      {len(added_missing)}")
    print(f"Extra fields kept:         {len(extra_fields)}")
    print("=" * 80)

    if added_missing:
        print("Added missing fields:")
        for field_name in added_missing:
            print(f"  - {field_name}")

    print("=" * 80)
    print("Final field order:")
    for field_name in fixed_dossier["final_fields"].keys():
        value = fixed_dossier["final_fields"][field_name].get("value")
        status = "empty" if value is None or value == "" or value == [] or value == {} else "filled"
        print(f"  - {field_name}: {status}")

    return dossier_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix company_dossier_merged.json so final_fields has all expected fields in prompt order."
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

    parser.add_argument(
        "--keep-extra-fields",
        action="store_true",
        help="Keep any fields that are not in the official FIELD_ORDER list. They will be appended at the end.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fix_dossier_fields(args)


if __name__ == "__main__":
    main()
