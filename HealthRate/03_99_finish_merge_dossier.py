"""
03_99_finish_merge_dossier.py

Standalone helper script to finish ONLY the final company_dossier_merged.json step.

Use this when:
- Phase 3 already finished per-document JSON files
- The script crashed during "Merging company dossier across all documents..."
- You do NOT want to rerun document extraction

Input:
    project_root/
        Phase03_ExplicitFieldsV2/
            <model_folder>/
                json_for_each_file/
                    <doc_id>.json
                    all_field_candidates.json   <-- ignored

Output:
    project_root/
        Phase03_ExplicitFieldsV2/
            <model_folder>/
                company_dossier_merged.json
                merge_debug/
                    loaded_documents_preview.json
                    merge_batches/
                    raw_llm_responses/

Examples:
    python 03_99_finish_merge_dossier.py --provider ollama --model deepseek-v3.2:cloud

    python 03_99_finish_merge_dossier.py --provider ollama --model deepseek-v3.2:cloud --model-folder deepseek-v3.2_cloud

    python 03_99_finish_merge_dossier.py --phase3-model-folder "C:\\Users\\amenx\\Desktop\\HealthRate\\Phase03_ExplicitFieldsV2\\deepseek-v3.2_cloud" --provider ollama --model deepseek-v3.2:cloud

NVIDIA:
    set NVIDIA_API_KEY=your_key_here
    python 03_99_finish_merge_dossier.py --provider nvidia --model deepseek-ai/deepseek-v4-pro
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests


PHASE3_FOLDER_NAME = "Phase03_ExplicitFieldsV2"

FIELD_NAMES = [
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

IMPORTANT_FIELD_NOTES = """
Important field interpretation:
- funding_milestones includes grants, awards, non-dilutive funding, government funding, DOD funding, NIH funding, NINDS funding, NSF funding, SBIR/STTR funding, DARPA funding, prior raises, current raises, and milestone/value inflection points.
- amount_raised_to_date can include investor funding and grant/non-dilutive funding if the documents state it.
- previous_investors can include prior funders, grant providers, or named funding sources when relevant.
- product_status includes clinical stage, technical development stage, lead indications, prototype status, and roadmap status.
- regulatory_exposure includes FDA, clinical, compliance, safety, approval, and regulatory dependency details.
- key_people should only include people with a role/title OR contact info OR clear company importance.
""".strip()


def safe_model_folder_name(model: str) -> str:
    name = model.strip().replace("/", "_").replace(":", "_")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "model"


def ensure_folder(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def write_json(path: Path, data: Any) -> None:
    ensure_folder(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def strip_thinking(text: str) -> str:
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def extract_json_from_text(text: str) -> Any:
    text = strip_thinking(text).strip()

    if not text:
        raise ValueError("LLM returned empty text.")

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return json.loads(text[first:last + 1])

    first = text.find("[")
    last = text.rfind("]")
    if first != -1 and last != -1 and last > first:
        return json.loads(text[first:last + 1])

    raise ValueError("Could not find JSON object or array in LLM response.")


def short_text(value: Any, max_chars: int = 120) -> str:
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def resolve_phase3_model_folder(args: argparse.Namespace) -> Path:
    if args.phase3_model_folder:
        folder = Path(args.phase3_model_folder).expanduser().resolve()
        if not folder.exists():
            raise FileNotFoundError(f"--phase3-model-folder does not exist: {folder}")
        return folder

    project_root = Path.cwd().resolve()
    phase3_root = project_root / PHASE3_FOLDER_NAME

    if not phase3_root.exists():
        raise FileNotFoundError(f"Could not find {phase3_root}")

    if args.model_folder:
        folder = phase3_root / args.model_folder
        if not folder.exists():
            raise FileNotFoundError(f"--model-folder does not exist: {folder}")
        return folder

    expected = phase3_root / safe_model_folder_name(args.model)
    if expected.exists():
        return expected

    subfolders = [p for p in phase3_root.iterdir() if p.is_dir()]
    if len(subfolders) == 1:
        print(f"[auto] Using only model folder found: {subfolders[0]}")
        return subfolders[0]

    raise FileNotFoundError(
        "Could not auto-detect model folder.\n"
        f"Tried: {expected}\n"
        "Use --model-folder or --phase3-model-folder."
    )


def load_document_jsons(model_folder: Path) -> list[dict[str, Any]]:
    json_folder = model_folder / "json_for_each_file"
    if not json_folder.exists():
        raise FileNotFoundError(f"Missing json_for_each_file folder: {json_folder}")

    docs = []
    for path in sorted(json_folder.glob("*.json"), key=lambda p: p.name.lower()):
        if path.name in {"all_field_candidates.json", "company_dossier_merged.json"}:
            continue

        try:
            data = read_json(path)
            if isinstance(data, dict):
                data["_source_json_file"] = str(path)
                docs.append(data)
        except Exception as e:
            print(f"[warning] Could not load {path.name}: {e}")

    return docs


def trim_quote(text: Any, max_chars: int) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[:max_chars] + " ...[trimmed]"
    return text


def prune_evidence_item(item: Any, max_quote_chars: int, default_doc_id: str | None, default_file_name: str | None) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {
            "doc_id": default_doc_id,
            "file_name": default_file_name,
            "quote": trim_quote(item, max_quote_chars),
        }

    result = {
        "doc_id": item.get("doc_id", default_doc_id),
        "file_name": item.get("file_name", default_file_name),
    }

    for key in ["quote", "evidence_quote"]:
        if key in item:
            result["quote"] = trim_quote(item[key], max_quote_chars)
            break

    for key in ["page", "page_start", "page_end", "source_chunk_index", "chunk_index"]:
        if key in item:
            result[key] = item[key]

    if "quote" not in result:
        result["quote"] = trim_quote(item, max_quote_chars)

    return result


def prune_field_obj(
    field_obj: Any,
    max_evidence_per_field: int,
    max_quote_chars: int,
    default_doc_id: str | None,
    default_file_name: str | None,
) -> dict[str, Any]:
    if not isinstance(field_obj, dict):
        return {
            "value": field_obj,
            "answer": None,
            "evidence": [],
        }

    evidence = field_obj.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [evidence]

    return {
        "value": field_obj.get("value"),
        "answer": field_obj.get("answer"),
        "evidence": [
            prune_evidence_item(item, max_quote_chars, default_doc_id, default_file_name)
            for item in evidence[:max_evidence_per_field]
        ],
    }


def prune_person(
    person: Any,
    max_evidence_per_person: int,
    max_quote_chars: int,
    default_doc_id: str | None,
    default_file_name: str | None,
) -> dict[str, Any] | None:
    if not isinstance(person, dict):
        return None

    full_name = person.get("full_name")
    if not full_name:
        return None

    evidence = person.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [evidence]

    return {
        "full_name": full_name,
        "role": person.get("role"),
        "linkedin_profiles": person.get("linkedin_profiles", []),
        "contact_information": person.get("contact_information", []),
        "prior_exits_experience": person.get("prior_exits_experience", []),
        "education": person.get("education", []),
        "document_risk_notes": person.get("document_risk_notes", []),
        "evidence": [
            prune_evidence_item(item, max_quote_chars, default_doc_id, default_file_name)
            for item in evidence[:max_evidence_per_person]
        ],
    }


def prepare_document_for_merge(doc: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    doc_id = doc.get("doc_id")
    file_name = doc.get("file_name")

    final_fields = doc.get("final_fields", {})
    if not isinstance(final_fields, dict):
        final_fields = {}

    clean_fields = {}
    for field_name, field_obj in final_fields.items():
        clean_fields[field_name] = prune_field_obj(
            field_obj,
            max_evidence_per_field=args.max_evidence_per_field,
            max_quote_chars=args.max_quote_chars,
            default_doc_id=doc_id,
            default_file_name=file_name,
        )

    key_people = doc.get("key_people", [])
    if not isinstance(key_people, list):
        key_people = []

    clean_people = []
    for person in key_people:
        clean_person = prune_person(
            person,
            max_evidence_per_person=args.max_evidence_per_person,
            max_quote_chars=args.max_quote_chars,
            default_doc_id=doc_id,
            default_file_name=file_name,
        )
        if clean_person:
            clean_people.append(clean_person)

    return {
        "doc_id": doc_id,
        "file_name": file_name,
        "document_category": doc.get("document_category"),
        "short_summary": doc.get("short_summary"),
        "long_summary": doc.get("long_summary"),
        "source_markdown_file": doc.get("source_markdown_file"),
        "final_fields": clean_fields,
        "key_people": clean_people,
    }


def make_batches(items: list[dict[str, Any]], max_tokens_per_batch: int) -> list[list[dict[str, Any]]]:
    batches = []
    current = []
    current_tokens = 0

    for item in items:
        item_text = json.dumps(item, ensure_ascii=False)
        item_tokens = estimate_tokens(item_text)

        if current and current_tokens + item_tokens > max_tokens_per_batch:
            batches.append(current)
            current = []
            current_tokens = 0

        current.append(item)
        current_tokens += item_tokens

    if current:
        batches.append(current)

    return batches


def call_ollama_chat(
    model: str,
    messages: list[dict[str, str]],
    ollama_url: str,
    temperature: float,
    max_tokens: int | None,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }

    if max_tokens is not None:
        payload["options"]["num_predict"] = max_tokens

    url = ollama_url.rstrip("/") + "/api/chat"
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")


def call_openai_compatible_chat(
    model: str,
    messages: list[dict[str, str]],
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int | None,
    timeout: int,
    extra_body: dict[str, Any] | None = None,
) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise ImportError("Install OpenAI package first: pip install openai") from e

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if extra_body:
        kwargs["extra_body"] = extra_body

    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content or ""


def call_llm(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    ollama_url: str,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int | None,
    timeout: int,
    retries: int,
) -> str:
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            if provider == "ollama":
                return call_ollama_chat(model, messages, ollama_url, temperature, max_tokens, timeout)

            if provider == "nvidia":
                key = api_key or os.getenv("NVIDIA_API_KEY")
                if not key:
                    raise ValueError("NVIDIA_API_KEY is not set.")
                return call_openai_compatible_chat(
                    model=model,
                    messages=messages,
                    base_url=base_url or "https://integrate.api.nvidia.com/v1",
                    api_key=key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    extra_body={"chat_template_kwargs": {"thinking": False}},
                )

            if provider == "openai":
                key = api_key or os.getenv("OPENAI_API_KEY")
                if not key:
                    raise ValueError("OPENAI_API_KEY is not set.")
                return call_openai_compatible_chat(
                    model=model,
                    messages=messages,
                    base_url=base_url or "https://api.openai.com/v1",
                    api_key=key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )

            if provider == "openai_compatible":
                key = api_key or os.getenv("OPENAI_API_KEY")
                if not key:
                    raise ValueError("API key is not set. Use --api-key or set OPENAI_API_KEY.")
                if not base_url:
                    raise ValueError("--base-url is required for provider=openai_compatible.")
                return call_openai_compatible_chat(
                    model=model,
                    messages=messages,
                    base_url=base_url,
                    api_key=key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )

            raise ValueError(f"Unknown provider: {provider}")

        except Exception as e:
            last_error = e
            if attempt < retries:
                print(f"  LLM call failed, retrying in 2s: {e}")
                time.sleep(2)

    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_error}")


def ask_json(prompt: str, args: argparse.Namespace, system_prompt: str, raw_output_path: Path) -> Any:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    raw = call_llm(
        provider=args.provider,
        model=args.model,
        messages=messages,
        ollama_url=args.ollama_url,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_output_tokens,
        timeout=args.timeout,
        retries=args.retries,
    )

    raw_output_path.write_text(raw, encoding="utf-8", errors="replace")

    try:
        return extract_json_from_text(raw)
    except Exception as first_error:
        print(f"  JSON parse failed: {first_error}")
        print("  Asking LLM to repair JSON...")

        repair_prompt = f"""
Fix the following text so it becomes valid JSON.

Rules:
- Return only valid JSON.
- Do not add new facts.
- Do not remove facts unless they are impossible to represent in JSON.
- Fix missing commas, trailing commas, unescaped quotes, broken arrays, broken objects, or invalid syntax.
- No markdown.
- No explanation.

Broken JSON/text:
{raw}
""".strip()

        repair_raw_path = raw_output_path.with_name(raw_output_path.stem + "_repair.txt")

        repair_raw = call_llm(
            provider=args.provider,
            model=args.model,
            messages=[
                {"role": "system", "content": "You repair invalid JSON. Return only valid JSON."},
                {"role": "user", "content": repair_prompt},
            ],
            ollama_url=args.ollama_url,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=0.0,
            max_tokens=args.max_output_tokens,
            timeout=args.timeout,
            retries=args.retries,
        )

        repair_raw_path.write_text(repair_raw, encoding="utf-8", errors="replace")
        return extract_json_from_text(repair_raw)


def field_names_text() -> str:
    return "\n".join(f"- {name}" for name in FIELD_NAMES)


def merge_prompt(items: list[dict[str, Any]], is_final: bool) -> str:
    label = "company dossier" if is_final else "partial company dossier"

    return f"""
You are merging per-document company extraction JSONs into one clean {label}.

Rules:
- Use only the provided JSON data.
- Do not invent facts.
- Prefer the most specific, most recent, and best-supported values.
- If documents disagree, mention uncertainty in the answer.
- Keep evidence for every final field.
- The final_fields object should have one clear answer per field.
- Do not create duplicate candidate lists inside final_fields.
- Each field should be shaped as:
  {{
    "value": "clean final value, list, or object",
    "answer": "clear bullet-style answer based only on evidence",
    "evidence": [
      {{
        "doc_id": "source doc id",
        "file_name": "source file name",
        "quote": "exact quote",
        "page_start": null,
        "page_end": null
      }}
    ]
  }}
- Use null for unknown values.
- Do not force every field.
- Include only fields that are supported by evidence.
- key_people should only include people with a defined role/title OR contact information OR clear company importance.
- Do not include organizations as people.
- Deduplicate people by full name.
- Generate one short_summary and one long_summary.
- Output valid JSON only.
- No markdown.
- No explanation.
- English only.

{IMPORTANT_FIELD_NOTES}

Allowed explicit fields:
{field_names_text()}

Return JSON in exactly this shape:
{{
  "short_summary": "short company summary",
  "long_summary": "comprehensive company summary",
  "final_fields": {{
    "field_name": {{
      "value": "clean final value, list, or object",
      "answer": "- bullet answer grounded in evidence",
      "evidence": [
        {{
          "doc_id": "source doc id",
          "file_name": "source file name",
          "quote": "exact quote",
          "page_start": null,
          "page_end": null
        }}
      ]
    }}
  }},
  "key_people": [
    {{
      "full_name": "Person full name",
      "role": "role/title if stated",
      "linkedin_profiles": [],
      "contact_information": [],
      "prior_exits_experience": [],
      "education": [],
      "document_risk_notes": [],
      "evidence": [
        {{
          "doc_id": "source doc id",
          "file_name": "source file name",
          "quote": "exact quote",
          "page_start": null,
          "page_end": null
        }}
      ]
    }}
  ],
  "documents": [
    {{
      "doc_id": "document id",
      "file_name": "file name",
      "document_category": "category",
      "short_summary": "document summary"
    }}
  ]
}}

Input JSON:
{json.dumps(items, ensure_ascii=False, indent=2)}
""".strip()


def finish_merge(args: argparse.Namespace) -> Path:
    model_folder = resolve_phase3_model_folder(args)

    debug_folder = ensure_folder(model_folder / "merge_debug")
    raw_folder = ensure_folder(debug_folder / "raw_llm_responses")
    batch_folder = ensure_folder(debug_folder / "merge_batches")

    docs_raw = load_document_jsons(model_folder)

    if not docs_raw:
        raise FileNotFoundError(f"No per-document JSON files found in {model_folder / 'json_for_each_file'}")

    docs = [prepare_document_for_merge(doc, args) for doc in docs_raw]
    write_json(debug_folder / "loaded_documents_preview.json", docs)

    print("=" * 80)
    print("FINISH MERGE DOSSIER")
    print("=" * 80)
    print(f"Model folder:        {model_folder}")
    print(f"Provider:            {args.provider}")
    print(f"Model:               {args.model}")
    print(f"Documents loaded:    {len(docs)}")
    print(f"Max batch tokens:    {args.max_batch_tokens}")
    print(f"Max output tokens:   {args.max_output_tokens}")
    print("=" * 80)

    batches = make_batches(docs, args.max_batch_tokens)
    print(f"Merge batches:       {len(batches)}")

    partials = []

    if len(batches) == 1 and not args.force_batch_merge:
        print("[1/1] Running direct final merge...")
        prompt = merge_prompt(batches[0], is_final=True)
        raw_path = raw_folder / "final_merge_raw.txt"
        dossier = ask_json(
            prompt=prompt,
            args=args,
            system_prompt="You are a careful company dossier merger. Return only valid JSON.",
            raw_output_path=raw_path,
        )

    else:
        for i, batch in enumerate(batches, start=1):
            print(
                f"[Batch {i}/{len(batches)}] "
                f"documents={len(batch)} "
                f"estimated_tokens={estimate_tokens(json.dumps(batch, ensure_ascii=False))}"
            )

            prompt = merge_prompt(batch, is_final=False)
            raw_path = raw_folder / f"batch_{i:03d}_raw.txt"

            partial = ask_json(
                prompt=prompt,
                args=args,
                system_prompt="You are a careful partial company dossier merger. Return only valid JSON.",
                raw_output_path=raw_path,
            )

            partial["_batch_index"] = i
            partial["_source_document_count"] = len(batch)

            batch_path = batch_folder / f"batch_{i:03d}.json"
            write_json(batch_path, partial)
            partials.append(partial)

        print("=" * 80)
        print("Running final merge from partial batch dossiers...")
        print("=" * 80)

        prompt = merge_prompt(partials, is_final=True)
        raw_path = raw_folder / "final_merge_raw.txt"

        dossier = ask_json(
            prompt=prompt,
            args=args,
            system_prompt="You are a careful final company dossier merger. Return only valid JSON.",
            raw_output_path=raw_path,
        )

    if not isinstance(dossier, dict):
        raise ValueError("Final dossier was not a JSON object.")

    dossier["model_used"] = args.model
    dossier["provider_used"] = args.provider
    dossier["source_document_count"] = len(docs_raw)
    dossier["merge_script"] = Path(__file__).name

    output_path = model_folder / "company_dossier_merged.json"
    write_json(output_path, dossier)

    print("=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)
    print(f"Saved: {output_path}")
    print(f"Debug folder: {debug_folder}")
    print("=" * 80)

    final_fields = dossier.get("final_fields", {})
    if isinstance(final_fields, dict):
        print("Fields in merged dossier:")
        for key in sorted(final_fields.keys()):
            field_obj = final_fields[key]
            value = field_obj.get("value") if isinstance(field_obj, dict) else field_obj
            print(f"  {key}: {short_text(value)}")

    people = dossier.get("key_people", [])
    if isinstance(people, list) and people:
        print("Key people:")
        for person in people:
            if isinstance(person, dict):
                print(f"  {person.get('full_name')}: {short_text(person.get('role'))}")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finish only the final company_dossier_merged.json merge from existing per-document JSON files."
    )

    parser.add_argument("--phase3-model-folder", type=str, default=None)
    parser.add_argument("--model-folder", type=str, default=None)

    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["ollama", "nvidia", "openai", "openai_compatible"],
    )

    parser.add_argument("--model", type=str, default="deepseek-v3.2:cloud")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)

    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=50000,
        help="Approximate max input tokens per merge batch. Uses characters / 4.",
    )

    parser.add_argument("--max-output-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-evidence-per-field", type=int, default=6)
    parser.add_argument("--max-evidence-per-person", type=int, default=6)
    parser.add_argument("--max-quote-chars", type=int, default=1200)
    parser.add_argument("--force-batch-merge", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    finish_merge(args)


if __name__ == "__main__":
    main()
