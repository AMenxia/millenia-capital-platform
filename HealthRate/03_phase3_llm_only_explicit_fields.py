
"""
03_phase3_llm_only_explicit_fields.py

Phase 3: LLM-centric company extraction.

This version does NOT use:
- Chroma
- vector search
- GLiNER

Input:
    project_root/
        Phase 01_TextExtraction/
            <doc_id>/
                file_llm_feed.md

Output:
    project_root/
        Phase03_ExplicitFieldsV2/
            <model_name>/
                company_dossier_merged.json
                json_for_each_file/
                    <doc_id>.json
                    all_field_candidates.json

Main idea:
1. Read the same Phase 1 markdown feed used by the earlier pipeline.
2. If the document is large, split it into chunks.
3. Send chunks directly to a stronger LLM.
4. Extract explicit company fields, document category, summaries, and key people.
5. Save each document JSON after that document finishes.
6. Save all_field_candidates.json incrementally after each document.
7. Merge all document outputs into company_dossier_merged.json.
8. In the merged dossier, every field gets:
   - value
   - answer
   - evidence

Example Ollama Cloud:
    python 03_phase3_llm_only_explicit_fields.py --model gpt-oss:120b-cloud --timeout 900

Quick test:
    python 03_phase3_llm_only_explicit_fields.py --model gpt-oss:120b-cloud --limit-docs 1 --timeout 900

OpenAI-compatible endpoint:
    python 03_phase3_llm_only_explicit_fields.py --provider openai-compatible --base-url http://localhost:11434/v1 --model qwen3:14b

Notes:
- For Ollama Cloud, sign in and pull the cloud model first:
      ollama signin
      ollama pull gpt-oss:120b-cloud
- The script still talks to your local Ollama server, but cloud models are routed by Ollama.
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


# ----------------------------
# FIXED FOLDERS
# ----------------------------

PHASE1_FOLDER_NAME = "Phase 01_TextExtraction"
PHASE3_FOLDER_NAME = "Phase03_ExplicitFieldsV2"
JSON_FOR_EACH_FILE_FOLDER_NAME = "json_for_each_file"
PHASE1_MARKDOWN_NAME = "file_llm_feed.md"

MERGED_DOSSIER_NAME = "company_dossier_merged.json"
ALL_FIELD_CANDIDATES_NAME = "all_field_candidates.json"


# ----------------------------
# DOCUMENT CATEGORIES
# ----------------------------

DOCUMENT_CATEGORIES = [
    "Pitch deck / investor deck",
    "Investment tear sheet / offering summary",
    "Executive summary",
    "Investor update",
    "Term sheet",
    "Financial model / projections",
    "Cap table / ownership summary",
    "Revenue metrics",
    "Product metrics",
    "Cohort / retention analysis",
    "Churn analysis",
    "CAC / LTV / unit economics analysis",
    "SAFE agreement / legal document",
    "Convertible note agreement / legal document",
    "Subscription agreement / purchase agreement",
    "Customer list",
    "Sales pipeline",
    "Customer contract",
    "Enterprise customer agreement",
    "Key commercial contract",
    "Founder resume / biography",
    "Employee roster",
    "Hiring plan",
    "Organizational chart",
    "Product demo / screenshots",
    "Product roadmap",
    "Technical architecture",
    "Security policy",
    "Privacy policy",
    "Market / TAM analysis",
    "Competitive landscape",
    "Strategic partnership document",
    "Licensing agreement",
    "IP assignment",
    "Patent / IP summary",
    "Corporate charter / certificate of incorporation",
    "Bylaws",
    "Board minutes / board consent",
    "Certificate of good standing",
    "Option plan / equity incentive plan",
    "Bank statement",
    "Tax filing",
    "Insurance policy",
    "Litigation disclosure",
    "Vendor agreement",
    "Millenia engagement agreement",
    "Unknown File Type",
]


# ----------------------------
# FIELD SCHEMA
# ----------------------------

FIELD_DEFINITIONS = {
    # Company identity
    "company_name": "Common company name or brand name.",
    "legal_company_name": "Full legal company name, such as Example, Inc. or Example LLC.",
    "website": "Company website URL or domain.",
    "industry_or_sector": "Industry, sector, vertical, or market category.",
    "company_stage_or_company_idea": "Company stage, concept, or high-level company idea.",
    "headquarters_location": "Headquarters or main operating location.",
    "incorporation_location": "State/country/jurisdiction of incorporation or formation.",
    "year_founded": "Year the company was founded.",
    "legal_entity_structure": "Legal entity type, such as C-Corp, LLC, corporation, limited partnership, etc.",

    # Company explanation
    "one_line_description": "One clear sentence explaining what the company does.",
    "company_problem": "The problem, pain point, or unmet need the company addresses.",
    "company_solution": "The solution, product, service, platform, or technology the company provides.",
    "business_model": "How the company makes or plans to make money.",
    "target_customers": "Customer types, buyers, users, accounts, patient groups, or market segments.",
    "competitive_advantage": "Moat, differentiation, unique advantage, or reason the company can win.",
    "product_status": "Product stage/status, such as concept, prototype, beta, launched, preclinical, IND-enabling, FDA-cleared, etc.",

    # Revenue / financial / operating metrics
    "revenue": "Revenue amount or revenue description, monthly or annual if stated.",
    "mrr": "Monthly recurring revenue.",
    "arr": "Annual recurring revenue.",
    "growth_rate": "Growth rate, including MoM, QoQ, YoY, revenue growth, user growth, or pipeline growth.",
    "gross_margin": "Gross margin or contribution margin.",
    "burn_rate": "Monthly burn or cash burn rate.",
    "runway_months": "Runway in months or years.",
    "cash_on_hand": "Cash balance or cash available.",
    "ltv": "Customer lifetime value.",
    "cac": "Customer acquisition cost.",
    "ltv_cac_ratio": "LTV/CAC ratio.",
    "payback_period": "CAC payback period or payback period.",

    # Fundraising / investment
    "amount_raised_to_date": "Total amount raised so far.",
    "cap_table_summary": "Cap table, ownership, share structure, option pool, or major ownership summary.",
    "previous_investors": "Existing or previous investors.",
    "valuation_last_round": "Valuation from a prior financing round.",
    "target_raise_amount": "Current fundraising target or amount being raised.",
    "minimum_investment": "Minimum investment amount, if listed.",
    "valuation_current_ask": "Current valuation ask if stated.",
    "pre_money_valuation": "Current pre-money valuation.",
    "post_money_valuation": "Current post-money valuation.",
    "current_round": "Current round type, such as Seed, Series A, bridge, SAFE, etc.",
    "instrument": "Investment instrument, such as SAFE, equity, priced round, convertible note, debt, etc.",
    "use_of_funds": "How the company plans to use the funds.",
    "funding_milestones": "Milestones expected to be reached with current or future funding.",

    # Product / customers / traction
    "users_total": "Total users, customers, patients, accounts, downloads, installations, or similar.",
    "users_active": "Active users or active customers.",
    "dau": "Daily active users.",
    "mau": "Monthly active users.",
    "sales_pipeline": "Sales pipeline value, stage, opportunities, accounts, or prospects.",
    "conversion_rates": "Conversion rates in funnel, sales, product, trial, or user conversion.",
    "average_order_value": "Average order value, average contract value, or average selling price.",
    "retention_rates": "Retention rate, renewal rate, repeat usage, or customer retention.",
    "churn_rate": "Churn rate or customer loss rate.",
    "engagement_metrics": "Engagement, usage, activity, frequency, session, or other product usage metrics.",
    "partnerships": "Partnerships, strategic partners, collaborators, channels, distributors, or alliances.",
    "contracts_signed": "Signed contracts, agreements, customer contracts, enterprise agreements, or committed deals.",
    "lois": "Letters of intent, memorandums of understanding, non-binding indications, or LOIs.",
    "key_milestones": "Important company, product, regulatory, technical, commercial, or financing milestones.",
    "key_hires": "Important hires, planned hires, new executives, or important team additions.",

    # Legal / IP / regulatory / risk
    "incorporation_documents": "Corporate charter, certificate of incorporation, bylaws, good standing, or corporate documents.",
    "ip_ownership": "IP ownership, licensing rights, assignments, exclusive licenses, or ownership status.",
    "trademarks_patents": "Patents, patent applications, trademarks, issued patents, pending patents, or patent families.",
    "regulatory_exposure": "Regulatory risk, FDA, SEC, HIPAA, financial regulation, clinical regulation, medical device regulation, etc.",
    "risk_disclosures": "Explicit risks, concerns, dependencies, litigation, compliance risks, or warnings.",
    "market_size": "Market size, TAM, SAM, SOM, market opportunity, market growth, or addressable market.",
}

FIELD_NAMES = list(FIELD_DEFINITIONS.keys())

LIST_FIELDS = {
    "industry_or_sector",
    "target_customers",
    "previous_investors",
    "use_of_funds",
    "funding_milestones",
    "users_total",
    "users_active",
    "sales_pipeline",
    "engagement_metrics",
    "partnerships",
    "contracts_signed",
    "lois",
    "key_milestones",
    "key_hires",
    "incorporation_documents",
    "ip_ownership",
    "trademarks_patents",
    "regulatory_exposure",
    "risk_disclosures",
    "market_size",
    "cap_table_summary",
}


# ----------------------------
# BASIC HELPERS
# ----------------------------

def compact_text(text: Any) -> str:
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate(text: Any, max_chars: int) -> str:
    text = compact_text(text)

    if max_chars is None or max_chars <= 0:
        return text

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "\n...[truncated]"


def safe_json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def safe_text_for_terminal(text: Any, max_chars: int = 120) -> str:
    text = compact_text(text)
    text = text.replace("\n", " ")

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def model_to_folder_name(model: str) -> str:
    cleaned = model.strip()
    cleaned = cleaned.replace(":", "_").replace("/", "_").replace("\\", "_")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_") or "model"


def get_project_root() -> Path:
    return Path.cwd().resolve()


def get_phase1_documents(phase1_folder: Path, markdown_name: str) -> list[dict[str, Any]]:
    docs = []

    if not phase1_folder.exists():
        raise FileNotFoundError(
            "Could not find Phase 1 folder:\n"
            f"{phase1_folder}\n\n"
            "Run this script from the project folder that contains Phase 01_TextExtraction."
        )

    for folder in sorted(phase1_folder.iterdir(), key=lambda p: p.name.lower()):
        if not folder.is_dir():
            continue

        markdown_path = folder / markdown_name

        if not markdown_path.exists():
            continue

        docs.append(
            {
                "doc_id": folder.name,
                "file_name": folder.name,
                "phase1_folder": str(folder),
                "source_markdown_file": str(markdown_path),
            }
        )

    return docs


def is_empty_value(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"", "null", "none", "n/a", "na", "nan", "not stated", "unknown"}

    if isinstance(value, list):
        return len([x for x in value if not is_empty_value(x)]) == 0

    if isinstance(value, dict):
        return len(value) == 0

    return False


def normalize_value_for_field(field_name: str, value: Any) -> Any:
    if value is None:
        return [] if field_name in LIST_FIELDS else None

    if field_name in LIST_FIELDS:
        if isinstance(value, list):
            cleaned = []
            seen = set()

            for item in value:
                item_text = compact_text(item)

                if not item_text:
                    continue

                key = item_text.lower()

                if key in seen:
                    continue

                seen.add(key)
                cleaned.append(item_text)

            return cleaned

        value_text = compact_text(value)

        if not value_text:
            return []

        return [value_text]

    if isinstance(value, list):
        values = [compact_text(x) for x in value if compact_text(x)]

        if not values:
            return None

        return "; ".join(values)

    if isinstance(value, dict):
        if not value:
            return None

        return value

    value_text = compact_text(value)

    if not value_text:
        return None

    return value_text


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Extract the first valid JSON object from a model response.
    Handles:
    - markdown fences
    - small amount of extra text
    - <think>...</think> blocks
    """

    text = text or ""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

        raise ValueError("Top-level JSON was not an object.")

    except Exception:
        pass

    start = text.find("{")

    if start == -1:
        raise ValueError("No JSON object found in model response.")

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        ch = text[index]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1

            if depth == 0:
                candidate = text[start:index + 1]
                parsed = json.loads(candidate)

                if not isinstance(parsed, dict):
                    raise ValueError("Top-level JSON was not an object.")

                return parsed

    raise ValueError("Could not parse JSON object from model response.")


def find_context(text: str, value: Any, evidence_quote: str = "", window: int = 650) -> str:
    text = compact_text(text)
    value_text = compact_text(value)
    evidence_quote = compact_text(evidence_quote)

    search_terms = []

    if evidence_quote:
        search_terms.append(evidence_quote)

    if value_text:
        search_terms.append(value_text)

    for term in search_terms:
        if not term:
            continue

        idx = text.lower().find(term.lower())

        if idx != -1:
            start = max(0, idx - window)
            end = min(len(text), idx + len(term) + window)
            return compact_text(text[start:end])

    return truncate(text, window * 2)


def chunk_markdown(text: str, max_chunk_chars: int, overlap_chars: int) -> list[dict[str, Any]]:
    """
    Split markdown into chunks.

    This tries to keep page markers and headings together. It is character-based
    because it avoids extra tokenizer dependencies.
    """

    text = compact_text(text)

    if len(text) <= max_chunk_chars:
        return [
            {
                "chunk_index": 1,
                "start_char": 0,
                "end_char": len(text),
                "text": text,
            }
        ]

    # Split at page markers or headings when possible.
    pieces = re.split(r"(?=\n---\nPage \d+\n---\n)|(?=\n## )", "\n" + text)

    pieces = [piece.strip() for piece in pieces if piece.strip()]

    chunks = []
    current_parts = []
    current_len = 0
    cursor = 0

    def flush_current():
        nonlocal current_parts, current_len, cursor

        if not current_parts:
            return

        chunk_text = compact_text("\n\n".join(current_parts))
        start_char = max(0, cursor - len(chunk_text))
        end_char = cursor

        chunks.append(
            {
                "chunk_index": len(chunks) + 1,
                "start_char": start_char,
                "end_char": end_char,
                "text": chunk_text,
            }
        )

        if overlap_chars > 0 and len(chunk_text) > overlap_chars:
            overlap_text = chunk_text[-overlap_chars:]
            current_parts = [overlap_text]
            current_len = len(overlap_text)
        else:
            current_parts = []
            current_len = 0

    for piece in pieces:
        piece_len = len(piece)

        # If a single piece is very large, split it directly.
        if piece_len > max_chunk_chars:
            flush_current()

            for start in range(0, piece_len, max_chunk_chars - overlap_chars):
                end = min(piece_len, start + max_chunk_chars)
                part = piece[start:end]
                chunks.append(
                    {
                        "chunk_index": len(chunks) + 1,
                        "start_char": cursor + start,
                        "end_char": cursor + end,
                        "text": compact_text(part),
                    }
                )

                if end >= piece_len:
                    break

            cursor += piece_len
            continue

        if current_len + piece_len + 2 > max_chunk_chars:
            flush_current()

        current_parts.append(piece)
        current_len += piece_len + 2
        cursor += piece_len + 2

    flush_current()

    for index, chunk in enumerate(chunks, start=1):
        chunk["chunk_index"] = index

    return chunks


# ----------------------------
# LLM CLIENT
# ----------------------------

def call_llm(
    messages: list[dict[str, str]],
    provider: str,
    model: str,
    ollama_url: str,
    base_url: str,
    api_key: str | None,
    temperature: float,
    timeout: int,
    max_retries: int = 2,
) -> str:
    """
    Call either Ollama or an OpenAI-compatible chat-completions endpoint.
    """

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            if provider == "ollama":
                url = ollama_url.rstrip("/") + "/api/chat"
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                    },
                }

                response = requests.post(url, json=payload, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "")

            if provider == "openai-compatible":
                url = base_url.rstrip("/")

                if url.endswith("/chat/completions"):
                    endpoint = url
                elif url.endswith("/v1"):
                    endpoint = url + "/chat/completions"
                else:
                    endpoint = url + "/v1/chat/completions"

                headers = {
                    "Content-Type": "application/json",
                }

                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": False,
                }

                response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]

            raise ValueError(f"Unsupported provider: {provider}")

        except Exception as error:
            last_error = error

            if attempt < max_retries:
                wait_seconds = 2 * attempt
                print(f"    LLM call failed, retrying in {wait_seconds}s: {error}")
                time.sleep(wait_seconds)

    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")


def json_llm_call(
    messages: list[dict[str, str]],
    provider: str,
    model: str,
    ollama_url: str,
    base_url: str,
    api_key: str | None,
    temperature: float,
    timeout: int,
    max_retries: int = 2,
) -> dict[str, Any]:
    text = call_llm(
        messages=messages,
        provider=provider,
        model=model,
        ollama_url=ollama_url,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )

    return extract_json_object(text)


def system_prompt() -> str:
    return (
        "You are a careful information extraction engine. "
        "Respond in English only. "
        "Return only valid JSON. "
        "Do not include markdown fences. "
        "Do not include reasoning. "
        "Do not invent facts. "
        "Use null or empty arrays when the documents do not clearly state something."
    )


# ----------------------------
# EXTRACTION PROMPTS
# ----------------------------

def field_definitions_text() -> str:
    lines = []

    for name, description in FIELD_DEFINITIONS.items():
        lines.append(f"- {name}: {description}")

    return "\n".join(lines)


def build_chunk_extraction_prompt(doc: dict[str, Any], chunk: dict[str, Any]) -> str:
    categories = "\n".join(f"- {category}" for category in DOCUMENT_CATEGORIES)

    prompt = f"""
You are extracting structured company information from one markdown document chunk.

Document:
- doc_id: {doc["doc_id"]}
- file_name: {doc["file_name"]}
- chunk_index: {chunk["chunk_index"]}

Allowed document categories:
{categories}

Allowed explicit fields:
{field_definitions_text()}

Task:
Extract only information explicitly supported by this chunk.

Return JSON with this exact shape:
{{
  "document_category_candidate": "one category from the allowed list or Unknown File Type",
  "chunk_summary": "brief factual summary of this chunk",
  "field_candidates": [
    {{
      "field_name": "one allowed field name",
      "value": "string, list, number, or null",
      "evidence_quote": "short exact quote from the chunk that supports the value"
    }}
  ],
  "key_people": [
    {{
      "full_name": "full human name",
      "role": "role/title if stated, otherwise null",
      "linkedin_profiles": ["only LinkedIn URLs explicitly stated"],
      "contact_information": ["email, phone, website profile, or other contact info explicitly stated"],
      "prior_exits_experience": ["experience explicitly stated"],
      "education": ["education explicitly stated"],
      "document_risk_notes": ["risk/background notes explicitly stated"],
      "evidence_quote": "short exact quote from the chunk that supports this person"
    }}
  ]
}}

Rules:
1. Use only the chunk text.
2. Do not guess.
3. Do not force-fill fields.
4. If no field candidates exist, return an empty field_candidates array.
5. If no key people exist, return an empty key_people array.
6. Only include real human people in key_people.
7. Do not include companies, universities, investors, legal parties, or groups as people.
8. For key_people, prefer people with a defined role, title, contact information, education, or relevant experience.
9. For evidence_quote, copy a short exact phrase from the chunk.
10. Return only JSON.

Chunk text:
{chunk["text"]}
""".strip()

    return prompt


def normalize_field_candidate(
    raw_candidate: Any,
    doc: dict[str, Any],
    chunk: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any] | None:
    if not isinstance(raw_candidate, dict):
        return None

    field_name = compact_text(raw_candidate.get("field_name"))

    if field_name not in FIELD_NAMES:
        return None

    value = normalize_value_for_field(field_name, raw_candidate.get("value"))

    if is_empty_value(value):
        return None

    evidence_quote = compact_text(raw_candidate.get("evidence_quote"))

    if not evidence_quote:
        evidence_quote = find_context(chunk["text"], value, "", window=180)

    surrounding_context = find_context(chunk["text"], value, evidence_quote, window=650)

    return {
        "candidate_id": candidate_id,
        "field_name": field_name,
        "value": value,
        "evidence_quote": evidence_quote,
        "surrounding_context": surrounding_context,
        "source": {
            "doc_id": doc["doc_id"],
            "file_name": doc["file_name"],
            "source_markdown_file": doc["source_markdown_file"],
            "phase1_folder": doc["phase1_folder"],
            "chunk_index": chunk["chunk_index"],
            "start_char": chunk["start_char"],
            "end_char": chunk["end_char"],
        },
    }


def looks_like_person_record(person: dict[str, Any]) -> bool:
    name = compact_text(person.get("full_name"))

    if not name:
        return False

    lowered = name.lower()

    banned_terms = [
        "company",
        "investor",
        "investors",
        "customer",
        "customers",
        "team",
        "management",
        "board",
        "advisors",
        "personnel",
        "representing",
        "indemnified",
        "party",
        "parties",
        "university",
        "inc",
        "llc",
        "corp",
        "corporation",
    ]

    for term in banned_terms:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            return False

    if len(name.split()) < 2:
        return False

    role = compact_text(person.get("role"))
    contact = person.get("contact_information") or []
    linkedin = person.get("linkedin_profiles") or []

    if role or contact or linkedin:
        return True

    # Still allow people with meaningful education or experience at the candidate stage.
    experience = person.get("prior_exits_experience") or []
    education = person.get("education") or []

    return bool(experience or education)


def normalize_people_candidate(
    raw_person: Any,
    doc: dict[str, Any],
    chunk: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(raw_person, dict):
        return None

    person = {
        "full_name": compact_text(raw_person.get("full_name")),
        "role": compact_text(raw_person.get("role")) or None,
        "linkedin_profiles": [compact_text(x) for x in raw_person.get("linkedin_profiles", []) if compact_text(x)],
        "contact_information": [compact_text(x) for x in raw_person.get("contact_information", []) if compact_text(x)],
        "prior_exits_experience": [compact_text(x) for x in raw_person.get("prior_exits_experience", []) if compact_text(x)],
        "education": [compact_text(x) for x in raw_person.get("education", []) if compact_text(x)],
        "document_risk_notes": [compact_text(x) for x in raw_person.get("document_risk_notes", []) if compact_text(x)],
        "evidence_quote": compact_text(raw_person.get("evidence_quote")),
    }

    if not looks_like_person_record(person):
        return None

    if not person["evidence_quote"]:
        person["evidence_quote"] = find_context(chunk["text"], person["full_name"], "", window=180)

    person["surrounding_context"] = find_context(
        chunk["text"],
        person["full_name"],
        person["evidence_quote"],
        window=650,
    )

    person["source"] = {
        "doc_id": doc["doc_id"],
        "file_name": doc["file_name"],
        "source_markdown_file": doc["source_markdown_file"],
        "phase1_folder": doc["phase1_folder"],
        "chunk_index": chunk["chunk_index"],
        "start_char": chunk["start_char"],
        "end_char": chunk["end_char"],
    }

    return person


def terminal_found_fields(candidates: list[dict[str, Any]], max_items: int = 5) -> str:
    parts = []

    for candidate in candidates[:max_items]:
        field = candidate.get("field_name")
        value = candidate.get("value")
        parts.append(f'{field}="{safe_text_for_terminal(value, 80)}"')

    if len(candidates) > max_items:
        parts.append(f"... +{len(candidates) - max_items} more")

    return ", ".join(parts)


def terminal_found_people(people: list[dict[str, Any]], max_items: int = 6) -> str:
    names = []

    for person in people[:max_items]:
        names.append(person.get("full_name", ""))

    if len(people) > max_items:
        names.append(f"... +{len(people) - max_items} more")

    return "; ".join(names)


# ----------------------------
# DOCUMENT MERGE
# ----------------------------

def compact_candidates_for_prompt(candidates: list[dict[str, Any]], max_chars: int) -> str:
    compact = []

    for candidate in candidates:
        source = candidate.get("source", {})

        compact.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "field_name": candidate.get("field_name"),
                "value": candidate.get("value"),
                "evidence_quote": candidate.get("evidence_quote"),
                "surrounding_context": candidate.get("surrounding_context"),
                "doc_id": source.get("doc_id"),
                "file_name": source.get("file_name"),
                "chunk_index": source.get("chunk_index"),
            }
        )

    return truncate(json.dumps(compact, indent=2, ensure_ascii=False), max_chars)


def compact_people_for_prompt(people: list[dict[str, Any]], max_chars: int) -> str:
    compact = []

    for person in people:
        source = person.get("source", {})

        compact.append(
            {
                "full_name": person.get("full_name"),
                "role": person.get("role"),
                "linkedin_profiles": person.get("linkedin_profiles", []),
                "contact_information": person.get("contact_information", []),
                "prior_exits_experience": person.get("prior_exits_experience", []),
                "education": person.get("education", []),
                "document_risk_notes": person.get("document_risk_notes", []),
                "evidence_quote": person.get("evidence_quote"),
                "surrounding_context": person.get("surrounding_context"),
                "doc_id": source.get("doc_id"),
                "file_name": source.get("file_name"),
                "chunk_index": source.get("chunk_index"),
            }
        )

    return truncate(json.dumps(compact, indent=2, ensure_ascii=False), max_chars)


def empty_field_object(field_name: str) -> dict[str, Any]:
    return {
        "value": [] if field_name in LIST_FIELDS else None,
        "answer": "The documents do not clearly state this.",
        "evidence": [],
    }


def candidate_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
    source = candidate.get("source", {})

    return {
        "quote": candidate.get("evidence_quote"),
        "surrounding_context": candidate.get("surrounding_context"),
        "doc_id": source.get("doc_id"),
        "file_name": source.get("file_name"),
        "chunk_index": source.get("chunk_index"),
        "start_char": source.get("start_char"),
        "end_char": source.get("end_char"),
        "source_markdown_file": source.get("source_markdown_file"),
    }


def build_fields_from_selected_ids(
    selected_fields: dict[str, Any],
    candidates_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    final_fields = {}

    for field_name in FIELD_NAMES:
        raw = selected_fields.get(field_name)

        if not isinstance(raw, dict):
            final_fields[field_name] = empty_field_object(field_name)
            continue

        selected_ids = raw.get("selected_candidate_ids", [])

        if not isinstance(selected_ids, list):
            selected_ids = []

        selected_candidates = [
            candidates_by_id[candidate_id]
            for candidate_id in selected_ids
            if candidate_id in candidates_by_id
        ]

        value = normalize_value_for_field(field_name, raw.get("value"))
        answer = compact_text(raw.get("answer"))

        if is_empty_value(value):
            value = [] if field_name in LIST_FIELDS else None

        if not answer:
            answer = "The documents do not clearly state this." if is_empty_value(value) else compact_text(value)

        final_fields[field_name] = {
            "value": value,
            "answer": answer,
            "evidence": [candidate_evidence(candidate) for candidate in selected_candidates],
        }

    return final_fields


def build_doc_merge_prompt(
    doc: dict[str, Any],
    chunk_summaries: list[str],
    document_category_candidates: list[str],
    field_candidates: list[dict[str, Any]],
    people_candidates: list[dict[str, Any]],
    max_prompt_chars: int,
) -> str:
    candidates_text = compact_candidates_for_prompt(field_candidates, max_prompt_chars // 2)
    people_text = compact_people_for_prompt(people_candidates, max_prompt_chars // 3)

    chunk_summaries_text = truncate(
        json.dumps(chunk_summaries, indent=2, ensure_ascii=False),
        max_prompt_chars // 6,
    )

    category_candidates_text = truncate(
        json.dumps(document_category_candidates, indent=2, ensure_ascii=False),
        4000,
    )

    field_template = {
        field_name: {
            "value": [] if field_name in LIST_FIELDS else None,
            "answer": "The documents do not clearly state this.",
            "selected_candidate_ids": [],
        }
        for field_name in FIELD_NAMES
    }

    prompt = f"""
You are merging extracted candidates for one document.

Document:
{json.dumps(doc, indent=2, ensure_ascii=False)}

Allowed document categories:
{json.dumps(DOCUMENT_CATEGORIES, indent=2, ensure_ascii=False)}

Document category candidates from chunks:
{category_candidates_text}

Chunk summaries:
{chunk_summaries_text}

Field candidates:
{candidates_text}

People candidates:
{people_text}

Task:
Create one clean JSON result for this document.

Return JSON with this exact shape:
{{
  "document_category": "one allowed category",
  "short_summary": "short factual summary",
  "long_summary": "longer factual summary",
  "final_fields": {json.dumps(field_template, indent=2, ensure_ascii=False)},
  "key_people": [
    {{
      "full_name": "clean full name",
      "role": "role/title if stated, otherwise null",
      "linkedin_profiles": [],
      "contact_information": [],
      "prior_exits_experience": [],
      "education": [],
      "document_risk_notes": [],
      "evidence": [
        {{
          "quote": "supporting quote",
          "surrounding_context": "nearby context",
          "doc_id": "{doc["doc_id"]}",
          "chunk_index": 1
        }}
      ]
    }}
  ]
}}

Rules:
1. Use only the provided candidates.
2. Do not invent facts.
3. For each explicit field, use selected_candidate_ids to point to candidate IDs.
4. For single-value fields, select at most one candidate ID.
5. For list fields, select all useful candidate IDs.
6. Each field must include both value and answer.
7. The answer can be longer when useful, but it must be based only on evidence.
8. For list fields, write answer as bullet-style text inside a string.
9. If a field is unclear, use null or [] and answer "The documents do not clearly state this."
10. key_people should only include people who have a defined role/title OR contact information OR LinkedIn/profile/contact medium.
11. Remove duplicate people.
12. Return only JSON.
""".strip()

    return prompt


def merge_one_document(
    doc: dict[str, Any],
    chunk_summaries: list[str],
    document_category_candidates: list[str],
    field_candidates: list[dict[str, Any]],
    people_candidates: list[dict[str, Any]],
    llm_args: dict[str, Any],
    max_prompt_chars: int,
) -> dict[str, Any]:
    candidates_by_id = {
        candidate["candidate_id"]: candidate
        for candidate in field_candidates
    }

    prompt = build_doc_merge_prompt(
        doc=doc,
        chunk_summaries=chunk_summaries,
        document_category_candidates=document_category_candidates,
        field_candidates=field_candidates,
        people_candidates=people_candidates,
        max_prompt_chars=max_prompt_chars,
    )

    response = json_llm_call(
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ],
        **llm_args,
    )

    selected_fields = response.get("final_fields", {})

    if not isinstance(selected_fields, dict):
        selected_fields = {}

    final_fields = build_fields_from_selected_ids(selected_fields, candidates_by_id)

    key_people = response.get("key_people", [])

    if not isinstance(key_people, list):
        key_people = []

    key_people = [person for person in key_people if isinstance(person, dict)]

    document_category = compact_text(response.get("document_category")) or "Unknown File Type"

    if document_category not in DOCUMENT_CATEGORIES:
        document_category = "Unknown File Type"

    return {
        "doc_id": doc["doc_id"],
        "file_name": doc["file_name"],
        "source_markdown_file": doc["source_markdown_file"],
        "phase1_folder": doc["phase1_folder"],
        "document_category": document_category,
        "short_summary": compact_text(response.get("short_summary")),
        "long_summary": compact_text(response.get("long_summary")),
        "final_fields": final_fields,
        "key_people": key_people,
    }


# ----------------------------
# GLOBAL DOSSIER MERGE
# ----------------------------

def select_merged_field(
    field_name: str,
    candidates: list[dict[str, Any]],
    llm_args: dict[str, Any],
    max_prompt_chars: int,
) -> dict[str, Any]:
    if not candidates:
        return empty_field_object(field_name)

    max_rule = "select at most ONE candidate_id" if field_name not in LIST_FIELDS else "select all useful candidate_ids"

    prompt = f"""
You are creating the final merged company dossier.

Field to merge:
- {field_name}: {FIELD_DEFINITIONS[field_name]}

Candidates for this field across all documents:
{compact_candidates_for_prompt(candidates, max_prompt_chars)}

Task:
Select the best value for the final dossier and generate a clear answer.

Return JSON with this exact shape:
{{
  "value": {"[]" if field_name in LIST_FIELDS else "null"},
  "answer": "The documents do not clearly state this.",
  "selected_candidate_ids": []
}}

Rules:
1. Use only the provided candidates.
2. Do not invent facts.
3. {max_rule}.
4. Compare candidate value + evidence_quote + surrounding_context + source document.
5. Prefer candidates with specific supporting context.
6. If values conflict, prefer the one most explicitly supported by the evidence.
7. For list fields, answer should be bullet-style text inside a string.
8. If unclear, return null or [] and answer "The documents do not clearly state this."
9. Return only JSON.
""".strip()

    response = json_llm_call(
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ],
        **llm_args,
    )

    selected_ids = response.get("selected_candidate_ids", [])

    if not isinstance(selected_ids, list):
        selected_ids = []

    candidates_by_id = {
        candidate["candidate_id"]: candidate
        for candidate in candidates
    }

    selected_candidates = [
        candidates_by_id[candidate_id]
        for candidate_id in selected_ids
        if candidate_id in candidates_by_id
    ]

    value = normalize_value_for_field(field_name, response.get("value"))
    answer = compact_text(response.get("answer"))

    if is_empty_value(value):
        value = [] if field_name in LIST_FIELDS else None

    if not answer:
        answer = "The documents do not clearly state this." if is_empty_value(value) else compact_text(value)

    return {
        "value": value,
        "answer": answer,
        "evidence": [candidate_evidence(candidate) for candidate in selected_candidates],
    }


def merge_key_people_globally(
    per_doc_outputs: list[dict[str, Any]],
    llm_args: dict[str, Any],
    max_prompt_chars: int,
) -> list[dict[str, Any]]:
    people = []

    for doc_output in per_doc_outputs:
        for person in doc_output.get("key_people", []):
            if not isinstance(person, dict):
                continue

            person_copy = dict(person)
            person_copy["source_doc_id"] = doc_output.get("doc_id")
            person_copy["source_file_name"] = doc_output.get("file_name")
            people.append(person_copy)

    if not people:
        return []

    people_text = truncate(json.dumps(people, indent=2, ensure_ascii=False), max_prompt_chars)

    prompt = f"""
You are merging key people across multiple company documents.

People records:
{people_text}

Task:
Create one clean deduplicated key_people list for the final company dossier.

Return JSON with this exact shape:
{{
  "key_people": [
    {{
      "full_name": "clean full name",
      "role": "role/title if stated, otherwise null",
      "linkedin_profiles": [],
      "contact_information": [],
      "prior_exits_experience": [],
      "education": [],
      "document_risk_notes": [],
      "evidence": []
    }}
  ]
}}

Rules:
1. Use only the provided people records.
2. Merge duplicate names.
3. Keep the most complete full name.
4. Only include people who have a defined role/title OR contact information OR LinkedIn/profile/contact medium.
5. Do not include companies, investors, universities, legal parties, or generic groups as people.
6. Preserve useful evidence from the source records.
7. Return only JSON.
""".strip()

    response = json_llm_call(
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ],
        **llm_args,
    )

    key_people = response.get("key_people", [])

    if not isinstance(key_people, list):
        return []

    clean = []

    for person in key_people:
        if not isinstance(person, dict):
            continue

        role = compact_text(person.get("role"))
        contacts = person.get("contact_information") or []
        linkedin = person.get("linkedin_profiles") or []

        if not role and not contacts and not linkedin:
            continue

        clean.append(person)

    return clean


def generate_merged_summaries(
    per_doc_outputs: list[dict[str, Any]],
    merged_fields: dict[str, Any],
    merged_people: list[dict[str, Any]],
    llm_args: dict[str, Any],
    max_prompt_chars: int,
) -> dict[str, str]:
    summary_input = {
        "document_summaries": [
            {
                "doc_id": doc.get("doc_id"),
                "document_category": doc.get("document_category"),
                "short_summary": doc.get("short_summary"),
                "long_summary": doc.get("long_summary"),
            }
            for doc in per_doc_outputs
        ],
        "merged_fields": {
            field_name: {
                "value": field_obj.get("value"),
                "answer": field_obj.get("answer"),
            }
            for field_name, field_obj in merged_fields.items()
        },
        "key_people": [
            {
                "full_name": person.get("full_name"),
                "role": person.get("role"),
            }
            for person in merged_people
        ],
    }

    prompt = f"""
You are writing the final summary for a merged company dossier.

Input:
{truncate(json.dumps(summary_input, indent=2, ensure_ascii=False), max_prompt_chars)}

Return JSON with this exact shape:
{{
  "short_summary": "one short paragraph",
  "long_summary": "more comprehensive multi-paragraph summary"
}}

Rules:
1. Use only the provided input.
2. Do not invent facts.
3. Mention uncertainty only when useful.
4. Return only JSON.
""".strip()

    response = json_llm_call(
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ],
        **llm_args,
    )

    return {
        "short_summary": compact_text(response.get("short_summary")),
        "long_summary": compact_text(response.get("long_summary")),
    }


# ----------------------------
# MAIN PROCESS
# ----------------------------

def process_document(
    doc: dict[str, Any],
    output_doc_path: Path,
    all_field_candidates: list[dict[str, Any]],
    llm_args: dict[str, Any],
    max_chunk_chars: int,
    chunk_overlap_chars: int,
    max_doc_merge_prompt_chars: int,
    terminal_value_chars: int,
) -> dict[str, Any]:
    markdown_path = Path(doc["source_markdown_file"])
    text = markdown_path.read_text(encoding="utf-8", errors="replace")

    chunks = chunk_markdown(
        text=text,
        max_chunk_chars=max_chunk_chars,
        overlap_chars=chunk_overlap_chars,
    )

    print(f"  chunks: {len(chunks)}")

    doc_field_candidates = []
    doc_people_candidates = []
    chunk_summaries = []
    category_candidates = []

    for chunk in chunks:
        chunk_index = chunk["chunk_index"]
        print(f"  [chunk {chunk_index}/{len(chunks)}] chunk_index={chunk_index}")

        prompt = build_chunk_extraction_prompt(doc=doc, chunk=chunk)

        try:
            response = json_llm_call(
                messages=[
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                **llm_args,
            )
        except Exception as error:
            print(f"    extraction failed: {error}")
            continue

        category_candidate = compact_text(response.get("document_category_candidate"))

        if category_candidate:
            category_candidates.append(category_candidate)

        chunk_summary = compact_text(response.get("chunk_summary"))

        if chunk_summary:
            chunk_summaries.append(chunk_summary)

        raw_field_candidates = response.get("field_candidates", [])

        if not isinstance(raw_field_candidates, list):
            raw_field_candidates = []

        chunk_field_candidates = []

        for raw_candidate in raw_field_candidates:
            candidate_id = f"field_cand_{len(all_field_candidates) + len(doc_field_candidates) + 1:06d}"

            candidate = normalize_field_candidate(
                raw_candidate=raw_candidate,
                doc=doc,
                chunk=chunk,
                candidate_id=candidate_id,
            )

            if candidate is None:
                continue

            chunk_field_candidates.append(candidate)
            doc_field_candidates.append(candidate)

        raw_people = response.get("key_people", [])

        if not isinstance(raw_people, list):
            raw_people = []

        chunk_people = []

        for raw_person in raw_people:
            person = normalize_people_candidate(
                raw_person=raw_person,
                doc=doc,
                chunk=chunk,
            )

            if person is None:
                continue

            chunk_people.append(person)
            doc_people_candidates.append(person)

        if chunk_field_candidates:
            print(f"    found: {terminal_found_fields(chunk_field_candidates, max_items=6)}")

        if chunk_people:
            print(f"    people: {terminal_found_people(chunk_people)}")

    print("  merging document candidates...")

    doc_output = merge_one_document(
        doc=doc,
        chunk_summaries=chunk_summaries,
        document_category_candidates=category_candidates,
        field_candidates=doc_field_candidates,
        people_candidates=doc_people_candidates,
        llm_args=llm_args,
        max_prompt_chars=max_doc_merge_prompt_chars,
    )

    safe_json_write(output_doc_path, doc_output)

    print(f"  wrote: {output_doc_path}")
    print(f"  field candidates found: {len(doc_field_candidates)}")
    print(f"  clean key people: {len(doc_output.get('key_people', []))}")

    return {
        "doc_output": doc_output,
        "field_candidates": doc_field_candidates,
    }


def run_phase3(
    provider: str,
    model: str,
    ollama_url: str,
    base_url: str,
    api_key: str | None,
    temperature: float,
    timeout: int,
    markdown_name: str,
    limit_docs: int | None,
    max_chunk_chars: int,
    chunk_overlap_chars: int,
    max_doc_merge_prompt_chars: int,
    max_global_field_prompt_chars: int,
    max_global_people_prompt_chars: int,
    max_summary_prompt_chars: int,
    terminal_value_chars: int,
) -> Path:
    project_root = get_project_root()
    phase1_folder = project_root / PHASE1_FOLDER_NAME

    model_folder_name = model_to_folder_name(model)
    model_output_folder = project_root / PHASE3_FOLDER_NAME / model_folder_name
    json_folder = model_output_folder / JSON_FOR_EACH_FILE_FOLDER_NAME

    merged_dossier_path = model_output_folder / MERGED_DOSSIER_NAME
    all_field_candidates_path = json_folder / ALL_FIELD_CANDIDATES_NAME

    docs = get_phase1_documents(
        phase1_folder=phase1_folder,
        markdown_name=markdown_name,
    )

    if limit_docs is not None:
        docs = docs[:limit_docs]

    llm_args = {
        "provider": provider,
        "model": model,
        "ollama_url": ollama_url,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": temperature,
        "timeout": timeout,
    }

    print("=" * 80)
    print("PHASE 3 LLM-ONLY COMPANY EXTRACTION START")
    print("=" * 80)
    print(f"Working folder:        {project_root}")
    print(f"Phase 1 folder:        {phase1_folder}")
    print(f"Markdown file:         {markdown_name}")
    print(f"Provider:              {provider}")
    print(f"Model:                 {model}")
    print(f"Output folder:         {model_output_folder}")
    print(f"Documents to process:  {len(docs)}")
    print(f"No Chroma:             True")
    print(f"No GLiNER:             True")
    print("=" * 80)

    json_folder.mkdir(parents=True, exist_ok=True)

    all_field_candidates = []
    per_doc_outputs = []

    for doc_index, doc in enumerate(docs, start=1):
        print(f"[Document {doc_index}/{len(docs)}] {doc['doc_id']}")

        output_doc_path = json_folder / f"{doc['doc_id']}.json"

        try:
            result = process_document(
                doc=doc,
                output_doc_path=output_doc_path,
                all_field_candidates=all_field_candidates,
                llm_args=llm_args,
                max_chunk_chars=max_chunk_chars,
                chunk_overlap_chars=chunk_overlap_chars,
                max_doc_merge_prompt_chars=max_doc_merge_prompt_chars,
                terminal_value_chars=terminal_value_chars,
            )

            per_doc_outputs.append(result["doc_output"])
            all_field_candidates.extend(result["field_candidates"])

            all_candidates_payload = {
                "provider": provider,
                "model": model,
                "markdown_name": markdown_name,
                "candidate_count": len(all_field_candidates),
                "candidates": all_field_candidates,
            }

            safe_json_write(all_field_candidates_path, all_candidates_payload)

            print(f"  updated field candidates JSON: {all_field_candidates_path}")

        except Exception as error:
            print(f"  Status: error")
            print(f"  Reason: {error}")

    print("=" * 80)
    print("All documents scanned. Merging dossier fields across all documents...")
    print("=" * 80)

    candidates_by_field = {field_name: [] for field_name in FIELD_NAMES}

    for candidate in all_field_candidates:
        field_name = candidate.get("field_name")

        if field_name in candidates_by_field:
            candidates_by_field[field_name].append(candidate)

    merged_fields = {}

    for field_index, field_name in enumerate(FIELD_NAMES, start=1):
        candidates = candidates_by_field.get(field_name, [])

        print(f"[Dossier field {field_index}/{len(FIELD_NAMES)}] {field_name} candidates={len(candidates)}")

        try:
            merged_fields[field_name] = select_merged_field(
                field_name=field_name,
                candidates=candidates,
                llm_args=llm_args,
                max_prompt_chars=max_global_field_prompt_chars,
            )
        except Exception as error:
            print(f"  merge failed: {error}")
            merged_fields[field_name] = empty_field_object(field_name)

    print("=" * 80)
    print("Merging key people across documents...")
    print("=" * 80)

    try:
        merged_people = merge_key_people_globally(
            per_doc_outputs=per_doc_outputs,
            llm_args=llm_args,
            max_prompt_chars=max_global_people_prompt_chars,
        )
    except Exception as error:
        print(f"  people merge failed: {error}")
        merged_people = []

    print("=" * 80)
    print("Generating merged dossier summary...")
    print("=" * 80)

    try:
        merged_summaries = generate_merged_summaries(
            per_doc_outputs=per_doc_outputs,
            merged_fields=merged_fields,
            merged_people=merged_people,
            llm_args=llm_args,
            max_prompt_chars=max_summary_prompt_chars,
        )
    except Exception as error:
        print(f"  summary generation failed: {error}")
        merged_summaries = {
            "short_summary": "",
            "long_summary": "",
        }

    merged_dossier = {
        "provider": provider,
        "model": model,
        "markdown_name": markdown_name,
        "short_summary": merged_summaries.get("short_summary", ""),
        "long_summary": merged_summaries.get("long_summary", ""),
        "final_fields": merged_fields,
        "key_people": merged_people,
        "documents": [
            {
                "doc_id": doc.get("doc_id"),
                "file_name": doc.get("file_name"),
                "document_category": doc.get("document_category"),
                "short_summary": doc.get("short_summary"),
                "source_markdown_file": doc.get("source_markdown_file"),
            }
            for doc in per_doc_outputs
        ],
        "candidate_files": {
            "all_field_candidates": str(all_field_candidates_path),
        },
    }

    safe_json_write(merged_dossier_path, merged_dossier)

    print("=" * 80)
    print("PHASE 3 LLM-ONLY COMPANY EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Documents completed:   {len(per_doc_outputs)}")
    print(f"Field candidates:      {len(all_field_candidates)}")
    print(f"Merged key people:     {len(merged_people)}")
    print(f"Merged dossier:        {merged_dossier_path}")
    print(f"Per-file JSON folder:  {json_folder}")
    print("=" * 80)

    return model_output_folder


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 LLM-only company extraction from Phase 1 markdown files."
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["ollama", "openai-compatible"],
        help="LLM provider. Use ollama for local/cloud Ollama models.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss:120b-cloud",
        help="Model name. Example: gpt-oss:120b-cloud, qwen3:14b, qwen3:32b.",
    )

    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL. Used when --provider ollama.",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434/v1",
        help="OpenAI-compatible base URL. Used when --provider openai-compatible.",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for OpenAI-compatible provider. If omitted, uses environment variable selected by --api-key-env.",
    )

    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable to read API key from for OpenAI-compatible provider.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Request timeout in seconds.",
    )

    parser.add_argument(
        "--markdown-name",
        type=str,
        default=PHASE1_MARKDOWN_NAME,
        help="Markdown file name inside each Phase 1 document folder.",
    )

    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Optional number of documents to process for testing.",
    )

    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=18000,
        help="Maximum characters per markdown chunk.",
    )

    parser.add_argument(
        "--chunk-overlap-chars",
        type=int,
        default=1200,
        help="Character overlap between chunks.",
    )

    parser.add_argument(
        "--max-doc-merge-prompt-chars",
        type=int,
        default=90000,
        help="Max characters used for per-document merge prompt.",
    )

    parser.add_argument(
        "--max-global-field-prompt-chars",
        type=int,
        default=90000,
        help="Max characters used when merging each field across documents.",
    )

    parser.add_argument(
        "--max-global-people-prompt-chars",
        type=int,
        default=90000,
        help="Max characters used when merging key people across documents.",
    )

    parser.add_argument(
        "--max-summary-prompt-chars",
        type=int,
        default=90000,
        help="Max characters used when generating the merged dossier summary.",
    )

    parser.add_argument(
        "--terminal-value-chars",
        type=int,
        default=120,
        help="Max value characters shown in terminal logs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = args.api_key

    if not api_key and args.provider == "openai-compatible":
        api_key = os.environ.get(args.api_key_env)

    run_phase3(
        provider=args.provider,
        model=args.model,
        ollama_url=args.ollama_url,
        base_url=args.base_url,
        api_key=api_key,
        temperature=args.temperature,
        timeout=args.timeout,
        markdown_name=args.markdown_name,
        limit_docs=args.limit_docs,
        max_chunk_chars=args.max_chunk_chars,
        chunk_overlap_chars=args.chunk_overlap_chars,
        max_doc_merge_prompt_chars=args.max_doc_merge_prompt_chars,
        max_global_field_prompt_chars=args.max_global_field_prompt_chars,
        max_global_people_prompt_chars=args.max_global_people_prompt_chars,
        max_summary_prompt_chars=args.max_summary_prompt_chars,
        terminal_value_chars=args.terminal_value_chars,
    )


if __name__ == "__main__":
    main()
