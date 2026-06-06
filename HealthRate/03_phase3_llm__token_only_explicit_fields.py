"""
03_phase3_llm_only_explicit_fields.py

Phase 3: LLM-only explicit field extraction from Phase 1 markdown files.

This version:
- Uses file_llm_feed.md
- Does NOT use Chroma
- Does NOT use GLiNER
- Splits each document by approximate token count
- Preserves page boundaries
- Runs one all-fields extraction prompt per chunk
- Merges chunk candidates into one JSON per document
- Merges all document JSON files into company_dossier_merged.json

Input:
    project_root/Phase 01_TextExtraction/<doc_id>/file_llm_feed.md

Output:
    project_root/Phase03_ExplicitFieldsV2/<model_folder>/
        company_dossier_merged.json
        json_for_each_file/<doc_id>.json
        json_for_each_file/all_field_candidates.json

Examples:
    python 03_phase3_llm_only_explicit_fields.py --provider ollama --model deepseek-v3.2:cloud
    python 03_phase3_llm_only_explicit_fields.py --provider ollama --model deepseek-v3.2:cloud --limit 1

NVIDIA:
    set NVIDIA_API_KEY=your_key_here
    python 03_phase3_llm_only_explicit_fields.py --provider nvidia --model deepseek-ai/deepseek-v4-pro
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

PHASE1_FOLDER_NAME = "Phase 01_TextExtraction"
PHASE3_FOLDER_NAME = "Phase03_ExplicitFieldsV2"
PHASE1_MARKDOWN_NAME = "file_llm_feed.md"

FIELD_DEFINITIONS = {
    "company_name": "Common company name used in the documents.",
    "legal_company_name": "Full legal company name, if explicitly stated.",
    "website": "Company website URL or website text, if explicitly stated.",
    "one_line_description": "One sentence describing what the company does.",
    "industry_or_sector": "Industry, sector, market category, or business domain.",
    "company_stage_or_company_idea": "Company stage, idea stage, startup stage, development stage, or business concept.",
    "headquarters_location": "Headquarters or main company location.",
    "incorporation_location": "State or country of incorporation, if explicitly stated.",
    "year_founded": "Year founded, if explicitly stated.",
    "legal_entity_structure": "Legal entity type such as C-Corp, LLC, Inc., corporation, partnership, etc.",
    "company_problem": "Problem, pain point, unmet need, or market gap the company addresses.",
    "company_solution": "Product, technology, service, platform, or solution offered by the company.",
    "business_model": "How the company makes or plans to make money.",
    "target_customers": "Target customers, users, buyers, patients, market segments, or customer types.",
    "competitive_advantage": "Differentiators, moat, advantage, defensibility, unique IP, or technical edge.",
    "product_status": "Current product status, development stage, clinical stage, prototype status, launch status, or roadmap status.",
    "use_of_funds": "How raised capital or funding will be used.",
    "key_milestones": "Important business, product, clinical, technical, or operational milestones.",
    "funding_milestones": (
        "Funding-related milestones and funding events. Capture grants, awards, non-dilutive funding, "
        "government funding, DOD/NIH/NINDS/NSF/SBIR/STTR/DARPA funding, prior raises, current raises, "
        "financing milestones, and milestone/value inflection points."
    ),
    "cash_on_hand": "Cash balance or cash on hand.",
    "current_round": "Current fundraising round such as Seed, Series A, bridge round, etc.",
    "amount_raised_to_date": "Total capital raised to date, including grants if stated.",
    "target_raise_amount": "Amount the company is currently trying to raise.",
    "minimum_investment": "Minimum investment amount, if stated.",
    "previous_investors": "Prior investors, backers, funders, or grant sources.",
    "valuation_last_round": "Valuation from the previous funding round.",
    "valuation_current_ask": "Current valuation ask or proposed valuation.",
    "pre_money_valuation": "Pre-money valuation.",
    "post_money_valuation": "Post-money valuation.",
    "instrument": "Investment instrument such as SAFE, equity, debt, convertible note, grant, etc.",
    "revenue_monthly_annual": "Monthly revenue, annual revenue, revenue run-rate, or general revenue figures.",
    "mrr_arr": "MRR and ARR, if applicable.",
    "growth_rate_mom_yoy": "Month-over-month or year-over-year growth rate.",
    "gross_margin": "Gross margin or margin information.",
    "burn_rate": "Monthly burn, cash burn, or burn rate.",
    "runway_months": "Runway in months.",
    "ltv": "Lifetime value.",
    "payback_period": "Payback period.",
    "churn_rate": "Churn rate.",
    "users_total_active": "Total users, active users, customers, patients, or accounts.",
    "dau_mau": "DAU, MAU, or active user metrics.",
    "sales_pipeline": "Sales pipeline, prospective customers, deals, or revenue pipeline.",
    "conversion_rates": "Conversion rates.",
    "average_order_value": "Average order value.",
    "retention_rates": "Retention rates.",
    "engagement_metrics": "Engagement metrics.",
    "partnerships": "Strategic partnerships, commercial partnerships, research partnerships, or channel partners.",
    "contracts_signed": "Contracts that have been signed.",
    "lois": "Letters of intent, LOIs, memorandums of understanding, or similar soft commitments.",
    "key_hires": "Key hires, planned hires, or important team additions.",
    "incorporation_documents": "Corporate formation or incorporation document information.",
    "ip_ownership": "IP ownership, assignment, licenses, patents owned, or IP control.",
    "trademarks_patents": "Trademarks, patents, patent applications, issued patents, pending patents, or IP portfolio details.",
    "regulatory_exposure": "Regulatory exposure, FDA status, clinical/regulatory requirements, or compliance dependencies.",
    "risk_disclosures": "Risks, disclaimers, litigation, uncertainties, dependencies, or stated risk factors.",
    "market_size": "TAM, SAM, SOM, market size, market opportunity, or market forecast.",
}

DOCUMENT_CATEGORIES = [
    "Pitch deck / investor deck", "Investment tear sheet / offering summary", "Executive summary", "Investor update",
    "Term sheet", "Financial model / projections", "Cap table / ownership summary", "Revenue metrics", "Product metrics",
    "Cohort / retention analysis", "Churn analysis", "CAC / LTV / unit economics analysis", "SAFE agreement / legal document",
    "Convertible note agreement / legal document", "Subscription agreement / purchase agreement", "Customer list", "Sales pipeline",
    "Customer contract", "Enterprise customer agreement", "Key commercial contract", "Founder resume / biography", "Employee roster",
    "Hiring plan", "Organizational chart", "Product demo / screenshots", "Product roadmap", "Technical architecture",
    "Security policy", "Privacy policy", "Market / TAM analysis", "Competitive landscape", "Strategic partnership document",
    "Licensing agreement", "IP assignment", "Patent / IP summary", "Corporate charter / certificate of incorporation", "Bylaws",
    "Board minutes / board consent", "Certificate of good standing", "Option plan / equity incentive plan", "Bank statement",
    "Tax filing", "Insurance policy", "Litigation disclosure", "Vendor agreement", "Millenia engagement agreement", "Unknown File Type",
]


def clean_doc_id(name: str) -> str:
    stem = Path(name).stem.lower().strip()
    stem = re.sub(r"[^a-z0-9]+", "-", stem)
    stem = re.sub(r"-+", "-", stem).strip("-")
    return stem or "document"


def safe_model_folder_name(model: str) -> str:
    name = model.strip().replace("/", "_").replace(":", "_")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_") or "model"


def ensure_folder(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, data: Any) -> None:
    ensure_folder(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def truncate_for_log(value: Any, max_len: int = 100) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def extract_json_from_text(text: str) -> Any:
    text = strip_thinking(text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return json.loads(text[first:last + 1])
    raise ValueError("Could not parse JSON from LLM response.")


def stable_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def split_markdown_into_pages(markdown_text: str) -> list[dict[str, Any]]:
    text = normalize_text(markdown_text)
    pattern = re.compile(r"(?m)^---\s*\nPage\s+(\d+)\s*\n---\s*$")
    matches = list(pattern.finditer(text))
    if not matches:
        return [{"page_start": None, "page_end": None, "text": text, "estimated_tokens": estimate_tokens(text)}]
    pages = []
    preface = text[:matches[0].start()].strip()
    if preface:
        pages.append({"page_start": None, "page_end": None, "text": preface, "estimated_tokens": estimate_tokens(preface)})
    for i, match in enumerate(matches):
        page_no = int(match.group(1))
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        page_text = text[start:end].strip()
        pages.append({"page_start": page_no, "page_end": page_no, "text": page_text, "estimated_tokens": estimate_tokens(page_text)})
    return pages


def make_token_chunks(pages: list[dict[str, Any]], max_tokens_per_chunk: int = 20000, overlap_tokens: int = 2000) -> list[dict[str, Any]]:
    chunks, i = [], 0
    while i < len(pages):
        chunk_pages, token_total, start_i = [], 0, i
        while i < len(pages):
            page_tokens = pages[i]["estimated_tokens"]
            if chunk_pages and token_total + page_tokens > max_tokens_per_chunk:
                break
            chunk_pages.append(pages[i])
            token_total += page_tokens
            i += 1
            if token_total >= max_tokens_per_chunk:
                break
        if not chunk_pages:
            chunk_pages = [pages[i]]
            i += 1
        page_numbers = [p["page_start"] for p in chunk_pages if p.get("page_start") is not None]
        chunk_text = "\n\n".join(p["text"] for p in chunk_pages).strip()
        chunks.append({
            "chunk_index": len(chunks) + 1,
            "page_start": min(page_numbers) if page_numbers else None,
            "page_end": max(page_numbers) if page_numbers else None,
            "estimated_tokens": estimate_tokens(chunk_text),
            "text": chunk_text,
        })
        if overlap_tokens > 0 and i < len(pages):
            overlap_total, overlap_count = 0, 0
            for p in reversed(chunk_pages):
                overlap_total += p["estimated_tokens"]
                overlap_count += 1
                if overlap_total >= overlap_tokens:
                    break
            i = max(start_i + 1, i - overlap_count)
    return chunks


def call_ollama_chat(model: str, messages: list[dict[str, str]], ollama_url: str, temperature: float, max_tokens: int | None, timeout: int) -> str:
    payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}}
    if max_tokens is not None:
        payload["options"]["num_predict"] = max_tokens
    response = requests.post(ollama_url.rstrip("/") + "/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")


def call_openai_compatible_chat(model: str, messages: list[dict[str, str]], base_url: str, api_key: str, temperature: float, max_tokens: int | None, timeout: int, extra_body: dict[str, Any] | None = None) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise ImportError("Install OpenAI package first: pip install openai") from e
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if extra_body:
        kwargs["extra_body"] = extra_body
    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content or ""


def call_llm(provider: str, model: str, messages: list[dict[str, str]], ollama_url: str, base_url: str | None, api_key: str | None, temperature: float, max_tokens: int | None, timeout: int, retries: int) -> str:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            if provider == "ollama":
                return call_ollama_chat(model, messages, ollama_url, temperature, max_tokens, timeout)
            if provider == "nvidia":
                key = api_key or os.getenv("NVIDIA_API_KEY")
                if not key:
                    raise ValueError("NVIDIA_API_KEY is not set.")
                return call_openai_compatible_chat(model, messages, base_url or "https://integrate.api.nvidia.com/v1", key, temperature, max_tokens, timeout, {"chat_template_kwargs": {"thinking": False}})
            if provider == "openai":
                key = api_key or os.getenv("OPENAI_API_KEY")
                if not key:
                    raise ValueError("OPENAI_API_KEY is not set.")
                return call_openai_compatible_chat(model, messages, base_url or "https://api.openai.com/v1", key, temperature, max_tokens, timeout)
            if provider == "openai_compatible":
                key = api_key or os.getenv("OPENAI_API_KEY")
                if not key:
                    raise ValueError("API key is not set. Use --api-key or set OPENAI_API_KEY.")
                if not base_url:
                    raise ValueError("--base-url is required for provider=openai_compatible.")
                return call_openai_compatible_chat(model, messages, base_url, key, temperature, max_tokens, timeout)
            raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:
            last_error = e
            if attempt < retries:
                print(f"    LLM call failed, retrying in 2s: {e}")
                time.sleep(2)
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_error}")


def field_definitions_text() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in FIELD_DEFINITIONS.items())


def document_categories_text() -> str:
    return "\n".join(f"- {cat}" for cat in DOCUMENT_CATEGORIES)



def extraction_prompt(doc_id: str, file_name: str, chunk: dict[str, Any]) -> str:
    schema = {
        "doc_id": doc_id,
        "file_name": file_name,
        "chunk_index": chunk.get("chunk_index"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "document_category_candidates": [
            {
                "category": "one category from the list",
                "reason": "short reason",
                "evidence_quote": "exact quote from chunk",
            }
        ],
        "field_candidates": {
            "field_name_here": [
                {
                    "value": "specific extracted value or list/object if needed",
                    "answer": "clear bullet-style answer based only on evidence",
                    "evidence": [
                        {
                            "quote": "exact quote from chunk",
                            "page_start": chunk.get("page_start"),
                            "page_end": chunk.get("page_end"),
                        }
                    ],
                }
            ]
        },
        "key_people_candidates": [
            {
                "full_name": "Person full name",
                "role": "role/title if stated",
                "linkedin_profiles": [],
                "contact_information": [],
                "prior_exits_experience": [],
                "education": [],
                "document_risk_notes": [],
                "evidence": [
                    {
                        "quote": "exact quote from chunk",
                        "page_start": chunk.get("page_start"),
                        "page_end": chunk.get("page_end"),
                    }
                ],
            }
        ],
        "short_summary_candidate": "brief chunk summary if useful",
        "long_summary_candidate": "more detailed chunk summary if useful",
    }

    return f"""
You are extracting structured company information from one parsed document chunk.

Document:
- doc_id: {doc_id}
- file_name: {file_name}
- page_start: {chunk.get('page_start')}
- page_end: {chunk.get('page_end')}
- chunk_index: {chunk.get('chunk_index')}

Rules:
- Do not force fields.
- Leave fields out when unsupported.
- Preserve exact numbers, dates, grant amounts, investor names, legal names, disease names, locations, and URLs.
- Capture grant funding under funding_milestones.
- funding_milestones includes grants, awards, non-dilutive funding, government funding, DOD, NIH, NINDS, NSF, SBIR, STTR, DARPA, prior raises, and current raises.
- If a sentence mentions money, grants, awards, raise amount, valuation, burn, runway, or funding source, do not ignore it.
- Evidence must quote exact text from the chunk.
- Do not invent LinkedIn URLs.
- Do not classify organizations as people.
- Only output valid JSON. No markdown. No explanation.

Available document categories:
{document_categories_text()}

Explicit fields:
{field_definitions_text()}

Return JSON exactly in this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Chunk text:
{chunk['text']}
""".strip()


def document_merge_prompt(doc_id: str, file_name: str, chunk_candidates: list[dict[str, Any]]) -> str:
    schema = {
        "doc_id": doc_id,
        "file_name": file_name,
        "document_category": "best category from the list",
        "short_summary": "short document summary",
        "long_summary": "comprehensive document summary",
        "final_fields": {
            "field_name": {
                "value": "clean value, list, or object",
                "answer": "LLM-generated answer based only on evidence",
                "evidence": [
                    {
                        "quote": "exact quote",
                        "page_start": None,
                        "page_end": None,
                        "source_chunk_index": None,
                    }
                ],
            }
        },
        "key_people": [
            {
                "full_name": "Person full name",
                "role": "role/title if stated",
                "linkedin_profiles": [],
                "contact_information": [],
                "prior_exits_experience": [],
                "education": [],
                "document_risk_notes": [],
                "evidence": [
                    {
                        "quote": "exact quote",
                        "page_start": None,
                        "page_end": None,
                        "source_chunk_index": None,
                    }
                ],
            }
        ],
    }

    return f"""
Merge extraction candidates from one document into one clean per-document JSON.

Document:
- doc_id: {doc_id}
- file_name: {file_name}

Rules:
- Use only the provided candidates.
- Do not invent facts.
- Prefer exact values with direct evidence.
- If candidates disagree, keep the most specific and best-supported answer.
- Keep evidence for each final field.
- Remove duplicate people.
- Include only people who have a defined role/title OR contact information OR clear document importance.
- For funding_milestones, include grants, awards, non-dilutive funding, DOD/NIH/NINDS/NSF/SBIR/STTR/DARPA funding, prior raises, and current raises when present.
- Make field answers more comprehensive than raw snippets, but still grounded in evidence.
- Output valid JSON only. No markdown. No explanation.

Available document categories:
{document_categories_text()}

Explicit fields:
{field_definitions_text()}

Return JSON in this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Candidate JSON:
{json.dumps(chunk_candidates, ensure_ascii=False, indent=2)}
""".strip()


def dossier_merge_prompt(document_jsons: list[dict[str, Any]]) -> str:
    schema = {
        "short_summary": "short company summary",
        "long_summary": "comprehensive company summary",
        "final_fields": {
            "field_name": {
                "value": "clean final value, list, or object",
                "answer": "LLM-generated answer based only on evidence",
                "evidence": [
                    {
                        "doc_id": "source doc id",
                        "quote": "exact quote",
                        "page_start": None,
                        "page_end": None,
                    }
                ],
            }
        },
        "key_people": [
            {
                "full_name": "Person full name",
                "role": "role/title if stated",
                "linkedin_profiles": [],
                "contact_information": [],
                "prior_exits_experience": [],
                "education": [],
                "document_risk_notes": [],
                "evidence": [
                    {
                        "doc_id": "source doc id",
                        "quote": "exact quote",
                        "page_start": None,
                        "page_end": None,
                    }
                ],
            }
        ],
        "documents": [
            {
                "doc_id": "document id",
                "file_name": "file name",
                "document_category": "category",
                "short_summary": "summary",
            }
        ],
    }

    return f"""
Merge multiple per-document company JSON files into one clean company dossier.

Rules:
- Use only the provided document JSONs.
- Do not invent facts.
- Prefer the most specific, most recent, and best-supported values.
- If documents disagree, explain uncertainty in the answer.
- Keep evidence for every final field.
- The final dossier should have one clear answer for each field, not duplicate candidates.
- Key people should only include people with a defined role/title OR contact information OR clear company importance.
- For funding_milestones, include grants, awards, non-dilutive funding, DOD/NIH/NINDS/NSF/SBIR/STTR/DARPA funding, prior raises, and current raises when present.
- Generate one short summary and one long summary based on all documents.
- Output valid JSON only. No markdown. No explanation.

Explicit fields:
{field_definitions_text()}

Return JSON in this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Per-document JSONs:
{json.dumps(document_jsons, ensure_ascii=False, indent=2)}
""".strip()

def ask_json(prompt: str, args: argparse.Namespace, system: str) -> dict[str, Any]:
    raw = call_llm(
        provider=args.provider,
        model=args.model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        ollama_url=args.ollama_url,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_output_tokens,
        timeout=args.timeout,
        retries=args.retries,
    )
    data = extract_json_from_text(raw)
    if not isinstance(data, dict):
        raise ValueError("LLM did not return a JSON object.")
    return data


def summarize_found_fields(chunk_result: dict[str, Any]) -> str:
    fields = chunk_result.get("field_candidates", {})
    people = chunk_result.get("key_people_candidates", [])
    parts = []
    if isinstance(fields, dict):
        for field_name, candidates in fields.items():
            if not candidates:
                continue
            first_candidate = stable_list(candidates)[0]
            value = first_candidate.get("value") if isinstance(first_candidate, dict) else first_candidate
            parts.append(f'{field_name}="{truncate_for_log(value)}"')
    if isinstance(people, list) and people:
        names = [p.get("full_name") for p in people[:5] if isinstance(p, dict) and p.get("full_name")]
        if names:
            parts.append(f"people={names}")
    return ", ".join(parts[:8]) if parts else "no obvious fields found"


def find_phase1_documents(phase1_folder: Path, markdown_name: str) -> list[dict[str, Any]]:
    docs = []
    for folder in sorted(phase1_folder.iterdir(), key=lambda p: p.name.lower()):
        if not folder.is_dir():
            continue
        md_path = folder / markdown_name
        if md_path.exists():
            docs.append({"doc_id": clean_doc_id(folder.name), "file_name": folder.name, "phase1_folder": str(folder), "markdown_path": md_path})
    return docs


def process_document(doc_record: dict[str, Any], json_for_each_file_folder: Path, all_field_candidates: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    doc_id, file_name, markdown_path = doc_record["doc_id"], doc_record["file_name"], doc_record["markdown_path"]
    pages = split_markdown_into_pages(read_text(markdown_path))
    chunks = make_token_chunks(pages, args.max_tokens_per_chunk, args.overlap_tokens)
    print(f"  pages: {len(pages)}")
    print(f"  chunks: {len(chunks)}")
    chunk_candidates = []

    for chunk in chunks:
        print(f"  [chunk {chunk['chunk_index']}/{len(chunks)}] pages={chunk.get('page_start')}-{chunk.get('page_end')} estimated_tokens={chunk.get('estimated_tokens')}")
        try:
            result = ask_json(
                extraction_prompt(doc_id, file_name, chunk),
                args,
                "You are a careful information extraction engine. Return only valid JSON.",
            )
            result.setdefault("doc_id", doc_id)
            result.setdefault("file_name", file_name)
            result.setdefault("chunk_index", chunk["chunk_index"])
            result.setdefault("page_start", chunk.get("page_start"))
            result.setdefault("page_end", chunk.get("page_end"))
            chunk_candidates.append(result)
            print(f"    found: {summarize_found_fields(result)}")
        except Exception as e:
            error_obj = {"doc_id": doc_id, "file_name": file_name, "chunk_index": chunk["chunk_index"], "page_start": chunk.get("page_start"), "page_end": chunk.get("page_end"), "status": "error", "error": repr(e)}
            chunk_candidates.append(error_obj)
            print(f"    extraction failed: {e}")
            if args.stop_on_error:
                raise

        all_field_candidates[doc_id] = {
            "doc_id": doc_id,
            "file_name": file_name,
            "source_markdown_file": str(markdown_path),
            "phase1_folder": doc_record["phase1_folder"],
            "chunk_count": len(chunks),
            "chunk_candidates": chunk_candidates,
        }
        write_json(json_for_each_file_folder / "all_field_candidates.json", all_field_candidates)

    print("  merging document candidates...")
    document_json = ask_json(
        document_merge_prompt(doc_id, file_name, chunk_candidates),
        args,
        "You are a careful document-level extraction merger. Return only valid JSON.",
    )
    document_json.setdefault("doc_id", doc_id)
    document_json.setdefault("file_name", file_name)
    document_json["source_markdown_file"] = str(markdown_path)
    document_json["phase1_folder"] = doc_record["phase1_folder"]
    document_json["chunk_count"] = len(chunks)

    document_output_path = json_for_each_file_folder / f"{doc_id}.json"
    write_json(document_output_path, document_json)
    all_field_candidates[doc_id]["document_output_file"] = str(document_output_path)
    write_json(json_for_each_file_folder / "all_field_candidates.json", all_field_candidates)
    return document_json


def run_phase3(args: argparse.Namespace) -> Path:
    project_root = Path.cwd().resolve()
    phase1_folder = project_root / PHASE1_FOLDER_NAME
    output_folder = ensure_folder(project_root / PHASE3_FOLDER_NAME / safe_model_folder_name(args.model))
    json_for_each_file_folder = ensure_folder(output_folder / "json_for_each_file")
    if not phase1_folder.exists():
        raise FileNotFoundError(f"Missing Phase 1 folder: {phase1_folder}")
    docs = find_phase1_documents(phase1_folder, args.markdown_file)
    if args.limit is not None:
        docs = docs[:args.limit]

    print("=" * 80)
    print("PHASE 3 LLM-ONLY COMPANY EXTRACTION START")
    print("=" * 80)
    print(f"Working folder:             {project_root}")
    print(f"Phase 1 folder:             {phase1_folder}")
    print(f"Markdown file:              {args.markdown_file}")
    print(f"Provider:                   {args.provider}")
    print(f"Model:                      {args.model}")
    print(f"Output folder:              {output_folder}")
    print(f"Documents to process:       {len(docs)}")
    print("No Chroma:                  True")
    print("No GLI/NER:                 True")
    print(f"Max tokens per chunk:       {args.max_tokens_per_chunk}")
    print(f"Overlap tokens:             {args.overlap_tokens}")
    print("Approx token estimate:      characters / 4")
    print("=" * 80)

    all_path = json_for_each_file_folder / "all_field_candidates.json"
    if args.resume and all_path.exists():
        try:
            all_field_candidates = json.loads(all_path.read_text(encoding="utf-8"))
        except Exception:
            all_field_candidates = {}
    else:
        all_field_candidates = {}

    document_jsons = []
    for index, doc_record in enumerate(docs, start=1):
        print(f"[Document {index}/{len(docs)}] {doc_record['doc_id']}")
        doc_output_path = json_for_each_file_folder / f"{doc_record['doc_id']}.json"
        if args.resume and doc_output_path.exists():
            print("  already exists, loading because --resume is enabled")
            document_jsons.append(json.loads(doc_output_path.read_text(encoding="utf-8")))
            continue
        try:
            document_jsons.append(process_document(doc_record, json_for_each_file_folder, all_field_candidates, args))
            print("  Status: ok")
        except Exception as e:
            print("  Status: error")
            print(f"  Reason: {e}")
            if args.stop_on_error:
                raise

    print("=" * 80)
    print("All documents scanned. Merging company dossier across all documents...")
    print("=" * 80)
    if document_jsons:
        dossier = ask_json(
            dossier_merge_prompt(document_jsons),
            args,
            "You are a careful company dossier merger. Return only valid JSON.",
        )
        dossier["model_used"] = args.model
        dossier["provider_used"] = args.provider
        dossier["source_document_count"] = len(document_jsons)
        write_json(output_folder / "company_dossier_merged.json", dossier)
        print(f"Saved merged dossier: {output_folder / 'company_dossier_merged.json'}")
    else:
        print("No document JSONs were created. Skipping company dossier merge.")

    print("=" * 80)
    print("PHASE 3 LLM-ONLY COMPANY EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Output folder: {output_folder}")
    return output_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 LLM-only explicit field extraction using token-based chunks.")
    parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "nvidia", "openai", "openai_compatible"], help="LLM provider.")
    parser.add_argument("--model", type=str, default="deepseek-v3.2:cloud", help="Model name.")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama server URL.")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for OpenAI-compatible providers.")
    parser.add_argument("--api-key", type=str, default=None, help="Optional API key. Prefer environment variables when possible.")
    parser.add_argument("--markdown-file", type=str, default=PHASE1_MARKDOWN_NAME, help="Markdown file name inside each Phase 1 document folder.")
    parser.add_argument("--max-tokens-per-chunk", type=int, default=20000, help="Approximate max input tokens per chunk. Uses characters / 4 estimate.")
    parser.add_argument("--overlap-tokens", type=int, default=2000, help="Approximate overlap tokens between chunks. Uses page-based overlap.")
    parser.add_argument("--max-output-tokens", type=int, default=12000, help="Max output tokens for the LLM response, when provider supports it.")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature.")
    parser.add_argument("--timeout", type=int, default=900, help="HTTP/API timeout seconds.")
    parser.add_argument("--retries", type=int, default=2, help="LLM call retries.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of documents to process for testing.")
    parser.add_argument("--resume", action="store_true", help="Skip documents that already have per-document JSON outputs.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately on the first document/chunk error.")
    return parser.parse_args()


def main() -> None:
    run_phase3(parse_args())


if __name__ == "__main__":
    main()
