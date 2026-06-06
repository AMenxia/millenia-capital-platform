from __future__ import annotations

from typing import Any

import requests


def run_ollama_llm(model: str, ollama_url: str, user_content: str, system_prompt: str, timeout: int = 240) -> str:
    url = f"{ollama_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
    }

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return str(data.get("message", {}).get("content", "")).strip()
