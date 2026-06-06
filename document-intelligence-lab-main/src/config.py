from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


ROOT_DIR = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT_DIR / "templates"
RUNS_DIR = Path(env_str("RUNS_DIR", "runs"))
OLLAMA_URL = env_str("OLLAMA_URL", "http://localhost:11434")

DEFAULT_VLM_MODEL = env_str("DEFAULT_VLM_MODEL", "llama3.2-vision:latest")
DEFAULT_LLM_MODEL = env_str("DEFAULT_LLM_MODEL", "deepseek-r1:8b")

CROP_ZOOM = env_float("CROP_ZOOM", 2.0)
OUTLINE_BOX_WIDTH = env_float("OUTLINE_BOX_WIDTH", 2.5)
