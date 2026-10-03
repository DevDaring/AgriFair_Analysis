"""Shared helpers for Submission 2. Metric code is imported from Next_Run, not copied."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from Next_Run.common import (CANONICAL, EQUAL_LETTER, Counts, error_type, harmonic_b,  # noqa: F401
                             metrics_from_counts, counts_by_cluster, read_jsonl, write_csv,
                             write_json, write_jsonl, scrub_secrets, sha256_file, sha256_obj)

CODES_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def read_csv(path: Path) -> List[Dict]:
    import csv
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path: Path = CONFIG_PATH) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def output_dir(cfg: Dict) -> Path:
    p = CODES_ROOT / cfg["output_directory"]; p.mkdir(parents=True, exist_ok=True); return p


def data_dir(cfg: Dict) -> Path:
    p = CODES_ROOT / cfg["data_directory"]; p.mkdir(parents=True, exist_ok=True); return p


# ------------------------------------------------------------------ economics / footprint

def cost_usd(cfg: Dict, tier: str, input_tokens: int, output_tokens: int, route: str = "") -> float:
    t = cfg["tariffs"][tier]
    if route.startswith("openrouter"):
        t = cfg.get("tariffs_openrouter", {}).get(tier, t)
    return input_tokens * t["input"] / 1e6 + output_tokens * t["output"] / 1e6


def cost_inr(cfg: Dict, usd: float) -> float:
    return usd * float(cfg["tariffs"]["usd_inr"])


def energy_wh(cfg: Dict, tier: str, tokens: int) -> Dict[str, float]:
    e = cfg["energy_proxy"][tier]
    return {"low": tokens / 1000 * e["low"], "high": tokens / 1000 * e["high"]}


# ------------------------------------------------------------------ text hygiene

_PHONE = re.compile(r"(?:\+91[\s-]?|0)?(?<![\d/.])[6-9]\d{4}[\s-]?\d{5}(?![\d/.])")   # Indian mobile numbers only; doses and prices untouched
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")


def strip_personal_data(text: str) -> str:
    """Remove phone numbers, e-mails and 12-digit id patterns from a farmer query."""
    t = _AADHAAR.sub("[id]", text or "")
    t = _EMAIL.sub("[email]", t)
    t = _PHONE.sub("[number]", t)
    return re.sub(r"\s+", " ", t).strip()


def prompt_key(*parts) -> str:
    """Stable cache key: sha256 over the JSON of every part that changes the response."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("", "0", "false", "False")
