"""The only reader of .env for the AgriFair GRAFT study.

Accepts the canonical variable names and a set of legacy aliases so the .env
that ships with the project works unchanged. Never
prints a value. Keys are read only through this module.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from GPU_Run.common.paths import REPO_ROOT

_ENV_PATH = REPO_ROOT / ".env"
_LOADED = False

# canonical -> list of accepted source names (first hit wins).
_ALIASES: Dict[str, List[str]] = {
    "HF_KEY": ["HF_KEY", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN"],
    "GCP_Key1": ["GCP_Key1", "GEMINI_API_KEY_1"],
    "GCP_Key2": ["GCP_Key2", "GEMINI_API_KEY_2"],
    "GCP_Key3": ["GCP_Key3", "GEMINI_API_KEY_3"],
    "GCP_key4": ["GCP_key4", "GEMINI_API_KEY_4"],
    "DEEPSEEK_KEY1": ["DEEPSEEK_KEY1", "DEEPSEEK_API_KEY_1"],
    "DEEPSEEK_KEY2": ["DEEPSEEK_KEY2", "DEEPSEEK_API_KEY_2"],
    "MISTRAL_KEY1": ["MISTRAL_KEY1", "MISTRAL_API_KEY1"],
    "MISTRAL_KEY2": ["MISTRAL_KEY2", "MISTRAL_API_KEY2"],
    "OPENROUTER_KEY1": ["OPENROUTER_KEY1", "OPENROUTER_API_KEY_1"],
    "OPENROUTER_KEY2": ["OPENROUTER_KEY2", "OPENROUTER_API_KEY_2"],
    "GITHUB_TOKEN": ["GITHUB_TOKEN", "Github_Classic_Token"],
    # native provider keys for the frontier panel, primary route for two of the models
    "OPENAI_DIRECT_KEY": ["OPENAI_DIRECT_KEY", "Open_AI_2009_Key", "OPENAI_API_KEY"],
    "ANTHROPIC_DIRECT_KEY": ["ANTHROPIC_DIRECT_KEY", "Calude_2009_API_Key",
                             "Claude_2009_API_Key", "ANTHROPIC_API_KEY"],
    "ANTHROPIC_WORKSPACE_ID": ["ANTHROPIC_WORKSPACE_ID", "Claude_workspace_id"],
    "XAI_DIRECT_KEY": ["XAI_DIRECT_KEY", "Grok_API_KEY", "GROK_API_KEY"],
    "AWS_ACCESS_KEY_1": ["AWS_ACCESS_KEY"],
    "AWS_SECRET_KEY_1": ["AWS_SECRET_KEY"],
    "AWS_ACCESS_KEY_2": ["AWS_ACCESS_KEY2"],
    "AWS_SECRET_KEY_2": ["AWS_SECRET_KEY2"],
    "AWS_REGION": ["AWS_REGION", "AWS_DEFAULT_REGION"],
    # provider model strings / base urls (non-secret, but read here too)
    "DEEPSEEK_API_BASE_URL": ["DEEPSEEK_API_BASE_URL"],
    "DEEPSEEK_JUDGE_MODEL_NAME": ["DEEPSEEK_JUDGE_MODEL_NAME", "DEEPSEEK_PRIMARY_MODEL_NAME"],
    "OPENROUTER_API_BASE_URL": ["OPENROUTER_API_BASE_URL"],
    "OPENROUTER_MODEL_NAME": ["OPENROUTER_PRIMARY_MODEL_NAME"],
    "GEMINI_MODEL_NAME": ["GEMINI_MODEL_NAME"],
    "MISTRAL_MODEL_NAME": ["MISTRAL_MODEL_NAME"],
}


def load_env(force: bool = False) -> None:
    """Parse .env into os.environ once. Lines are KEY=VALUE; quotes stripped."""
    global _LOADED
    if _LOADED and not force:
        return
    if _ENV_PATH.exists():
        for raw in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            # strip inline comments only when value is quoted
            val = val.strip()
            if val and val[0] in "\"'" and val[0] in val[1:]:
                q = val[0]
                val = val[1:val.index(q, 1)]
            else:
                val = val.split("#", 1)[0].strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = val
    _LOADED = True


def get(canonical: str, default: Optional[str] = None) -> Optional[str]:
    """Return the value for a canonical key, trying its legacy aliases."""
    load_env()
    for name in _ALIASES.get(canonical, [canonical]):
        v = os.environ.get(name)
        if v:
            return v
    return default


def get_round_robin(prefix_canonicals: List[str]) -> List[str]:
    """Collect all present keys for a provider into a round-robin list."""
    out: List[str] = []
    for c in prefix_canonicals:
        v = get(c)
        if v:
            out.append(v)
    return out


def has(canonical: str) -> bool:
    return get(canonical) is not None


def hf_token() -> Optional[str]:
    return get("HF_KEY")
