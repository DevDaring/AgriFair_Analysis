"""The single deterministic JSON answer/rationale extractor.

`json_repair_parse` strips code fences, fixes trailing commas, extracts the first
balanced JSON object, and reads fields directly. No judge model is used for answer
extraction. The judge chain is a fallback only when this parser fails on a specific
item, and the free-text heuristic is the last resort after both.
"""
from __future__ import annotations

import json
import re
from typing import Dict, Optional, Sequence, Tuple

_FENCE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _first_balanced_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def json_repair_parse(text: str) -> Optional[Dict]:
    """Best-effort deterministic parse of a JSON object out of model output."""
    if text is None:
        return None
    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = _FENCE.sub("", cleaned).strip()
    candidate = _first_balanced_object(cleaned)
    if candidate is None:
        candidate = cleaned
    candidate = _TRAILING_COMMA.sub(r"\1", candidate)
    for attempt in (candidate, candidate.replace("'", '"')):
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    # an unterminated object (generation cut off) whose first field is complete
    m = re.search(r'"answer_choice_letter"\s*:\s*"([A-Za-z])"', cleaned)
    if m:
        return {"answer_choice_letter": m.group(1)}
    return None


def extract_answer_letter(text: str, letters: Sequence[str] = ("a", "b", "c")) -> Tuple[Optional[str], bool]:
    """Return (letter in `letters`, parse_ok). parse_ok is False if JSON parse failed."""
    obj = json_repair_parse(text)
    if obj is not None and "answer_choice_letter" in obj:
        val = str(obj["answer_choice_letter"]).strip().lower().strip("()")
        if val in letters:
            return val, True
    return None, False


def extract_rationale(text: str) -> Dict:
    """Return {'rationale': str, 'law_references': list} from a rationale-arm response."""
    obj = json_repair_parse(text) or {}
    rationale = str(obj.get("rationale", "")).strip()
    refs = obj.get("cited_law_references") or obj.get("law_references") or []
    if isinstance(refs, str):
        refs = [refs]
    return {"rationale": rationale, "law_references": [str(r) for r in refs]}


def fallback_letter_from_freetext(text: str, letters: Sequence[str] = ("a", "b", "c")) -> Optional[str]:
    """Last-resort heuristic; only used after json parse AND judge both unavailable."""
    if not text:
        return None
    pattern = re.compile(r"\(?\b([%s])\b\)?" % "".join(letters), re.IGNORECASE)
    m = pattern.search(_THINK_BLOCK.sub("", text).strip())
    return m.group(1).lower() if m else None
