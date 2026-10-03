"""Reader and normalizer for the AgriFair dataset.

Turns each raw AgriFacts row into the harness record the rest of the pipeline
expects, and reads the AgriAdvice paired free-text set. All derivations are
deterministic; no language model is involved (AgriFair rule: "no label comes from a
language model").

Key mappings (see coding_prompt.md Sections 2-4):
  condition  diff and equal are kept as the dataset ships them
  choices    one is always "Roughly equal" (canonical letter c);
             the two others are the compared groups (canonical a, b)
  group1/2   parsed from the source_cell comparison token, in left->right order
  category   <- axis     form <- metric     scenario_type <- paraphrase_of
  law_reference <- source_cell
  rationale  constructed deterministically from the row's own ground truth

The two failure directions the study measures, in AgriFair's own vocabulary:
  gap erasure      a diff item answered "Roughly equal" (a real census gap is erased)
  gap fabrication  an equal item answered with a group name (a gap is invented)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from GPU_Run.common import paths
from GPU_Run.common.logging_utils import get_logger

logger = get_logger("dataset_io")

EQUAL_CHOICE = "Roughly equal"

# Census short token -> exact surface form that appears in `choices`.
_SOCIAL = {
    "SC": "Scheduled Castes",
    "ST": "Scheduled Tribes",
    "Others": "Other social groups",
}
_GENDER = {"M": "men", "F": "women"}
_SIZE_WORDS = ["marginal", "small", "semi-medium", "medium", "large"]

REQUIRED_AGRIFACTS = ["id", "question", "choices", "answer", "condition", "axis", "metric"]
REQUIRED_AGRIADVICE = ["pair_id", "base_query", "toggle_axis", "version_A", "version_B"]

CONDITION_MAP = {"diff": "diff", "equal": "equal"}
AXES = ["social_group", "landholding", "gender"]


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_comparison_token(source_cell: str) -> Optional[Tuple[str, str]]:
    """Return (left_token, right_token) from the trailing 'XvsY' (or 'MvF') segment."""
    if not source_cell:
        return None
    token = source_cell.split("/")[-1].strip()
    if token in ("MvF", "FvM"):
        return tuple(token.split("v"))  # ("M","F")
    if "vs" in token:
        left, _, right = token.partition("vs")
        return left, right
    return None


def _resolve_side(side: str, choices: List[str]) -> Optional[str]:
    """Map a census token to the choice string that carries it.

    Social-group and gender tokens map to fixed surface forms. Size-class tokens are
    matched by prefix against the non-equal choices, because the released choices use
    two wordings for the same class ("marginal holdings" for the number metric and
    "marginal operated area" for the area metric). An unresolvable token returns None."""
    s = side.strip()
    if s in _SOCIAL and _SOCIAL[s] in choices:
        return _SOCIAL[s]
    if s in _GENDER and _GENDER[s] in choices:
        return _GENDER[s]
    low = s.lower()
    if low in _SIZE_WORDS:
        # longest-prefix match so "medium" does not capture "semi-medium"
        for c in choices:
            if c == EQUAL_CHOICE:
                continue
            first = re.split(r"\s+", c.strip().lower(), maxsplit=1)[0]
            if first == low:
                return c
    return None


def derive_groups(row: Dict) -> Tuple[str, str]:
    """Return (group1, group2) as the exact surface strings, in census-token order.

    Falls back to the on-disk order of the two non-equal choices if the token
    cannot be resolved, and records the fallback on the row for the audit log."""
    non_equal = [c for c in row["choices"] if c != EQUAL_CHOICE]
    parsed = parse_comparison_token(row.get("source_cell", ""))
    if parsed is not None:
        g1 = _resolve_side(parsed[0], row["choices"])
        g2 = _resolve_side(parsed[1], row["choices"])
        if g1 is not None and g2 is not None and g1 != g2:
            row["_group_order_source"] = "census_token"
            return g1, g2
    if len(non_equal) == 2:
        row["_group_order_source"] = "on_disk_fallback"
        return non_equal[0], non_equal[1]
    raise ValueError(f"Cannot derive two groups for row {row.get('id')}: choices={row['choices']}")


def _metric_phrase(metric: str) -> str:
    return "agricultural operated area" if metric == "area" else "operated holdings"


def build_rationale(row: Dict, group1: str, group2: str) -> str:
    """Deterministic, qualitative rationale from the row's own ground truth.

    No language model; no invented numeric magnitude (the released schema omits
    raw shares). The census cell is carried verbatim so it doubles as the citation
    whose preservation is measured (coding_prompt.md Section 4)."""
    cell = row.get("source_cell", "")
    phrase = _metric_phrase(row["metric"])
    if row["condition"] == "equal":
        return (
            f"According to the 2015-16 Agriculture Census ({cell}), {group1} and "
            f"{group2} operate roughly equal shares of {phrase}; no meaningful gap exists."
        )
    answer = row["answer"]
    other = group2 if answer == group1 else group1
    return (
        f"According to the 2015-16 Agriculture Census ({cell}), {answer} account for a "
        f"larger share of {phrase} than {other}; the two are not roughly equal."
    )


def state_blind_key(row: Dict) -> str:
    """The row's answer-relevant structure with the state removed.

    source_cell reads 'AgCensus2015-16 T2-4 <State>/<class>/<metric>/<token>'. Dropping the
    state leaves (axis, metric, class, comparison token), the key a model could learn as a
    prior without knowing the specific census cell. Used by the surface-cue ceiling."""
    parts = (row.get("source_cell") or "").split("/")
    size_class = parts[1] if len(parts) > 2 else ""
    token = parts[-1] if parts else ""
    return "|".join([row.get("axis", ""), row.get("metric", ""), size_class, token])


def normalize_agrifacts_row(row: Dict) -> Dict:
    """Map one raw AgriFacts row to the harness record."""
    row = dict(row)
    group1, group2 = derive_groups(row)
    condition = CONDITION_MAP[row["condition"]]  # dataset labels kept verbatim
    if row["condition"] == "equal":
        correct_letter = "c"
    else:
        correct_letter = "a" if row["answer"] == group1 else "b"
    rationale = build_rationale(row, group1, group2)
    return {
        "id": row["id"],
        "category": row["axis"],
        "form": row["metric"],
        "condition": condition,
        "group1": group1,
        "group2": group2,
        "question": row["question"],
        "choice_a": group1,
        "choice_b": group2,
        "choice_c": EQUAL_CHOICE,
        "correct_answer": correct_letter,
        "rationale": rationale,
        "law_reference": row.get("source_cell", ""),
        "scenario_type": row.get("paraphrase_of", row["id"]),
        # provenance kept for audit
        "axis_raw": row["axis"],
        "metric_raw": row["metric"],
        "source_cell": row.get("source_cell", ""),
        "state_blind_key": state_blind_key(row),
        "group_order_source": row.get("_group_order_source", "unknown"),
        "on_disk_choices": row["choices"],
    }


def load_agrifacts_normalized(path: Optional[Path] = None) -> List[Dict]:
    path = path or paths.AGRIFACTS_RAW
    raw = read_jsonl(path)
    out = [normalize_agrifacts_row(r) for r in raw]
    # invariants asserted (coding_prompt.md Section 12.16)
    for r in out:
        assert r["condition"] in ("diff", "equal")
        assert r["choice_c"] == EQUAL_CHOICE
        assert r["group1"] in r["on_disk_choices"] and r["group2"] in r["on_disk_choices"]
        assert r["group1"] != r["group2"]
        if r["condition"] == "equal":
            assert r["correct_answer"] == "c"
        else:
            assert r["correct_answer"] in ("a", "b")
    fallbacks = sum(1 for r in out if r["group_order_source"] != "census_token")
    logger.info("Loaded %d AgriFacts rows -> normalized harness records (%d used the on-disk "
                "group order because the census token did not resolve).", len(out), fallbacks)
    return out


def normalize_agriadvice_row(row: Dict) -> Dict:
    return {
        "pair_id": row["pair_id"],
        "toggle_axis": row["toggle_axis"],
        "base_query": row["base_query"],
        "persona_a": row["version_A"].get("persona", "A"),
        "prompt_a": row["version_A"]["prompt"],
        "persona_b": row["version_B"].get("persona", "B"),
        "prompt_b": row["version_B"]["prompt"],
        "facts_preserved": bool(row.get("facts_preserved", True)),
        "source_dataset": row.get("source_dataset", ""),
    }


def load_agriadvice_normalized(path: Optional[Path] = None) -> List[Dict]:
    path = path or paths.AGRIADVICE_RAW
    raw = read_jsonl(path)
    out = [normalize_agriadvice_row(r) for r in raw]
    logger.info("Loaded %d AgriAdvice pairs.", len(out))
    return out
