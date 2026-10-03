"""Shared foundations for the final bounded audit round (Codes/Next_Run).

Everything here is deliberately small and testable, because the plan's whole value rests on
getting a few definitions exactly right:

  * the three-option canonical answer space and what each kind of error is called
  * the harmonic score B, which is undefined - not zero - when a condition is absent
  * one normalised prediction record per (model, method, seed, split, item)
  * source-cell clustering, the statistical unit for every interval and test

Nothing in this module reads .env, calls a network, or writes into the legacy results/ or
data/ directories. Stage scripts inherit those properties by going through it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

CODES_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODES_ROOT))

import numpy as np  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
CANONICAL = ("a", "b", "c")
EQUAL_LETTER = "c"           # "Roughly equal" is always canonical c (prompts.py section 2.1)
AXES = ("social_group", "landholding", "gender")
TEST_SLICES = ("structure_familiar", "structure_novel")

# ----------------------------------------------------------------------------- config

def load_config(path: Path = CONFIG_PATH) -> Dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in ("allow_paid_api", "allow_training", "allow_new_attribution", "gpu_enabled"):
        cfg[k] = bool(cfg.get(k, False))
    return cfg


def smoke(cfg: Dict) -> Dict:
    """The smoke section, or an empty dict. Stages call this to decide whether to shrink."""
    sm = cfg.get("smoke") or {}
    return sm if sm.get("enabled") else {}


def limit_arms(arms: Dict, cfg: Dict, always_keep=("graft_proposed", "frozen_base", "ablation_placement_uniform")):
    """Smoke only: keep at most N arms per tier, always including the reference and one control
    so that every family has at least one computable contrast."""
    n = smoke(cfg).get("max_arms_per_tier")
    if not n:
        return arms
    out, per_tier = {}, {}
    for arm in sorted(arms, key=lambda a: (a.tier, a.method not in always_keep, a.method, a.seed)):
        if arm.method in always_keep or per_tier.get(arm.tier, 0) < n:
            out[arm] = arms[arm]
            per_tier[arm.tier] = per_tier.get(arm.tier, 0) + (arm.method not in always_keep)
    return out


def output_dir(cfg: Dict) -> Path:
    out = CODES_ROOT / cfg["output_directory"]
    out.mkdir(parents=True, exist_ok=True)
    return out


def legacy_results_dir() -> Path:
    return CODES_ROOT / "results"


def legacy_data_dir() -> Path:
    return CODES_ROOT / "data"


def assert_not_legacy(path: Path) -> None:
    """Refuse to write anywhere under results/ or data/. The plan's first rule."""
    p = Path(path).resolve()
    for forbidden in (legacy_results_dir(), legacy_data_dir()):
        if forbidden.resolve() in p.parents or p == forbidden.resolve():
            raise PermissionError(f"refusing to write into legacy directory: {p}")


# ----------------------------------------------------------------------------- secrets

_SECRET_KEY_RE = re.compile(r"(?i)(api[_-]?key|token|secret|password|passwd)\s*[=:]\s*\S+")
_SECRET_VALUE_RE = re.compile(r"\b(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b")


def scrub_secrets(text: str) -> str:
    """Redact anything that looks like a credential before it reaches a log or report."""
    text = _SECRET_KEY_RE.sub(lambda m: m.group(0).split("=")[0].split(":")[0] + "=<redacted>", text)
    return _SECRET_VALUE_RE.sub("<redacted>", text)


# ----------------------------------------------------------------------------- hashing

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- io

def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    assert_not_legacy(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping], columns: Optional[Sequence[str]] = None) -> None:
    import csv
    assert_not_legacy(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns) if columns else (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k)) for k in cols})


def write_json(path: Path, obj) -> None:
    assert_not_legacy(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float) and v != v:
        return ""
    return v


# ----------------------------------------------------------------------------- answers

def is_valid_letter(pred) -> bool:
    return isinstance(pred, str) and pred in CANONICAL


def error_type(gold: str, pred, parse_ok: bool = True) -> str:
    """Classify one prediction against canonical gold.

    Section 5.2 of the plan: erasure is a diff item answered c; wrong_group is a diff item
    answered with the other group's letter; fabrication is an equal item answered a or b;
    invalid is anything that is not a canonical letter. These are disjoint by construction.
    """
    if not parse_ok or not is_valid_letter(pred):
        return "invalid"
    if pred == gold:
        return "correct"
    if gold == EQUAL_LETTER:
        return "fabrication"          # equal item, model asserted a gap
    if pred == EQUAL_LETTER:
        return "erasure"              # diff item, model erased the gap
    return "wrong_group"              # diff item, model picked the other group


def harmonic_b(acc_diff: float, acc_equal: float) -> float:
    """B = 2ab/(a+b). NaN if either accuracy is NaN (condition absent); 0 if both are 0."""
    if acc_diff != acc_diff or acc_equal != acc_equal:
        return float("nan")
    if acc_diff + acc_equal == 0:
        return 0.0
    return 2.0 * acc_diff * acc_equal / (acc_diff + acc_equal)


@dataclass
class Counts:
    """Per-group contingency counts. Everything downstream is computed from these."""
    n_diff: int = 0
    correct_diff: int = 0
    n_equal: int = 0
    correct_equal: int = 0
    erasure: int = 0
    wrong_group: int = 0
    fabrication: int = 0
    invalid: int = 0

    def add(self, gold: str, pred, parse_ok: bool = True) -> None:
        et = error_type(gold, pred, parse_ok)
        if gold == EQUAL_LETTER:
            self.n_equal += 1
            self.correct_equal += et == "correct"
        else:
            self.n_diff += 1
            self.correct_diff += et == "correct"
        if et == "erasure":
            self.erasure += 1
        elif et == "wrong_group":
            self.wrong_group += 1
        elif et == "fabrication":
            self.fabrication += 1
        elif et == "invalid":
            self.invalid += 1

    def as_array(self) -> np.ndarray:
        return np.array([self.n_diff, self.correct_diff, self.n_equal, self.correct_equal,
                         self.erasure, self.wrong_group, self.fabrication, self.invalid], dtype=np.int64)

    @staticmethod
    def columns() -> List[str]:
        return ["n_diff", "correct_diff", "n_equal", "correct_equal",
                "erasure", "wrong_group", "fabrication", "invalid"]


def metrics_from_counts(c: np.ndarray) -> Dict[str, float]:
    """Summary metrics from a counts vector (or a summed array of them)."""
    n_diff, cd, n_eq, ce, er, wg, fab, inv = [float(x) for x in c]
    acc_d = cd / n_diff if n_diff else float("nan")
    acc_e = ce / n_eq if n_eq else float("nan")
    n = n_diff + n_eq
    return {
        "n_items": int(n),
        "n_diff": int(n_diff), "n_equal": int(n_eq),
        "accuracy_diff": acc_d, "accuracy_equal": acc_e,
        "overall_accuracy": (cd + ce) / n if n else float("nan"),
        "harmonic_b": harmonic_b(acc_d, acc_e),
        "erasure_rate_on_diff": er / n_diff if n_diff else float("nan"),
        "wrong_group_rate_on_diff": wg / n_diff if n_diff else float("nan"),
        "fabrication_rate_on_equal": fab / n_eq if n_eq else float("nan"),
        "invalid_rate": inv / n if n else float("nan"),
    }


# ----------------------------------------------------------------------------- items

_SOURCE_CELL_RE = re.compile(
    r"^(?P<edition>AgCensus\d{4}-\d{2})\s+(?P<table>T[\w-]+)\s+(?P<state>[^/]+)/(?P<size_class>[^/]+)/(?P<metric>[^/]+)/(?P<comparison>[^/]+)$")


def parse_source_cell(cell: str) -> Dict[str, str]:
    """Deterministic decomposition of a source_cell key. Returns empty fields on mismatch
    rather than guessing, so a malformed key is visible in the ledger."""
    m = _SOURCE_CELL_RE.match(cell or "")
    if not m:
        return {"edition": "", "table": "", "state": "", "size_class": "", "metric": "", "comparison": "", "parse_ok": False}
    d = m.groupdict()
    d["parse_ok"] = True
    return d


def parse_state_blind_key(key: str) -> Dict[str, str]:
    """axis|metric|size_class|comparison -> the four predeclared metadata fields."""
    parts = (key or "").split("|")
    if len(parts) != 4:
        return {"axis": "", "metric": "", "size_class": "", "comparison": "", "parse_ok": False}
    return {"axis": parts[0], "metric": parts[1], "size_class": parts[2], "comparison": parts[3], "parse_ok": True}


def load_split(name: str) -> List[Dict]:
    fname = {"train": "train_instances.jsonl", "validation": "validation_instances.jsonl",
             "test": "test_instances_frozen.jsonl"}[name]
    return read_jsonl(legacy_data_dir() / fname)


def test_items_by_id() -> Dict[str, Dict]:
    return {r["id"]: r for r in load_split("test")}


# ----------------------------------------------------------------------------- predictions

@dataclass(frozen=True, order=True)
class Arm:
    tier: str
    method: str
    seed: int

    @property
    def key(self) -> Tuple[str, str, int]:
        return (self.tier, self.method, self.seed)

    @property
    def held_out_axis(self) -> Optional[str]:
        """Leave-one-axis-out arms are evaluated only on their held-out axis, so their
        expected coverage is that axis's items, not the full test set."""
        from GPU_Run.common.paths import split_loao_method
        return split_loao_method(self.method)[1]


def expected_item_ids(arm: "Arm", items: Mapping[str, Dict]) -> set:
    axis = arm.held_out_axis
    return {i for i, it in items.items() if axis is None or it["category"] == axis}


def prediction_files() -> Dict[Arm, Path]:
    from GPU_Run.common.paths import per_item_prediction_paths
    return {Arm(t, m, s): p for (t, m, s), p in per_item_prediction_paths().items()}


def normalise_predictions(arm: Arm, path: Path, items: Mapping[str, Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    """One record per (model, method, seed, split, item_id) with cluster keys attached.

    Returns (records, issues). Issues counts duplicate ids, unknown ids, and missing ids
    so that coverage is explicit rather than silently assumed (plan section 7.2)."""
    raw = read_jsonl(path)
    seen: Dict[str, int] = {}
    records = []
    unknown = 0
    for r in raw:
        iid = r.get("id")
        if iid not in items:
            unknown += 1
            continue
        seen[iid] = seen.get(iid, 0) + 1
        if seen[iid] > 1:
            continue                      # keep the first, count the duplicate
        it = items[iid]
        pred = r.get("pred_canonical")
        parse_ok = bool(r.get("parse_ok", is_valid_letter(pred)))
        records.append({
            "tier": arm.tier, "method": arm.method, "seed": arm.seed, "split": it.get("split", "test"),
            "item_id": iid, "source_cell": it["source_cell"], "state_blind_key": it["state_blind_key"],
            "axis": it["category"], "form": it["form"], "condition": it["condition"],
            "test_slice": it.get("test_slice", ""), "gold": it["correct_answer"],
            "pred": pred if is_valid_letter(pred) else "", "parse_ok": parse_ok,
            "error_type": error_type(it["correct_answer"], pred, parse_ok),
            "provenance": str(path.name),
        })
    expected = expected_item_ids(arm, items)
    issues = {"duplicate_ids": sum(1 for v in seen.values() if v > 1), "unknown_ids": unknown,
              "missing_ids": len(expected - set(seen)), "records": len(records),
              "expected_scope": "held_out_axis" if arm.held_out_axis else "full_test"}
    return records, issues


def counts_by_cluster(records: Sequence[Dict], cluster_field: str = "source_cell",
                      clusters: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """(n_clusters x 8) counts matrix in a fixed cluster order. Clusters absent from
    `records` get a zero row so that two methods can be aligned by index."""
    order = list(clusters) if clusters is not None else sorted({r[cluster_field] for r in records})
    idx = {c: i for i, c in enumerate(order)}
    acc = [Counts() for _ in order]
    for r in records:
        acc[idx[r[cluster_field]]].add(r["gold"], r["pred"] or None, r["parse_ok"])
    return np.vstack([a.as_array() for a in acc]) if order else np.zeros((0, 8), dtype=np.int64), order


def align_pair(a: Sequence[Dict], b: Sequence[Dict]) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    """Restrict two prediction sets to their common item ids, in the same order, and report
    what was dropped from each side. Never compare different subsets silently."""
    ia = {r["item_id"]: r for r in a}
    ib = {r["item_id"]: r for r in b}
    common = sorted(set(ia) & set(ib))
    cov = {"common": len(common), "only_in_a": len(set(ia) - set(ib)), "only_in_b": len(set(ib) - set(ia))}
    return [ia[i] for i in common], [ib[i] for i in common], cov
