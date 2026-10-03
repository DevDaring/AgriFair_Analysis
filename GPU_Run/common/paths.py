"""Canonical filesystem paths for the AgriFair study.

Repository root is the Codes/ folder. This module is imported everywhere and
creates the generated output trees (data/, results/, checkpoints/, models/)
on import so no script has to guard for their absence.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Tuple

# GPU_Run/common/paths.py -> parents[2] == Codes/
REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_on_path() -> None:
    p = str(REPO_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_on_path()

# Raw download (as shipped by Debk/AgriFair).
DATASET_DIR = REPO_ROOT / "Dataset"
AGRIFACTS_RAW = DATASET_DIR / "agrifacts.jsonl"
AGRIADVICE_RAW = DATASET_DIR / "agriadvice.jsonl"
DATASET_README = DATASET_DIR / "README.md"

# Generated trees (git-ignored, recreated by code).
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
GEOMETRY_DIR = RESULTS_DIR / "geometry"
DRY_RUN_DIR = RESULTS_DIR / "dry_run"
EVAL_CACHE_DIR = RESULTS_DIR / "evaluation_cache"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
MODELS_DIR = REPO_ROOT / "models"
EXTERNAL_WANG_DIR = DATA_DIR / "external_wang"

# Derived data files (written by Dataset_Prep).
TEMPLATED_ALL = DATA_DIR / "templated_all_instances.jsonl"
TRAIN_INSTANCES = DATA_DIR / "train_instances.jsonl"
VALIDATION_INSTANCES = DATA_DIR / "validation_instances.jsonl"
TEST_INSTANCES_FROZEN = DATA_DIR / "test_instances_frozen.jsonl"
COUNTERFACTUAL_PAIRS = DATA_DIR / "counterfactual_pairs.jsonl"
AGRIADVICE_PAIRS = DATA_DIR / "agriadvice_pairs.jsonl"
GENERAL_REPLAY = DATA_DIR / "general_replay.jsonl"
EXTERNAL_CAPABILITY_PROBE = DATA_DIR / "external_capability_probe.jsonl"
DATASET_REVISIONS = DATA_DIR / "dataset_revisions.json"
MODEL_REVISIONS = MODELS_DIR / "model_revisions.json"

# Key result files.
FROZEN_TEST_SHA256 = RESULTS_DIR / "frozen_test_set_sha256.txt"
DATA_HYGIENE_LOG = RESULTS_DIR / "data_hygiene_log.csv"
CONTAMINATION_REPORT = RESULTS_DIR / "contamination_report.csv"
RUN_METADATA = RESULTS_DIR / "run_metadata.jsonl"
MAIN_EVALUATION = RESULTS_DIR / "main_evaluation_results.csv"
MATCHED_BUDGET_REFERENCE = RESULTS_DIR / "matched_budget_reference.json"

# Leave-one-axis-out arms are named <base_method>_loao_<axis>.
LOAO_INFIX = "_loao_"

_PREDICTION_FILENAME = re.compile(
    r"^per_item_predictions_(?P<tier>.+?)_(?P<method>.+)_seed(?P<random_seed>\d+)\.jsonl$"
)


def split_loao_method(method: str) -> Tuple[str, Optional[str]]:
    """Return (base_method, held_out_axis). held_out_axis is None for full-mixture arms."""
    if LOAO_INFIX in method:
        base, _, axis = method.partition(LOAO_INFIX)
        return base, axis
    return method, None


def parse_per_item_prediction_path(path):
    """Split a per-item prediction filename into (tier, method, random_seed).

    Method names contain underscores (baseline_igu_lora, reference_vanilla_qlora) while
    tier names do not (smoke, small-instruct, broad-instruct), so the tier match is
    non-greedy and the method takes the remainder. Returns None on a filename that
    does not match."""
    m = _PREDICTION_FILENAME.match(Path(path).name)
    if not m:
        return None
    return m.group("tier"), m.group("method"), int(m.group("random_seed"))


def per_item_prediction_paths():
    """Every per-item prediction file present, as {(tier, method, seed): Path}."""
    out = {}
    for p in sorted(RESULTS_DIR.glob("per_item_predictions_*.jsonl")):
        parsed = parse_per_item_prediction_path(p)
        if parsed is not None:
            out[parsed] = p
    return out


def attribution_path(label: str, held_out_axis: Optional[str] = None) -> Path:
    suffix = f"{LOAO_INFIX}{held_out_axis}" if held_out_axis else ""
    return RESULTS_DIR / f"attribution_{label}{suffix}.json"


def lora_config_path(label: str, held_out_axis: Optional[str] = None) -> Path:
    suffix = f"{LOAO_INFIX}{held_out_axis}" if held_out_axis else ""
    return RESULTS_DIR / f"lora_config_{label}{suffix}.json"


_OUTPUT_TREES = [
    DATA_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    GEOMETRY_DIR,
    DRY_RUN_DIR,
    EVAL_CACHE_DIR,
    CHECKPOINTS_DIR,
    MODELS_DIR,
    EXTERNAL_WANG_DIR,
]

for _d in _OUTPUT_TREES:
    _d.mkdir(parents=True, exist_ok=True)
