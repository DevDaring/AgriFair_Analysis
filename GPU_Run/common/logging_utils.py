"""Logging, CSV writing with descriptive-name enforcement, run-metadata jsonl.

Every results CSV must use full descriptive column names with no abbreviations
so a table stays readable years later. `write_csv` enforces this.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from GPU_Run.common.paths import RESULTS_DIR, RUN_METADATA

# column names shorter than this with no underscore are treated as abbreviations
_ALLOWED_SHORT = {"id", "axis", "form", "tier", "scope", "seed", "value"}


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(RESULTS_DIR / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def _check_descriptive(columns: Sequence[str]) -> None:
    bad = []
    for c in columns:
        if c in _ALLOWED_SHORT:
            continue
        if "_" not in c and len(c) < 6:
            bad.append(c)
    if bad:
        raise ValueError(
            f"Non-descriptive CSV column names; write them out in full: {bad}"
        )


def write_csv(path: Path, rows: Sequence[Mapping], columns: Sequence[str] | None = None) -> None:
    """Write rows to CSV, enforcing descriptive column names."""
    rows = list(rows)
    if not rows:
        if columns:
            _check_descriptive(columns)
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(list(columns))
        return
    if columns is None:
        columns = list(rows[0].keys())
    _check_descriptive(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def append_csv_row(path: Path, row: Mapping, columns: Sequence[str]) -> None:
    """Append a single row, writing the header if the file is new (incremental writes)."""
    _check_descriptive(columns)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def append_jsonl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def log_run_metadata(script: str, payload: Dict) -> None:
    rec = {"script": script, "timestamp": time.time(), **payload}
    append_jsonl(RUN_METADATA, rec)
