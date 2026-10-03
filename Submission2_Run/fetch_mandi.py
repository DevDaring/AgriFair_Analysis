"""Snapshot the current Agmarknet mandi prices for the study states into data_submission2/mandi/.

Run once a day for at least a week before the market-grounded advice experiment (E3b).

Files are named by the ARRIVAL DATE in the data, not by the machine's clock. The portal serves only
the current trading day, and at a morning run that is often still the previous day, so naming by the
clock would file yesterday's prices under today's date. Rows are grouped by arrival date and each
date gets its own mandi_<YYYY-MM-DD>.jsonl plus a hashed manifest.

A snapshot is replaced only by a strictly larger one for the same date. The portal fills in a day
progressively, so a later fetch of the same day is more complete, but a thin early-morning fetch
never overwrites a full snapshot already on disk.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import os

from Submission2_Run import common as C
from Submission2_Run.build_kcc_sample import fetch_api


def _iso(arrival: str) -> str:
    """'25/09/2026' -> '2026-09-25'."""
    return dt.datetime.strptime(arrival.strip(), "%d/%m/%Y").date().isoformat()


def _rows_on_disk(path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def main() -> None:
    cfg = C.load_config(); key = os.environ.get("DATA_GOV_IN_API_KEY", "")
    if not key:
        raise SystemExit("DATA_GOV_IN_API_KEY not set")
    d = C.data_dir(cfg) / "mandi"; d.mkdir(exist_ok=True)
    rows = []
    mf = cfg["mandi_fetch"]
    for st in mf["states"]:
        rows += fetch_api(mf["resource_id"], key, limit=500, max_rows=20000, filters={"state": st})
    if not rows:
        raise SystemExit("portal returned no rows")

    by_date = collections.defaultdict(list)
    for r in rows:
        by_date[_iso(r["arrival_date"])].append(r)

    notes = []
    for day, day_rows in sorted(by_date.items()):
        out = d / f"mandi_{day}.jsonl"
        before = _rows_on_disk(out)
        if before >= len(day_rows):
            notes.append(f"{day}: kept existing {before} rows (fetched only {len(day_rows)})")
            continue
        C.write_jsonl(out, day_rows)
        C.write_json(d / f"mandi_{day}.manifest.json",
                     {"date": day, "rows": len(day_rows), "states": mf["states"],
                      "resource_id": mf["resource_id"], "sha256": C.sha256_file(out),
                      "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                      "replaced_rows": before})
        notes.append(f"{day}: {len(day_rows)} rows" + (f" (replaced {before})" if before else " (new)"))
    print("; ".join(notes))


if __name__ == "__main__":
    main()
