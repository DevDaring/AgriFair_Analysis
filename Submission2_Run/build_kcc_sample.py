"""4.1 Real farmer queries from the Kisan Call Centre (data.gov.in / AIKosh).

Input: either --csv <export.csv> (columns are normalised case-insensitively; expected fields
include state, district, crop or category, query text and the Farm Tele-Advisor answer) or
the data.gov.in API when kcc.resource_id and DATA_GOV_IN_API_KEY are set.

Output: data_submission2/kcc_queries.jsonl with n_queries rows, stratified by state x
query_type, personal data stripped, the FTA answer kept as reference_answer, and empty
question_hi / question_bn fields to be filled by --translations.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from Submission2_Run import common as C

TYPE_RULES = {   # query_type from free-text category / query, first match wins
    "plant_protection": r"pest|disease|insect|fungus|blight|rot|attack|control|spray|weed",
    "nutrient_management": r"fertili[sz]er|urea|dap|nutrient|manure|dose|npk|zinc|deficien",
    "weather": r"weather|rain|forecast|temperature|frost|humidity|monsoon",
    "government_schemes": r"scheme|subsid|pm[- ]?kisan|insurance|yojana|loan|kcc card|registration",
    "market": r"price|market|mandi|rate|sell|msp|procure",
}
_QUERY_KEYS = ("querytext", "query_text", "query", "question", "farmer_query")
_STUB = re.compile(r"^\s*(farmer|farmers?)\s+(asked|asking|enquired|wants?)\s+(about|for|regarding|information about|query on)\s*|^\s*(information about|query (on|about)|asked about|enquired about)\s*", re.I)
_LATIN = re.compile(r"[A-Za-z]")


def clean_query(q: str) -> str:
    """Turn an operator summary into a question: drop the 'FARMER ASKING ABOUT' prefix, fix case, end with '?'."""
    q = _STUB.sub("", q or "").strip(" ?.:-")
    if not q:
        return ""
    if q.isupper():
        q = q.lower(); q = q[0].upper() + q[1:]
    return q + "?"


def answer_script(a: str) -> str:
    """'latin' if the reference answer is mostly Latin script, else 'indic' (Devanagari, Bengali, Tamil ...)."""
    letters = [ch for ch in (a or "") if ch.isalpha()]
    if not letters:
        return "none"
    return "latin" if sum(1 for ch in letters if _LATIN.match(ch)) / len(letters) > 0.6 else "indic"
_ANSWER_KEYS = ("kccans", "kcc_ans", "answer", "response", "fta_answer")
_DISTRICT_KEYS = ("districtname", "district")
_DATE_KEYS = ("createdon", "created_on", "date")
_STATE_API = {"Tamil Nadu": "TAMILNADU", "West Bengal": "WEST BENGAL"}   # portal spellings that differ from the display name
_STATE_DISPLAY = {v: k for k, v in _STATE_API.items()}
_STATE_KEYS = ("statename", "state_name", "state")
_CROP_KEYS = ("crop", "crop_name")
_CAT_KEYS = ("querytype", "query_type", "category", "sector")


def _norm(row: Dict) -> Dict:
    r = {k.lower().replace(" ", "_"): str(v if v is not None else "").strip() for k, v in row.items()}
    pick = lambda keys: next((r[k] for k in keys if k in r and r[k]), "")
    return {"state": _STATE_DISPLAY.get(pick(_STATE_KEYS).upper(), pick(_STATE_KEYS).title()), "district": pick(_DISTRICT_KEYS).title(), "crop": pick(_CROP_KEYS), "category": pick(_CAT_KEYS),
            "query": pick(_QUERY_KEYS), "answer": pick(_ANSWER_KEYS), "created_on": pick(_DATE_KEYS)}


def query_type(category: str, query: str) -> str:
    text = f"{category} {query}".lower()
    for qt, pat in TYPE_RULES.items():
        if re.search(pat, text):
            return qt
    return "other"


def fetch_api(resource_id: str, api_key: str, limit: int = 5000, max_rows: int = 200000, filters: Optional[Dict] = None) -> List[Dict]:
    """Page through a resource. `filters` maps field -> value (server-side filters[field]=value)."""
    rows, offset = [], 0
    while offset < max_rows:
        params = {"api-key": api_key, "format": "json", "limit": limit, "offset": offset}
        params.update({f"filters[{k}]": v for k, v in (filters or {}).items()})
        batch = None
        for attempt in range(5):                     # the portal returns 502/504 under load; back off and retry
            try:
                r = requests.get(f"https://api.data.gov.in/resource/{resource_id}", params=params, timeout=120,
                                 headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AgriFair-research/1.0", "Accept": "application/json"})
                if r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                batch = r.json().get("records", []); break
            except (requests.RequestException, ValueError) as e:
                if attempt == 4:
                    # never re-raise the original: its message can carry the full URL including the key (requests
                    # writes "for url: ..." or "with url: ..."), so every api-key value is removed, whatever the wording
                    msg = re.sub(r"api-key=[^&\s'\"]+", "api-key=[removed]", str(e))
                    raise SystemExit(f"data.gov.in request failed after 5 attempts ({type(e).__name__}: {msg[:300]})")
                time.sleep(2 ** attempt)
        if not batch:
            break
        rows += batch; offset += limit
        if len(batch) < limit:
            break
    return rows


def stratified_sample(rows: List[Dict], cfg: Dict, seed: int) -> List[Dict]:
    k = cfg["kcc"]; states = set(k["states"]); types = list(k["query_types"])
    for r in rows:
        r["query"] = clean_query(r["query"]); r["answer_script"] = answer_script(r["answer"])
    pool = [r for r in rows if r["state"] in states and len(r["query"].split()) >= 4 and len(r["answer"].split()) >= 4]
    for r in pool:
        r["query_type"] = query_type(r["category"], r["query"])
    pool = [r for r in pool if r["query_type"] in types]
    rng = random.Random(seed)
    per_cell = max(1, k["n_queries"] // (len(states) * len(types)))
    out = []
    for s in sorted(states):
        for t in types:
            cell = [r for r in pool if r["state"] == s and r["query_type"] == t]
            rng.shuffle(cell); out += cell[:per_cell]
    rng.shuffle(out)
    return out[: k["n_queries"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, help="local KCC export")
    ap.add_argument("--translations", type=Path, help="CSV: query_id,question_hi,question_bn")
    args = ap.parse_args()
    cfg = C.load_config(); out = C.data_dir(cfg) / "kcc_queries.jsonl"
    if args.translations:
        rows = C.read_jsonl(out)
        tr = {r["query_id"]: r for r in csv.DictReader(open(args.translations, encoding="utf-8"))}
        for r in rows:
            t = tr.get(r["query_id"], {})
            r["question_hi"], r["question_bn"] = t.get("question_hi", ""), t.get("question_bn", "")
        C.write_jsonl(out, rows); print(f"translations merged into {out}"); return
    if args.csv:
        raw = [_norm(r) for r in csv.DictReader(open(args.csv, encoding="utf-8", errors="replace"))]
        source = {"kind": "csv", "path": str(args.csv), "sha256": C.sha256_file(args.csv)}
    else:
        rid, key = cfg["kcc"]["resource_id"], os.environ.get("DATA_GOV_IN_API_KEY", "")
        if not (rid and key):
            raise SystemExit("give --csv, or set kcc.resource_id in config.yaml and DATA_GOV_IN_API_KEY")
        # one server-side pull per state and year: the resource holds ~48M rows, so an unfiltered page-through is not an option
        raw = []
        for st in cfg["kcc"]["states"]:
            for yr in cfg["kcc"].get("years", [2024, 2025]):
                raw += [_norm(r) for r in fetch_api(rid, key, max_rows=cfg["kcc"].get("max_rows_per_state_year", 20000),
                                                     filters={"StateName": _STATE_API.get(st, st.upper()), "year": str(yr)})]
        source = {"kind": "data.gov.in", "resource_id": rid, "states": cfg["kcc"]["states"], "years": cfg["kcc"].get("years", [2024, 2025])}
    sample = stratified_sample(raw, cfg, cfg["analysis_seed"])
    rows = [{"query_id": f"q{i:04d}", "state": r["state"], "district": r["district"], "crop": r["crop"], "query_type": r["query_type"], "created_on": r["created_on"], "answer_script": r["answer_script"],
             "question_en": C.strip_personal_data(r["query"]), "reference_answer": C.strip_personal_data(r["answer"]),
             "question_hi": "", "question_bn": ""} for i, r in enumerate(sample)]
    C.write_jsonl(out, rows)
    C.write_json(out.with_suffix(".manifest.json"), {"source": source, "raw_rows": len(raw), "sampled": len(rows),
                                                     "by_state_type": {f"{r['state']}|{r['query_type']}": 0 for r in rows}})
    print(f"wrote {len(rows)} queries -> {out}")


if __name__ == "__main__":
    main()
