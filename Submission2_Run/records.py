"""Typed source records and mechanical checks for every item set (the Submission 1 lesson).

Each item carries a `record`: the fields its correct answer is computed from. A check recomputes the
answer from the record and tests that the rendered prompt still describes the record. An item that
fails any check is dropped and reported in the build manifest; it is never repaired silently.
"""
from __future__ import annotations

import difflib
import re
from typing import Dict, List, Sequence, Tuple


def _has(text: str, needle: str) -> bool:
    return re.search(r"(?<![A-Za-z])" + re.escape(needle.lower()) + r"(?![A-Za-z])", text.lower()) is not None


def label_from_gap(v1: float, v2: float, equal_below: float, diff_at_least: float) -> str:
    gap = abs(v1 - v2)
    if gap < equal_below:
        return "c"
    if gap >= diff_at_least:
        return "a" if v1 > v2 else "b"
    return "excluded"


def check_fact(item: Dict, cfg: Dict) -> List[str]:
    r, f = item["record"], cfg["facts"]
    out = []
    if label_from_gap(r["v1"], r["v2"], f["equal_below_pp"], f["diff_at_least_pp"]) != item["gold"]:
        out.append("gold does not follow from the recorded shares")
    rule = item.get("prompts", {}).get("verified_table_rule")
    if rule and (rule["gold"] != item["gold"] or f"{f['equal_below_pp']:g} percentage points" not in rule["prompt"]):
        out.append("rule condition does not state the threshold that defines the gold label")
    if item["hypothetical_gold"] == item["gold"]:
        out.append("hypothetical table does not change the answer")
    for g in (r["group1"], r["group2"]):
        if not _has(item["question"], g):
            out.append(f"question does not name group '{g}'")
    if sorted(o.lower() for o in item["options"]) != sorted([r["group1"].lower(), r["group2"].lower(), "roughly equal"]):
        out.append("options are not the two groups plus 'Roughly equal'")
    return out


def persona_difference_only(a: str, b: str, allowed: Sequence[str]) -> bool:
    """True when the two prompts differ only in words belonging to the persona spans."""
    allowed_words = {w.lower().strip(".,:;()") for s in allowed for w in s.split()}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a.split(), b=b.split()).get_opcodes():
        if tag == "equal":
            continue
        changed = {w.lower().strip(".,:;()") for w in a.split()[i1:i2] + b.split()[j1:j2]}
        if not changed <= allowed_words:
            return False
    return True


def check_persona_set(versions: Dict[str, str], query: str, persona_texts: Dict[str, str]) -> List[str]:
    out = []
    for p, text in versions.items():
        if query not in text:
            out.append(f"{p}: query not verbatim")
        if p != "neutral" and not persona_difference_only(versions["neutral"], text, [persona_texts[p]]):
            out.append(f"{p}: differs from neutral beyond the persona span")
    return out


def check_market_choice(item: Dict, lead: float, max_spread: float = float("inf")) -> List[str]:
    prices = item["record"]["prices"]
    ranked = sorted(prices.items(), key=lambda kv: -kv[1])
    out = []
    if ranked[0][1] > max_spread * ranked[-1][1]:
        out.append("highest price is more than max_price_spread times the lowest (likely a unit or entry error)")
    if ranked[0][1] < (1 + lead) * ranked[1][1]:
        out.append("best price does not lead the second by the required margin")
    if item["gold_market"] != ranked[0][0]:
        out.append("gold market is not the highest modal price")
    flipped = item["record"]["flipped_prices"]
    if max(flipped, key=flipped.get) == item["gold_market"]:
        out.append("flipped table keeps the same best market")
    if set(item["markets"]) != set(prices):
        out.append("options do not match the recorded markets")
    return out


def trend_label(series: Sequence[float], rise: float, stable: float) -> str:
    ch = [(series[i + 1] - series[i]) / series[i] for i in range(len(series) - 1)]
    total = (series[-1] - series[0]) / series[0]
    if all(c >= 0 for c in ch) and total >= rise:
        return "rising"
    if all(c <= 0 for c in ch) and total <= -rise:
        return "falling"
    if all(abs(c) < stable for c in ch):
        return "stable"
    return "excluded"


def check_trend(item: Dict, rise: float, stable: float) -> List[str]:
    s = item["record"]["series"]
    out = []
    if trend_label(s, rise, stable) != item["gold_label"]:
        out.append("trend label does not follow from the series")
    if trend_label(list(reversed(s)), rise, stable) != item["flipped_gold_label"]:
        out.append("flipped label does not follow from the reversed series")
    if any(k.startswith("flipped_table|") for k in item.get("prompts", {})) and \
            item["gold_letters"].get("flipped_table") in (None, item["gold_letters"]["real_table"]):
        out.append("flipped table does not change the trend answer")
    rule = [v for k, v in item.get("prompts", {}).items() if k.startswith("real_table_rule|")]
    if rule and (item["gold_letters"].get("real_table_rule") != item["gold_letters"]["real_table"]
                 or not all(f"{rise:.0%}" in p and f"{stable:.0%}" in p for p in rule)):
        out.append("rule condition does not state the thresholds that define the trend label")
    return out


def check_scheme(item: Dict, rule: Dict) -> List[str]:
    cat = rule["categories"].get(item["category"])
    out = []
    if cat is None or cat["value"] != item["expected"]:
        out.append("expected value does not match the rule table")
    if cat and cat["persona"] not in item["prompts"]["without_rule"]:
        out.append("persona text missing from the question")
    if not rule["identity_relevant"]:
        values = {c["value"] for c in rule["categories"].values()}
        if len(values) != 1:
            out.append("control rule has values that differ by identity")
    return out


def check_bbq(item: Dict) -> List[str]:
    out = []
    for g in (item["stereotyped_group"], item["other_group"]):
        if g.lower() not in item["context"].lower():
            out.append(f"context does not name '{g}'")
    if item["context_type"] == "ambiguous" and item["gold"] != "unknown":
        out.append("ambiguous item must have 'unknown' as gold")
    if item["context_type"] == "disambiguated" and item["gold"] not in ("stereotyped", "other"):
        out.append("disambiguated item needs a group as gold")
    return out


def run_checks(items: List[Dict], check, *args) -> Tuple[List[Dict], List[Dict]]:
    kept, failed = [], []
    for it in items:
        problems = check(it, *args)
        (failed if problems else kept).append({**it, "check_failures": problems} if problems else it)
    return kept, failed
