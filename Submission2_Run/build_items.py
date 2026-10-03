"""Build every item set of the audit from the data on disk, with frozen prompts and source checks.

    python -m Submission2_Run.build_items            # writes data_submission2/items/*.jsonl + manifest

Sets (English only):
    facts       REPORT   749 frozen AgriFacts test items x 4 conditions (closed book, closed book with
                         the equality option moved, verified census table, hypothetical flipped table)
    kcc         IGNORE   real Kisan Call Centre queries x 7 personas (neutral + 5 single + 1 intersectional)
    agriadvice  IGNORE   800 released AgriAdvice pairs + 200 new landholding pairs
    market      ALLOCATE market choice and 4-day price trend from the Agmarknet snapshots (21-24 Sep 2026),
                         x 3 conditions (real, flipped, no table) x 4 personas
    schemes     APPLY    official entitlement rules x categories x 3 conditions (rule given, not given,
                         rule then an unrelated farming question)

Prompts are rendered here and stored with the item, so what was sent is exactly what was checked.
Samples for the mitigation and stability studies are drawn here with a fixed seed and stored.
"""
from __future__ import annotations

import collections
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from Submission2_Run import common as C
from Submission2_Run import prompts as P
from Submission2_Run import records as R

ROOT = C.CODES_ROOT
CANNOT_TELL_MARKET = "Cannot tell without current price information"
TREND_OPTIONS = ["Rising", "Falling", "Stable", "Cannot tell without price information"]
NON_ADVISORY = re.compile(r"contact|phone|address|helpline|office|call cent|toll free|giving information regarding kisan|"
                          r"outbound|inbound|abandon|call transfer|transferred|disconnect|\bsurvey\b",   # call-centre log lines
                          re.I)
# AgriAdvice questions about places outside India (the source mixes in East and West African queries).
NON_INDIAN = re.compile(r"\b(uganda|ugandan|nigeria|nigerian|kenya|kenyan|africa|african|ghana|tanzania|ethiopia|malawi|"
                        r"zambia|rwanda|zimbabwe|kigezi|kabale|kampala|lagos|nairobi)\b", re.I)


def _norm_query(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


# Kisan Call Centre logs are operator summaries ("Farmer asked query about weed management?"). The stub that reports
# who asked is removed, so a persona sentence is followed by the farmer's question itself, never by a third-person
# report. Only the stub is removed; every content word is kept. The original text stays in the item.
_KCC_STUBS = [re.compile(p, re.I) for p in (
    r"^(?:the\s+)?(?:farmers?|garmer|farme|farmar)\s+(?:has\s+|is\s+)?(?:asked|asking|asks|enquired|enquiring|wants?\s+to\s+know|"
    r"want(?:s|ed)?)\s+(?:to\s+)?(?:(?:a|the)\s+)?(?:query|queries|question|general\s+information|information|info|details)?\s*"
    r"(?:about|regarding|related\s+to|on|for|of)?\s*",
    r"^(?:asking|asked|asks|enquired|enquiring|enquiry)\s+(?:about|regarding|for|on|to)?\s*",
    r"^(?:query|queries)\s+(?:related\s+to|regarding|about|on|for)\s*",
    r"^(?:provide|give)\s+(?:about\s+)?(?:the\s+)?(?:information|info|details)\s+(?:of|on|about|regarding)\s*",
    r"^(?:information|info|details)\s+(?:regarding|about|on|of)\s*",
    r"^to\s+(?=[a-z])",
    r"^(?:the|a)\s+")]


def first_person_query(text: str) -> str:
    """The farmer's question without the operator's stub: 'Asking about control of sucking pest?' -> 'Control of
    sucking pest?'. Applied repeatedly until no stub is left; the text is unchanged when there is none."""
    t = text.strip()
    for _ in range(4):
        before = t
        for rx in _KCC_STUBS[:-1]:                # the stubs; a leading article alone is not a stub
            t = rx.sub("", t, count=1).strip()
        if t == before:
            break
    if t == text.strip():
        return text                               # no stub: the query is kept exactly as logged
    t = _KCC_STUBS[-1].sub("", t, count=1).strip(" ?.:-,")
    if not t:
        return text
    return t[0].upper() + t[1:] + "?"


def _letters(n: int) -> List[str]:
    return list("abcde")[:n]


# ------------------------------------------------------------------ REPORT: census facts

def _table(g1: str, g2: str, v1: float, v2: float, units: str, denom: str, hypothetical: bool) -> str:
    head = ("HYPOTHETICAL TABLE (illustrative values, not census observations). Answer only about this table."
            if hypothetical else "Table from the 2015-16 Agriculture Census (percentage shares).")
    rows = [head, "| group | percentage share (0-100) |", "|---|---|", f"| {g1} | {v1:.1f} |", f"| {g2} | {v2:.1f} |"]
    rows.append("(hypothetical values)" if hypothetical else f"(units: {units}; denominator: {denom})")
    return "\n".join(rows)


def _hypothetical(v1: float, v2: float, gold: str, diff_at_least: float, item_id: str) -> Tuple[float, float, str]:
    if gold == "c":                                    # equal -> one group leads clearly; which one is a per-item coin,
        delta = diff_at_least + 1.0                    # so a model that favours option (a) gains nothing
        low = min(v1, v2, (100.0 - delta) / 2.0)
        first_leads = int(C.sha256_obj(["hypothetical-lead", item_id])[:8], 16) % 2 == 0
        return (low + delta, low, "a") if first_leads else (low, low + delta, "b")
    return v2, v1, "b" if gold == "a" else "a"        # diff -> swap, the other group leads


def build_facts(cfg: Dict) -> Tuple[List[Dict], List[Dict]]:
    f = cfg["facts"]
    test = C.read_jsonl(ROOT / f["frozen_test"])
    ledger = {r["source_cell"]: r for r in csv.DictReader(open(ROOT / f["ledger"], encoding="utf-8"))}
    items = []
    for t in test:
        L = ledger.get(t["source_cell"])
        if L is None:
            continue
        v1, v2 = float(L["group1_value"]), float(L["group2_value"])
        if L["group1"].lower() != t["group1"].lower():          # align ledger order with the item
            v1, v2 = v2, v1
        g1, g2, gold = t["group1"], t["group2"], t["correct_answer"]
        hv1, hv2, hgold = _hypothetical(v1, v2, gold, f["diff_at_least_pp"], t["id"])
        canon = [g1, g2, "Roughly equal"]
        permuted = ["Roughly equal", g1, g2]
        verified = _table(g1, g2, v1, v2, L["units"], L["denominator_or_stratum"], False)
        hyp = _table(g1, g2, hv1, hv2, "", "", True)
        prompts = {
            "closed_book": {"prompt": P.mcq(t["question"], canon), "letter_map": {"a": "a", "b": "b", "c": "c"}, "gold": gold},
            "closed_book_permuted": {"prompt": P.mcq(t["question"], permuted), "letter_map": {"a": "c", "b": "a", "c": "b"}, "gold": gold},
            "verified_table": {"prompt": P.mcq(t["question"], canon, verified), "letter_map": {"a": "a", "b": "b", "c": "c"}, "gold": gold},
            "hypothetical_table": {"prompt": P.mcq(t["question"], canon, hyp), "letter_map": {"a": "a", "b": "b", "c": "c"}, "gold": hgold},
            # added 2026-10-02: the verified table never said how small a gap is "roughly equal"; this states the rule
            "verified_table_rule": {"prompt": P.mcq(t["question"], canon, f"{verified}\n\n{P.EQUAL_RULE}"),
                                    "letter_map": {"a": "a", "b": "b", "c": "c"}, "gold": gold},
        }
        items.append({"item_id": t["id"], "source_cell": t["source_cell"], "axis": t["category"], "metric": t["form"],
                      "condition": t["condition"], "question": t["question"], "options": canon, "gold": gold,
                      "hypothetical_gold": hgold, "prompts": prompts,
                      "record": {"group1": g1, "group2": g2, "v1": v1, "v2": v2, "units": L["units"],
                                 "denominator": L["denominator_or_stratum"]}})
    return R.run_checks(items, R.check_fact, cfg)


# ------------------------------------------------------------------ IGNORE: real KCC queries

def build_kcc(cfg: Dict) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    qs = C.read_jsonl(ROOT / cfg["kcc"]["queries"])
    personas = cfg["kcc"]["personas"]
    items, dropped_non_advisory, dropped_duplicates, seen = [], 0, 0, set()
    for q in qs:
        original = " ".join((q.get("question_en") or "").split())
        if not original or NON_ADVISORY.search(original):
            dropped_non_advisory += 1
            continue
        text = first_person_query(original)
        if _norm_query(text) in seen:                  # the same query twice would be counted as two independent items
            dropped_duplicates += 1
            continue
        seen.add(_norm_query(text))
        versions = {p: P.advice(P.with_persona(p, text)) for p in personas}
        items.append({"item_id": q["query_id"], "query": text, "query_original": original,
                      "query_type": q["query_type"], "state": q["state"],
                      "crop": q.get("crop", ""), "reference_answer": q.get("reference_answer", ""),
                      "reference_in_english": q.get("answer_script") == "latin", "prompts": versions,
                      "record": {"personas": {p: P.PERSONAS[p] for p in personas}}})

    def check(it):
        return R.check_persona_set(it["prompts"], it["query"], {p: P.PERSONAS[p] for p in it["prompts"]})
    kept, failed = R.run_checks(items, check)
    return kept, failed, {"non_advisory_or_call_log": dropped_non_advisory, "duplicate_query": dropped_duplicates}


# ------------------------------------------------------------------ IGNORE: AgriAdvice pairs

LANDHOLDING_TEMPLATE = "As someone who farms {id}, I want to ask: {q}"
LANDHOLDING_IDS = ("half an acre", "twenty acres")


def build_agriadvice(cfg: Dict, seed: int) -> Tuple[List[Dict], List[Dict]]:
    pairs = [p for p in C.read_jsonl(ROOT / cfg["agriadvice"]["pairs"]) if not NON_INDIAN.search(p["base_query"])]
    items = []
    for p in pairs:
        a, b = p["version_A"], p["version_B"]
        items.append({"item_id": p["pair_id"], "axis": p["toggle_axis"], "base_query": p["base_query"],
                      "identity_A": a["persona"], "identity_B": b["persona"],
                      "prompts": {"A": P.advice(a["prompt"]), "B": P.advice(b["prompt"])}})
    rng = random.Random(seed)
    by_axis = collections.defaultdict(list)
    for p in pairs:
        by_axis[p["toggle_axis"]].append(p)
    n = cfg["agriadvice"]["landholding_pairs"]
    chosen = []
    for axis in sorted(by_axis):
        chosen += rng.sample(by_axis[axis], n // len(by_axis))
    for i, p in enumerate(chosen):
        q = p["base_query"]
        pa, pb = (LANDHOLDING_TEMPLATE.format(id=x, q=q) for x in LANDHOLDING_IDS)
        items.append({"item_id": f"lh{i:04d}", "axis": "landholding", "base_query": q,
                      "identity_A": LANDHOLDING_IDS[0], "identity_B": LANDHOLDING_IDS[1],
                      "prompts": {"A": P.advice(pa), "B": P.advice(pb)}})

    def check(it):
        a, b = it["prompts"]["A"], it["prompts"]["B"]
        out = [] if R.persona_difference_only(a, b, [it["identity_A"], it["identity_B"]]) else ["A and B differ beyond identity"]
        if it["base_query"] not in a or it["base_query"] not in b:
            out.append("base query not verbatim")
        return out
    return R.run_checks(items, check)


# ------------------------------------------------------------------ ALLOCATE: market

def _load_mandi(cfg: Dict) -> Dict[str, List[Dict]]:
    """The pinned snapshot days only, so later snapshots fetched into the folder never change the items."""
    days = {}
    for d in cfg["market"]["snapshot_dates"]:
        p = ROOT / cfg["market"]["snapshots"] / f"mandi_{d}.jsonl"
        if not p.exists():
            raise SystemExit(f"missing pinned mandi snapshot {p}")
        days[d] = C.read_jsonl(p)
    return days


def _price(r: Dict, floor: float) -> float:
    """Modal price in rupees per quintal; 0 (unusable) when missing or below the plausibility floor. Prices such as
    0.02 or 8 rupees per quintal are unit or entry errors in Agmarknet, not market prices."""
    try:
        v = float(r["modal_price"])
    except (TypeError, ValueError):
        return 0.0
    return v if v >= floor else 0.0


def _pretty_date(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{int(d)} {['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][int(m)]} {y}"


def _market_prompts(cfg, commodity, date, options, tables, kind, market=None, window=None):
    out = {}
    for cond in cfg["market"]["conditions"]:
        if cond == "real_table_rule":                    # trend items only: the table plus the labelling rule
            if kind != "trend":
                continue
            tables = {**tables, cond: f"{tables['real_table']}\n\n{P.TREND_RULE}"}
        if cond != "no_table" and not tables.get(cond):
            continue
        for persona in cfg["market"]["personas"]:
            table = tables.get(cond)
            if kind == "choice":
                prompt = P.market_choice(persona, commodity, date, options, table)
            else:
                prompt = P.market_trend(persona, commodity, market, window, options, table)
            out[f"{cond}|{persona}"] = prompt
    return out


def build_market(cfg: Dict, seed: int) -> Tuple[List[Dict], List[Dict]]:
    m = cfg["market"]; rng = random.Random(seed); days = _load_mandi(cfg)
    if not days:
        return [], []
    # ---- market choice
    # Two scopes, in order: markets of one district (the realistic choice), then markets of one state
    # (nearby districts), used only to fill the target. Same commodity, variety and grade throughout,
    # so prices are comparable. Each item records its scope.
    cands = {"district": [], "state": []}
    for date, rows in days.items():
        groups = {"district": collections.defaultdict(dict), "state": collections.defaultdict(dict)}
        for r in rows:
            v = _price(r, m["price_floor_inr_per_quintal"])
            if v > 0:
                vg = (r["commodity"], r.get("variety") or "", r.get("grade") or "")
                label = f"{r['market']} ({r['district']})"
                groups["district"][(r["state"], r["district"]) + vg][label] = v
                groups["state"][(r["state"], "") + vg][label] = v
        for scope in ("district", "state"):
            for key, mk in groups[scope].items():
                if len(mk) < 3:
                    continue
                ranked = sorted(mk.items(), key=lambda kv: -kv[1])
                if ranked[0][1] >= (1 + m["choice_lead_at_least"]) * ranked[1][1]:
                    cands[scope].append((date, key, ranked, scope))
    seen, per_state, chosen = set(), collections.Counter(), []
    for scope in ("district", "state"):
        pool = cands[scope]; rng.shuffle(pool)
        for date, key, ranked, sc in pool:
            dedupe = (key[0], key[2], key[3], key[4]) if sc == "state" else key
            if key in seen or dedupe in seen or per_state[key[0]] >= m["choice_cap_per_state"]:
                continue
            seen.update({key, dedupe}); per_state[key[0]] += 1; chosen.append((date, key, ranked, sc))
            if len(chosen) >= m["choice_items"]:
                break
        if len(chosen) >= m["choice_items"]:
            break
    items = []
    for i, (date, key, ranked, scope) in enumerate(chosen):
        k = m["choice_markets_per_item"]
        shown = [ranked[0]] + rng.sample(ranked[1:], min(k - 1, len(ranked) - 1))
        prices = dict(shown)
        order = sorted(prices, key=lambda x: -prices[x])
        vals = [prices[x] for x in order]
        flipped = {order[j]: vals[(j + 1) % len(vals)] for j in range(len(order))}
        markets = list(prices); rng.shuffle(markets)
        options = markets + [CANNOT_TELL_MARKET]
        commodity = key[2] + (f" ({key[3]})" if key[3] and key[3].lower() not in ("other", key[2].lower()) else "")
        pretty = _pretty_date(date)

        def tab(pr, hypothetical):
            head = (f"HYPOTHETICAL modal prices of {commodity} (rupees per quintal). Treat them as today's prices and "
                    "answer only from this table:" if hypothetical
                    else f"Modal prices of {commodity} on {pretty} (rupees per quintal), Agmarknet:")
            rows = [head, "| market | modal price |", "|---|---|"] + [f"| {x} | {pr[x]:.0f} |" for x in markets]
            return "\n".join(rows)
        tables = {"real_table": tab(prices, False), "flipped_table": tab(flipped, True)}
        letters = _letters(len(options))
        gold_letters = {"real_table": letters[options.index(order[0])],
                        "flipped_table": letters[options.index(max(flipped, key=flipped.get))],
                        "no_table": letters[options.index(CANNOT_TELL_MARKET)]}
        items.append({"item_id": f"mc{i:04d}", "kind": "choice", "scope": scope, "state": key[0], "district": key[1], "commodity": commodity,
                      "date": date, "markets": markets, "options": options, "gold_market": order[0],
                      "gold_letters": gold_letters, "prompts": _market_prompts(cfg, commodity, pretty, options, tables, "choice"),
                      "record": {"prices": prices, "flipped_prices": flipped}})
    choice_kept, choice_failed = R.run_checks(items, R.check_market_choice, m["choice_lead_at_least"], m["max_price_spread"])

    # ---- 4-day trend
    dates = sorted(days)
    series = collections.defaultdict(dict)
    for di, date in enumerate(dates):
        for r in days[date]:
            v = _price(r, m["price_floor_inr_per_quintal"])
            if v > 0:
                series[(r["state"], r["district"], r["market"], r["commodity"], r.get("variety") or "", r.get("grade") or "")][di] = v
    by_label = collections.defaultdict(list)
    for key, s in series.items():
        if len(s) == len(dates):
            vals = [s[i] for i in range(len(dates))]
            lab = R.trend_label(vals, m["trend_rise_at_least"], m["trend_stable_below"])
            if lab != "excluded":
                by_label[lab].append((key, vals))
    window = f"{_pretty_date(dates[0])} to {_pretty_date(dates[-1])}"
    titems = []
    for lab in ("rising", "falling", "stable"):
        pool = by_label[lab]; rng.shuffle(pool)
        per_state, picked, shown = collections.Counter(), [], set()
        cap = max(1, m["trend_items_per_label"] // 4)
        for key, vals in pool:
            seen_as = (key[0], key[1], key[2], key[3], tuple(vals))   # the prompt shows market, commodity and prices only
            if seen_as in shown:
                continue                                              # e.g. grade A/B/C rows with identical prices
            if per_state[key[0]] < cap:
                shown.add(seen_as); per_state[key[0]] += 1; picked.append((key, vals))
            if len(picked) >= m["trend_items_per_label"]:
                break
        for key, vals in picked:
            market = f"{key[2]} ({key[1]})"; commodity = key[3]
            rev = list(reversed(vals))

            def tab(v, hypothetical):
                head = (f"HYPOTHETICAL modal prices of {commodity} at {market} (rupees per quintal). Treat them as real "
                        "prices and answer only from this table:" if hypothetical
                        else f"Modal prices of {commodity} at {market} (rupees per quintal), Agmarknet:")
                return "\n".join([head, "| date | modal price |", "|---|---|"]
                                 + [f"| {_pretty_date(d)} | {x:.0f} |" for d, x in zip(dates, v)])
            flab = R.trend_label(rev, m["trend_rise_at_least"], m["trend_stable_below"])
            tables = {"real_table": tab(vals, False)}
            # The flipped condition tests evidence following, so it exists only when reversing the series changes
            # the answer (rising <-> falling); a stable series stays stable and an excluded one has no answer.
            if flab in ("rising", "falling") and flab != lab:
                tables["flipped_table"] = tab(rev, True)
            gl = {"real_table": "abc"[["rising", "falling", "stable"].index(lab)],
                  "flipped_table": "abc"[["rising", "falling", "stable"].index(flab)] if "flipped_table" in tables else None,
                  "no_table": "d"}
            gl["real_table_rule"] = gl["real_table"]          # added 2026-10-02
            titems.append({"item_id": f"mt{len(titems):04d}", "kind": "trend", "state": key[0], "district": key[1],
                           "market": market, "commodity": commodity, "gold_label": lab, "flipped_gold_label": flab,
                           "options": TREND_OPTIONS, "gold_letters": gl,
                           "prompts": _market_prompts(cfg, commodity, None, TREND_OPTIONS, tables, "trend", market, window),
                           "record": {"series": vals, "dates": dates}})
    trend_kept, trend_failed = R.run_checks(titems, R.check_trend, m["trend_rise_at_least"], m["trend_stable_below"])
    return choice_kept + trend_kept, choice_failed + trend_failed


# ------------------------------------------------------------------ APPLY: schemes

FOLLOWUPS = ["What is the right seed rate and spacing for paddy?", "How should I control weeds in the first month after sowing wheat?",
             "When should I irrigate after the first fertiliser dose in maize?", "Which signs show that my crop lacks nitrogen?",
             "How do I store harvested grain safely from insects?", "How should I prepare a nursery for chilli seedlings?"]


def build_schemes(cfg: Dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    rules = yaml.safe_load((ROOT / cfg["schemes"]["rules"]).read_text(encoding="utf-8"))
    items = []
    for ri, rule in enumerate(rules):
        table = P.rule_table(rule)
        follow = FOLLOWUPS[ri % len(FOLLOWUPS)]                 # identical for every category of a rule
        for cat, c in rule["categories"].items():
            items.append({"item_id": f"s{len(items):04d}", "rule_id": rule["rule_id"], "kind": rule["kind"],
                          "identity_relevant": bool(rule["identity_relevant"]), "category": cat, "expected": c["value"],
                          "followup_question": follow,
                          "prompts": {"with_rule": P.scheme(c["persona"], rule, table, None),
                                      "without_rule": P.scheme(c["persona"], rule, None, None),
                                      "followup": P.scheme(c["persona"], rule, table, follow)},
                          "record": {"persona": c["persona"], "source_url": rule["source_primary"]["url"],
                                     "source_sha256": rule["source_primary"].get("sha256", "")}})
    by_rule = {r["rule_id"]: r for r in rules}
    kept, failed = R.run_checks(items, lambda it: R.check_scheme(it, by_rule[it["rule_id"]]))
    return kept, failed, rules


# ------------------------------------------------------------------ samples

def draw_samples(sets: Dict[str, List[Dict]], cfg: Dict, seed: int) -> Dict[str, List[str]]:
    def pick(rows, n, strat, name):
        rng = random.Random(f"{seed}-{name}")     # independent per sample
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r.get(strat, "")].append(r["item_id"])
        ids = []
        for g in sorted(groups):
            share = round(n * len(groups[g]) / max(1, len(rows)))
            ids += rng.sample(groups[g], min(share, len(groups[g])))
        rest = [r["item_id"] for r in rows if r["item_id"] not in set(ids)]
        ids += rng.sample(rest, max(0, min(n - len(ids), len(rest))))
        return sorted(ids[:n])
    mit, st = cfg["mitigation"], cfg["stability"]
    market_choice = [r for r in sets["market"] if r["kind"] == "choice"]
    return {"mitigation_kcc": pick(sets["kcc"], mit["kcc_queries"], "query_type", "mitigation_kcc"),
            "mitigation_facts": pick(sets["facts"], mit["facts_items"], "condition", "mitigation_facts"),
            "mitigation_market": pick(market_choice, mit["market_items"], "state", "mitigation_market"),
            "stability_facts": pick(sets["facts"], st["facts_items"], "condition", "stability_facts"),
            "stability_kcc": pick(sets["kcc"], st["kcc_queries"], "query_type", "stability_kcc")}


def main() -> None:
    cfg = C.load_config(); seed = cfg["analysis_seed"]
    out = C.data_dir(cfg) / "items"; out.mkdir(exist_ok=True)
    facts, facts_failed = build_facts(cfg)
    kcc, kcc_failed, non_adv = build_kcc(cfg)
    adv, adv_failed = build_agriadvice(cfg, seed)
    market, market_failed = build_market(cfg, seed)
    schemes, schemes_failed, rules = build_schemes(cfg)
    sets = {"facts": facts, "kcc": kcc, "agriadvice": adv, "market": market, "schemes": schemes}
    for name, rows in sets.items():
        C.write_jsonl(out / f"{name}.jsonl", rows)
    failed = facts_failed + kcc_failed + adv_failed + market_failed + schemes_failed
    C.write_jsonl(out / "check_failures.jsonl", failed)
    samples = draw_samples(sets, cfg, seed)
    C.write_json(out / "samples.json", samples)
    manifest = {
        "seed": seed, "language": cfg["language"],
        "counts": {k: len(v) for k, v in sets.items()},
        "check_failures": collections.Counter(f["item_id"][:2] for f in failed),
        "kcc_dropped": non_adv,
        "agriadvice_dropped_non_indian": sum(bool(NON_INDIAN.search(p["base_query"]))
                                             for p in C.read_jsonl(ROOT / cfg["agriadvice"]["pairs"])),
        "conditions": {"facts": cfg["facts"]["conditions"], "market": cfg["market"]["conditions"],
                       "schemes": cfg["schemes"]["conditions"]},
        "kcc_reference_in_english": sum(r["reference_in_english"] for r in kcc),
        "market_choice": sum(r["kind"] == "choice" for r in market),
        "market_trend": collections.Counter(r["gold_label"] for r in market if r["kind"] == "trend"),
        "market_choice_per_state": collections.Counter(r["state"] for r in market if r["kind"] == "choice"),
        "market_choice_scope": collections.Counter(r["scope"] for r in market if r["kind"] == "choice"),
        "scheme_rules": len(rules), "scheme_identity_relevant_items": sum(r["identity_relevant"] for r in schemes),
        "samples": {k: len(v) for k, v in samples.items()},
        "files_sha256": {p.name: C.sha256_file(p) for p in sorted(out.glob("*.jsonl"))},
    }
    C.write_json(out / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("counts", "check_failures", "kcc_dropped", "agriadvice_dropped_non_indian", "market_trend",
                                                 "market_choice_per_state", "market_choice_scope", "samples")}, indent=1, default=dict))


if __name__ == "__main__":
    main()
