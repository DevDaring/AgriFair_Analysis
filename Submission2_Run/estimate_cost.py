"""Dry-run cost estimate: builds every task of every stage exactly as run_audit would, sends nothing.

    python -m Submission2_Run.estimate_cost

Input tokens come from the real prompt lengths (characters / 3.6). Output tokens are assumed per answer
type (below) and should be replaced by measured means after the first stage. Bedrock spend is split over
the two AWS accounts and Sarvam spend over the two keys, because the router alternates between them.
"""
from __future__ import annotations

import collections
from typing import Dict

from Submission2_Run import common as C
from Submission2_Run.run_audit import Runner

CHARS_PER_TOKEN = 3.6
OUTPUT_TOKENS = {"answer": 15, "advice": 260, "wat": 140, "rdt": 120, "scheme": 40, "followup": 140}
EXTRA_CALLS = {"plain": (1, 1.0), "self_awareness": (1, 1.0), "cot_debias": (1, 2.2),
               "self_reflection": (2, 1.0), "self_help": (2, 1.0)}      # calls, output multiplier
JUDGE = {"pairwise": (1000, 140), "coverage": (700, 25), "bfs": (500, 21)}     # pilot means, 2026-10-01   # tokens in, out per call
GST = 0.18


class DryRunner(Runner):
    def __init__(self, cfg: Dict):                      # no router, no cache: nothing can be sent
        self.cfg, self.smoke, self.pilot, self.models, self.repeat = cfg, False, False, list(cfg["models"]), 1
        self.items_dir = C.data_dir(cfg) / "items"


def _kind(t: Dict) -> str:
    s = t["meta"].get("set", "")
    if s in ("kcc", "agriadvice"):
        return "advice"
    if s == "related_wat":
        return "wat"
    if s == "related_rdt":
        return "rdt"
    if s == "schemes":
        return "followup" if t["meta"]["condition"] == "followup" else "scheme"
    return "answer"


def estimate(cfg: Dict) -> Dict:
    run = DryRunner(cfg)
    per_tier = collections.defaultdict(lambda: collections.Counter())
    advice_answers = collections.Counter()
    for stage in ("facts", "kcc", "agriadvice", "market", "schemes", "related", "mitigation", "stability"):
        for t in getattr(run, f"stage_{stage}")():
            calls, mult = EXTRA_CALLS[t["method"]]
            tin = len(t["prompt"]) / CHARS_PER_TOKEN
            tout = OUTPUT_TOKENS[_kind(t)] * mult
            if t["method"] == "self_reflection":
                tin = tin * 2 + tout                           # second call repeats the prompt and the first answer
            if t["method"] == "self_help":
                tin = tin * 2.3; tout += tin / 2.3              # rewrite call outputs roughly the prompt again
            c = per_tier[t["tier"]]
            c["calls"] += calls; c["input_tokens"] += tin; c["output_tokens"] += tout * (1 if calls == 1 else 1.6)
            c[f"calls_{stage}"] += calls
            if _kind(t) == "advice" and t["meta"].get("set") == "kcc" and t["meta"].get("persona") != "neutral":
                advice_answers[t["tier"]] += 1
    # judge calls (primary + secondary share)
    pairs = sum(per_tier[t]["calls_kcc"] for t in cfg["models"]) * 6 / 7 + sum(per_tier[t]["calls_agriadvice"] for t in cfg["models"]) / 2 \
        + sum(per_tier[t]["calls_mitigation"] for t in cfg["models"]) * 0.4
    j = per_tier[cfg["judge_primary"]]
    kcc_items = len(C.read_jsonl(run.items_dir / "kcc.jsonl"))
    english_refs = sum(r["reference_in_english"] for r in C.read_jsonl(run.items_dir / "kcc.jsonl"))
    cov = english_refs * len(cfg["kcc"]["personas"]) * len(cfg["models"])
    bfs = sum(advice_answers.values())
    for name, n in (("pairwise", pairs), ("coverage", cov), ("bfs", bfs)):
        j["calls"] += n; j["input_tokens"] += n * JUDGE[name][0]; j["output_tokens"] += n * JUDGE[name][1]
    s = per_tier[cfg["judge_secondary"]]
    n2 = pairs * float(cfg["judge_secondary_share"])
    s["calls"] += n2; s["input_tokens"] += n2 * JUDGE["pairwise"][0]; s["output_tokens"] += n2 * JUDGE["pairwise"][1]
    rows, by_acct = [], collections.defaultdict(float)
    inr = float(cfg["tariffs"]["usd_inr"])
    for tier, c in per_tier.items():
        usd = C.cost_usd(cfg, tier, c["input_tokens"], c["output_tokens"])
        accts = ("sarvam-key-1", "sarvam-key-2") if tier.startswith("sarvam") else ("aws-account-1", "aws-account-2")
        for a in accts:
            by_acct[a] += usd / 2
        rows.append({"tier": tier, "calls": int(c["calls"]), "input_Mtok": round(c["input_tokens"] / 1e6, 2),
                     "output_Mtok": round(c["output_tokens"] / 1e6, 2), "usd": round(usd, 2), "inr": round(usd * inr),
                     "provider": "Sarvam" if tier.startswith("sarvam") else "AWS Bedrock"})
    total = sum(r["usd"] for r in rows)
    return {"rows": rows, "by_account_usd": {k: round(v, 2) for k, v in sorted(by_acct.items())},
            "total_usd": round(total, 2), "total_inr": round(total * inr), "total_inr_with_gst": round(total * inr * (1 + GST)),
            "kcc_items": kcc_items, "english_reference_items": english_refs}


def main() -> None:
    cfg = C.load_config(); e = estimate(cfg)
    print(f"{'model':24s} {'provider':12s} {'calls':>8s} {'in Mtok':>8s} {'out Mtok':>8s} {'USD':>8s} {'INR':>8s}")
    for r in sorted(e["rows"], key=lambda r: -r["usd"]):
        print(f"{r['tier']:24s} {r['provider']:12s} {r['calls']:8d} {r['input_Mtok']:8.2f} {r['output_Mtok']:8.2f} {r['usd']:8.2f} {r['inr']:8d}")
    print("per account / key (USD):", e["by_account_usd"])
    print(f"total USD {e['total_usd']}  INR {e['total_inr']}  INR incl. 18% GST {e['total_inr_with_gst']}")
    C.write_json(C.CODES_ROOT / cfg["output_directory"] / "cost_estimate.json", e)


if __name__ == "__main__":
    main()
