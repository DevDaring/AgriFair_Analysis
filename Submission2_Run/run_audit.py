"""Send every frozen prompt to every audited model, with caching, a spend cap and a thread pool.

    python -m Submission2_Run.run_audit --stage facts|kcc|agriadvice|market|schemes|mitigation|related|stability|all
                                        [--repeat N] [--models a,b] [--smoke]

--smoke     4 items per set through the offline FakeRouter into results_submission2_SMOKE; never spends.
--pilot     4 items per set through the REAL models into results_submission2_PILOT, with a 1 USD cap; checks
            that every model follows the answer formats before the full run (works while api_enabled is false).
--repeat    which repeat this run is. Repeat 1 is the main run; `--stage stability --repeat 2` on a later
            day re-asks the stability panel so day-to-day change can be measured.

Each completed prompt becomes one row in results_submission2[/_SMOKE]/raw/<stage>_r<repeat>.jsonl with the
model, route (which AWS account or Sarvam key served it), model id, tokens, latency, cost and raw text.
Rows are rebuilt from the cache on every run, so an interrupted run resumes without re-spending.
"""
from __future__ import annotations

import argparse
import collections
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Submission2_Run import common as C
from Submission2_Run import prompts as P
from Submission2_Run import repair as RP
from Submission2_Run.cache import ResponseCache
from GPU_Run.common import api_models as AM
from Submission2_Run.providers import FakeRouter, Router

STAGES = ["facts", "kcc", "agriadvice", "market", "schemes", "related", "mitigation", "stability"]


class _Pools:
    """Several executors behind the one shutdown() call the execute loop uses."""
    def __init__(self, pools: Dict):
        self.pools = pools

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        for ex in self.pools.values():
            ex.shutdown(wait=wait, cancel_futures=cancel_futures)


class BudgetExceeded(Exception):
    """The running spend passed budget_usd_cap. Stops the run; the cache keeps every paid answer."""


class Runner:
    def __init__(self, cfg: Dict, smoke: bool, models: List[str], repeat: int = 1, pilot: bool = False):
        self.cfg, self.smoke, self.pilot, self.models, self.repeat = cfg, smoke, pilot, models, repeat
        self.out = C.CODES_ROOT / (cfg["output_directory"] + ("_SMOKE" if smoke else "_PILOT" if pilot else ""))
        (self.out / "raw").mkdir(parents=True, exist_ok=True)
        self.items_dir = C.data_dir(cfg) / "items"
        self.cache = ResponseCache(self.out / "cache" / "responses.sqlite")
        if smoke:
            self.router = FakeRouter()
        elif pilot:
            self.router = Router(use_openrouter=bool(cfg.get("openrouter_fallback", True))); self.cfg = {**cfg, "budget_usd_cap": 1.0}
        else:
            if not cfg["api_enabled"]:
                raise SystemExit("api_enabled is false in config.yaml; flip it deliberately before a paid run")
            self.router = Router(use_openrouter=bool(cfg.get("openrouter_fallback", True)))
        self._lock = threading.Lock()
        # The cap covers the whole run: spend already in this cache (earlier processes, other stages) counts.
        # Smoke never spends; the pilot keeps its own 1 USD cap per process.
        self.spent_before_usd = 0.0 if (smoke or pilot) else self.cache.priced_total(lambda t, i, o, r: C.cost_usd(cfg, t, i, o, r))
        self.spent_usd = 0.0
        # Responses the code cannot parse go to this extractor (never an audited model); see repair.py.
        self.extractor = cfg.get("parse_repair_model") or cfg["judge_secondary"]
        self._sem = {t: threading.Semaphore(cfg["concurrency"].get(t, cfg["concurrency"]["default"]))
                     for t in set(models) | {cfg["judge_primary"], cfg["judge_secondary"], self.extractor}}

    # ---------------------------------------------------------------- items
    def items(self, name: str) -> List[Dict]:
        rows = C.read_jsonl(self.items_dir / f"{name}.jsonl")
        return rows[: self.cfg["smoke"]["items_per_set"]] if (self.smoke or self.pilot) else rows

    def sample(self, name: str) -> set:
        return set(C.read_json(self.items_dir / "samples.json")[name])

    @staticmethod
    def _model_ids(tier: str) -> set:
        s = AM.REGISTRY[tier]
        return {x for x in (s.bedrock_model_id, s.native_model_id, s.openrouter_model_id) if x}

    # ---------------------------------------------------------------- one call, cached
    def call(self, tier: str, prompt: str, max_tokens: int, temperature: float = 0.0) -> Dict:
        max_tokens = max(max_tokens, int((self.cfg.get("min_tokens_by_model") or {}).get(tier, 0)))
        key = C.prompt_key(tier, prompt, max_tokens, temperature, self.repeat)
        hit = self.cache.get(key)
        if hit and not self.smoke and hit["model_id"] not in self._model_ids(tier):
            hit = None                          # the config now names a different model version: ask again
        if hit:
            return {"text": hit["text"], "input_tokens": hit["input_tokens"], "output_tokens": hit["output_tokens"],
                    "cost_usd": C.cost_usd(self.cfg, tier, hit["input_tokens"], hit["output_tokens"], hit["route"]),
                    "latency_seconds": hit["latency_seconds"], "route": hit["route"], "model_id": hit["model_id"],
                    "ok": True, "cached": True, "truncated": int(hit["output_tokens"] or 0) >= max_tokens}
        # The route chain itself never retries (each route is tried once). When every route fails, which in
        # practice is Bedrock throttling both accounts at once, back off and try the chain again.
        sem = self._sem.setdefault(tier, threading.Semaphore(self.cfg["concurrency"]["default"]))
        attempts = int(self.cfg.get("chain_retries", 6)) + 1
        for attempt in range(attempts):
            with sem:
                r = self.router.ask(tier, prompt, max_tokens, temperature)
            if r.ok or self.smoke or attempt == attempts - 1:
                break
            time.sleep(min(60.0, 2.0 ** attempt) + random.random())
        usd = 0.0 if self.smoke else C.cost_usd(self.cfg, tier, r.input_tokens, r.output_tokens, r.route)
        with self._lock:
            self.spent_usd += usd
            over = self.spent_before_usd + self.spent_usd > float(self.cfg["budget_usd_cap"])
        if r.ok:
            self.cache.put(key, tier, "", self.repeat, r.route, r.model_id, r.text, r.input_tokens, r.output_tokens,
                           r.latency_seconds)
        if over:
            raise BudgetExceeded(f"budget cap {self.cfg['budget_usd_cap']} USD reached; the cache is intact")
        return {"text": r.text, "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                "cost_usd": C.cost_usd(self.cfg, tier, r.input_tokens, r.output_tokens, r.route),
                "latency_seconds": r.latency_seconds, "route": r.route, "model_id": r.model_id, "ok": r.ok, "cached": False,
                "truncated": r.ok and int(r.output_tokens or 0) >= max_tokens}     # the reply used its whole budget

    # ---------------------------------------------------------------- one recipe (one or two calls)
    def recipe(self, tier: str, method: str, prompt: str, max_tokens: int) -> Dict:
        reason = int(self.cfg["reasoning_max_tokens"])
        calls, restored = [], []
        if method == "plain":
            calls.append(self.call(tier, prompt, max_tokens))
        elif method == "self_awareness":
            calls.append(self.call(tier, f"{prompt}\n\n{P.SELF_AWARENESS}", max_tokens))
        elif method == "cot_debias":
            calls.append(self.call(tier, f"{prompt}\n\n{P.COT_DEBIAS}", reason))
        elif method == "self_reflection":
            first = self.call(tier, prompt, max_tokens); calls.append(first)
            if not first["ok"]:
                return self._recipe_result(calls, restored)                       # failed call: row marked failed
            if not first["text"].strip():                                         # empty (billed) answer: nothing to
                return self._recipe_result(calls, restored, expected_calls=1)     # reflect on; parsed as unrecoverable
            calls.append(self.call(tier, f"{prompt}\n\nYour answer:\n{first['text']}\n\n{P.SELF_REFLECTION}", max_tokens))
        elif method == "self_help":
            rw = self.call(tier, P.SELF_HELP.format(query=prompt), reason); calls.append(rw)
            if not rw["ok"]:
                return self._recipe_result(calls, restored)
            system, query = P.parse_self_help(rw["text"], prompt)
            query, restored = P.restore_task_parts(prompt, query)   # options / format the rewrite dropped
            calls.append(self.call(tier, f"{system}\n\n{query}".strip(), max_tokens))
        else:
            raise ValueError(method)
        return self._recipe_result(calls, restored, expected_calls=2 if method in ("self_reflection", "self_help") else 1)

    @staticmethod
    def _recipe_result(calls: List[Dict], restored: List[str], expected_calls: int = 2) -> Dict:
        last = calls[-1]
        return {"restored": ",".join(restored), "text": last["text"], "route": ";".join(sorted({c["route"] for c in calls})),
                "model_id": last["model_id"],
                "input_tokens": sum(c["input_tokens"] for c in calls), "output_tokens": sum(c["output_tokens"] for c in calls),
                "cost_usd": sum(c["cost_usd"] for c in calls),
                "latency_seconds": round(sum(c["latency_seconds"] for c in calls), 3), "calls": len(calls),
                "ok": all(c["ok"] for c in calls) and len(calls) == expected_calls, "cached": all(c["cached"] for c in calls),
                "truncated": bool(last.get("truncated")), "first_text": calls[0]["text"] if len(calls) > 1 else ""}

    # ---------------------------------------------------------------- task execution
    def answer(self, t: Dict) -> Tuple[Dict, Dict]:
        """One task: the recipe, then the parse step (deterministic, else the extractor, verified)."""
        r = self.recipe(t["tier"], t["method"], t["prompt"], t["max_tokens"])
        try:
            parsed = RP.resolve(self, t["meta"], t["prompt"], r["text"] if r["ok"] else "", self.extractor,
                                truncated=r.get("truncated", False))
        except BudgetExceeded:
            raise
        except Exception as e:                  # a parse-step error must not lose the paid answer
            parsed = {"parse_kind": RP.kind_of(t["meta"]), "parse_method": "failed", "parsed_text": "{}",
                      "repair_cost_usd": 0.0, "parse_error": type(e).__name__}
        return r, parsed

    @staticmethod
    def _error_result(e: Exception) -> Tuple[Dict, Dict]:
        r = {"text": "", "route": "none", "model_id": "", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
             "latency_seconds": 0.0, "calls": 0, "ok": False, "cached": False, "first_text": "", "restored": "", "truncated": False,
             "error": f"{type(e).__name__}: {str(e)[:120]}"}
        return r, {"parse_kind": None, "parse_method": "failed", "parsed_text": "{}", "repair_cost_usd": 0.0}

    def execute(self, stage: str, tasks: List[Dict]) -> List[Dict]:
        rows, done, errors, t0 = [], 0, 0, time.time()
        # One worker pool per model: a model that is throttled (its tasks sleeping between retries) can never take
        # the worker threads another model needs. Each pool has twice the model's concurrency, so retries waiting
        # out a back-off do not idle that model's own request slots.
        with ExitStack() as stack:
            pools = {tier: stack.enter_context(ThreadPoolExecutor(
                max_workers=2 * self.cfg["concurrency"].get(tier, self.cfg["concurrency"]["default"])))
                for tier in sorted({t["tier"] for t in tasks})}
            ex = _Pools(pools)
            futs = {pools[t["tier"]].submit(self.answer, t): t for t in tasks}
            try:
                for fut in as_completed(futs):
                    t = futs[fut]
                    try:
                        r, parsed = fut.result()
                    except BudgetExceeded as e:
                        ex.shutdown(wait=False, cancel_futures=True)
                        self._write(stage, rows); raise SystemExit(str(e))
                    except Exception as e:          # e.g. a cache write that kept failing: logged as a failed row
                        r, parsed = self._error_result(e)
                        parsed["parse_kind"] = RP.kind_of(t["meta"]); errors += 1
                    rows.append({**t["meta"], "stage": stage, "tier": t["tier"], "method": t["method"], "repeat": self.repeat,
                                 "route": r["route"], "model_id": r["model_id"], "input_tokens": r["input_tokens"],
                                 "output_tokens": r["output_tokens"], "latency_seconds": r["latency_seconds"], "calls": r["calls"],
                                 "cost_usd": round(r["cost_usd"], 7), "cached": r["cached"], "failed": not r["ok"],
                                 "truncated": r.get("truncated", False), "raw_text": r["text"], "first_text": r["first_text"],
                                 "self_help_restored": r.get("restored", ""),
                                 **({"error": r["error"]} if r.get("error") else {}), **parsed})
                    done += 1
                    if done % 500 == 0 or done == len(tasks):
                        print(f"  {stage}: {done}/{len(tasks)}  spent this run ${self.spent_usd:.2f}  {time.time() - t0:.0f}s", flush=True)
            except KeyboardInterrupt:               # Ctrl-C: cancel every queued (unpaid) task, keep what finished
                ex.shutdown(wait=False, cancel_futures=True)
                self._write(stage, rows)
                raise
        self._write(stage, rows)
        methods = collections.Counter(r["parse_method"] for r in rows if r["parse_method"] != "not_applicable")
        print(f"  {stage}: parsing {dict(methods)}; calls failed after all retries {sum(r['failed'] for r in rows)}; "
              f"cut off at the token limit {sum(r['truncated'] for r in rows)}; task errors {errors}", flush=True)
        return rows

    def _write(self, stage: str, rows: List[Dict]) -> None:
        path = self.out / "raw" / f"{stage}_r{self.repeat}.jsonl"
        if set(self.models) < set(self.cfg["models"]) and path.exists():
            # --models ran a subset of the audited models: keep every other model's rows
            rows = rows + [r for r in C.read_jsonl(path) if r["tier"] not in self.models]
        rows.sort(key=lambda r: (r["tier"], r.get("unit_id", ""), r["method"]))
        C.write_jsonl(path, rows)

    def task(self, tier, method, prompt, max_tokens, **meta) -> Dict:
        return {"tier": tier, "method": method, "prompt": prompt, "max_tokens": max_tokens, "meta": meta}

    # ---------------------------------------------------------------- stages
    def stage_facts(self, subset: Optional[set] = None, conditions=None, methods=("plain",)) -> List[Dict]:
        mt = self.cfg["answer_max_tokens"]; tasks = []
        for it in self.items("facts"):
            if subset is not None and it["item_id"] not in subset:
                continue
            for cond in conditions or self.cfg["facts"]["conditions"]:
                p = it["prompts"][cond]
                for tier in self.models:
                    for m in methods:
                        tasks.append(self.task(tier, m, p["prompt"], mt, unit_id=f"{it['item_id']}|{cond}", set="facts",
                                               item_id=it["item_id"], condition=cond, gold=p["gold"], letter_map=p["letter_map"],
                                               axis=it["axis"], item_condition=it["condition"], source_cell=it["source_cell"]))
        return tasks

    def stage_kcc(self, subset: Optional[set] = None, personas=None, methods=("plain",)) -> List[Dict]:
        mt = self.cfg["advice_max_tokens"]; tasks = []
        for it in self.items("kcc"):
            if subset is not None and it["item_id"] not in subset:
                continue
            for persona in personas or self.cfg["kcc"]["personas"]:
                for tier in self.models:
                    for m in methods:
                        tasks.append(self.task(tier, m, it["prompts"][persona], mt, unit_id=f"{it['item_id']}|{persona}",
                                               set="kcc", item_id=it["item_id"], persona=persona, query_type=it["query_type"],
                                               state=it["state"]))
        return tasks

    def stage_agriadvice(self) -> List[Dict]:
        mt = self.cfg["advice_max_tokens"]
        return [self.task(tier, "plain", it["prompts"][side], mt, unit_id=f"{it['item_id']}|{side}", set="agriadvice",
                          item_id=it["item_id"], side=side, axis=it["axis"], identity=it[f"identity_{side}"])
                for it in self.items("agriadvice") for side in ("A", "B") for tier in self.models]

    def stage_market(self, subset: Optional[set] = None, conditions=None, methods=("plain",), personas=None) -> List[Dict]:
        mt = self.cfg["answer_max_tokens"]; tasks = []
        for it in self.items("market"):
            if subset is not None and it["item_id"] not in subset:
                continue
            for key, prompt in it["prompts"].items():
                cond, persona = key.split("|")
                if (conditions and cond not in conditions) or (personas and persona not in personas):
                    continue
                for tier in self.models:
                    for m in methods:
                        tasks.append(self.task(tier, m, prompt, mt, unit_id=f"{it['item_id']}|{key}", set="market",
                                               item_id=it["item_id"], kind=it["kind"], condition=cond, persona=persona,
                                               gold=it["gold_letters"].get(cond), n_options=len(it["options"]),
                                               state=it["state"], scope=it.get("scope", "")))
        return tasks

    def stage_schemes(self) -> List[Dict]:
        tasks = []
        for it in self.items("schemes"):
            for cond, prompt in it["prompts"].items():
                mt = self.cfg["advice_max_tokens"] if cond == "followup" else 120
                for tier in self.models:
                    tasks.append(self.task(tier, "plain", prompt, mt, unit_id=f"{it['item_id']}|{cond}", set="schemes",
                                           item_id=it["item_id"], rule_id=it["rule_id"], kind=it["kind"], category=it["category"],
                                           identity_relevant=it["identity_relevant"], expected=it["expected"], condition=cond))
        return tasks

    def stage_related(self) -> List[Dict]:
        mt = self.cfg["answer_max_tokens"]; tasks = []
        for it in self.items("related_bbq"):
            for tier in self.models:
                tasks.append(self.task(tier, "plain", it["prompt"], mt, unit_id=it["item_id"], set="related_bbq",
                                       **{k: it[k] for k in ("item_id", "axis", "template", "order", "context_type", "direction",
                                                             "polarity", "gold", "letter_map")}))
        for it in self.items("related_wat"):
            for tier in self.models:
                tasks.append(self.task(tier, "plain", it["prompt"], 400, unit_id=it["item_id"], set="related_wat",
                                       **{k: it[k] for k in ("item_id", "axis", "label_a", "label_b", "Xa", "Xb")}))
        for it in self.items("related_rdt"):
            for tier in self.models:
                tasks.append(self.task(tier, "plain", it["prompt"], 250, unit_id=it["item_id"], set="related_rdt",
                                       item_id=it["item_id"], axis=it["axis"], marginalised_is=it["marginalised_is"]))
        for it in self.items("related_numeric"):
            for tier in self.models:
                tasks.append(self.task(tier, "plain", it["prompt"], mt, unit_id=it["unit_id"], set="related_numeric",
                                       **{k: it[k] for k in ("item_id", "scenario", "context", "persona", "anchor_price")}))
        return tasks

    def stage_mitigation(self) -> List[Dict]:
        """BiasFreeBench's four prompting methods on statistical samples of three cells."""
        methods = tuple(self.cfg["mitigation"]["methods"])
        sub = (lambda name: None) if (self.smoke or self.pilot) else self.sample
        # every persona INCLUDING the placebos: a method's effect on identity is read against its effect on the
        # placebo, exactly as for the plain prompt
        return (self.stage_kcc(sub("mitigation_kcc"), methods=methods)
                + self.stage_facts(sub("mitigation_facts"), conditions=["closed_book"], methods=methods)
                + self.stage_market(sub("mitigation_market"), conditions=["real_table", "no_table"], methods=methods))

    def stage_stability(self) -> List[Dict]:
        sub = (lambda name: None) if (self.smoke or self.pilot) else self.sample
        return (self.stage_facts(sub("stability_facts"), conditions=["closed_book"])
                + self.stage_kcc(sub("stability_kcc"), personas=["neutral"]))

    def run(self, stage: str) -> List[Dict]:
        tasks = getattr(self, f"stage_{stage}")()
        print(f"{stage}: {len(tasks)} recipes over {len(self.models)} models", flush=True)
        return self.execute(stage, tasks)

    def finish(self, stages: List[str]) -> None:
        C.write_json(self.out / "raw" / f"run_manifest_{'_'.join(stages)}_r{self.repeat}.json",
                     {"stages": stages, "models": self.models, "repeat": self.repeat, "smoke": self.smoke, "pilot": self.pilot,
                      "cached_responses": self.cache.count(), "spent_usd_this_run": round(self.spent_usd, 4),
                      "tariff_date": self.cfg["tariffs"]["tariff_date"],
                      "route_counts": dict(getattr(self.router, "route_counts", {})),
                      "route_failures": dict(getattr(self.router, "failure_counts", {})),
                      "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=STAGES + ["all"])
    ap.add_argument("--repeat", type=int, default=1); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--models")
    a = ap.parse_args(argv); cfg = C.load_config()
    models = [m.strip() for m in a.models.split(",")] if a.models else list(cfg["models"])
    stages = [s for s in STAGES if s != "stability"] if a.stage == "all" else [a.stage]
    if a.repeat > 1 and stages != ["stability"]:
        sys.exit("repeats beyond 1 are for --stage stability only")
    run = Runner(cfg, a.smoke, models, a.repeat, pilot=a.pilot)
    for s in stages:
        run.run(s)
    run.finish(stages)


if __name__ == "__main__":
    main()
