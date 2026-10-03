"""One tiny call through every route of every model (each AWS account, each Sarvam key). Costs well under
one US cent in total. Run before a paid stage to confirm ids, credentials and routes.

    python -m Submission2_Run.probe_models --yes
"""
from __future__ import annotations

import argparse
import time

from GPU_Run.common import api_models as AM
from Submission2_Run import common as C
from Submission2_Run.providers import DISPLAY, Router

PROMPT = 'Reply with one JSON object only: {"answer_choice_letter": "a"}'


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--yes", action="store_true", help="confirm the (tiny) spend")
    a = ap.parse_args(argv)
    if not a.yes:
        raise SystemExit("pass --yes to send one short request per route")
    cfg = C.load_config(); router = Router()
    rows = []
    for tier in list(cfg["models"]) + [cfg["judge_primary"], cfg["judge_secondary"]]:
        for name, call in router._ordered_routes(AM.REGISTRY[tier], advance=False):
            t0 = time.time()
            try:
                text, i, o = call(PROMPT, 32)
                rows.append({"tier": tier, "route": name, "ok": True, "seconds": round(time.time() - t0, 2),
                             "input_tokens": i, "output_tokens": o, "reply": text.strip()[:60]})
            except Exception as e:  # report and continue: the point is to see every route
                rows.append({"tier": tier, "route": name, "ok": False, "seconds": round(time.time() - t0, 2),
                             "error": f"{type(e).__name__}: {str(e)[:120]}"})
            r = rows[-1]
            print(f"{DISPLAY.get(tier, tier):30s} {name:15s} {'OK ' if r['ok'] else 'FAIL'} {r['seconds']:5.1f}s "
                  f"{r.get('reply', r.get('error', ''))}")
    C.write_json(C.CODES_ROOT / cfg["output_directory"] / f"probe_{time.strftime('%Y%m%d_%H%M')}.json", rows)
    if not all(r["ok"] for r in rows):
        raise SystemExit("some routes failed; see above")


if __name__ == "__main__":
    main()
