#!/usr/bin/env bash
# Submission 2, stability repeat 2 (one-off). Cron starts it on 3 Oct 2026 at 11:00 IST; it removes its own cron line.
# Re-asks the stability sample (150 facts + 150 KCC neutral queries per model) on a later day than the main run, then
# judges the day-1-vs-day-2 pairs (noise floor), fills any throttled judge calls, and re-runs the analysis.
cd /home/Debz/Research/Finetune_AgriFair/Codes || exit 1
PY=/home/Debz/Research/Finetune_AgriFair/.venv/bin/python
OUT=results_submission2
[ "$(TZ=Asia/Kolkata date +%F)" = "2026-10-03" ] || exit 0          # acts only on 3 Oct 2026
[ -f "$OUT/STABILITY_DONE" ] && exit 0
( crontab -l 2>/dev/null | grep -v "run_stability.sh" ) | crontab -     # one-off: remove this job's cron line
{
  echo "start $(TZ=Asia/Kolkata date)"
  $PY -m Submission2_Run.run_audit --stage stability --repeat 2 \
    && $PY -m Submission2_Run.judge && $PY -m Submission2_Run.judge \
    && $PY -m Submission2_Run.analysis \
    && touch "$OUT/STABILITY_DONE" || touch "$OUT/STABILITY_FAILED"
  echo "end $(TZ=Asia/Kolkata date)"
} > "$OUT/stability_run.log" 2>&1
