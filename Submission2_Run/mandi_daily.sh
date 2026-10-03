#!/usr/bin/env bash
# Daily Agmarknet snapshot for Submission 2. Installed in crontab at 10:00 IST; safe to run by hand.
# Writes data_submission2/mandi/mandi_<arrival-date>.jsonl plus a hashed manifest, and logs one line.
#
# Retries: 8 attempts, 20 minutes apart (about 2 h 20 min). The 25 Sep 2026 run failed because a
# data.gov.in outage outlasted the old 3-attempt, 30-minute window. Log times are IST.
cd /home/Debz/Research/Finetune_AgriFair/Codes || exit 1
set -a; . ./.env; set +a
LOG=data_submission2/mandi/mandi_cron.log
stamp() { TZ=Asia/Kolkata date '+%F %H:%M IST'; }
ATTEMPTS=8
for attempt in $(seq 1 $ATTEMPTS); do
  if out=$(/home/Debz/Research/Finetune_AgriFair/.venv/bin/python -m Submission2_Run.fetch_mandi 2>&1); then
    echo "$(stamp) ok   attempt=$attempt $(echo "$out" | tail -1)" >> "$LOG"; exit 0
  fi
  [ "$attempt" -lt "$ATTEMPTS" ] && sleep 1200
done
echo "$(stamp) FAIL after $ATTEMPTS attempts: $(echo "$out" | tail -1 | cut -c1-200)" >> "$LOG"
exit 1
