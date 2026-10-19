#!/usr/bin/env bash
# Transcript-based Gemma runs on the Mac (gemma_local_text.py), one model at a
# time, restarting the python process every BATCH calls. On a 24GB Mac, E4B's
# swap creeps up over a long run until each call slows to minutes; a fresh
# process resets that. Finished calls are skipped (no --force), so each restart
# continues where the last stopped, and re-running this script resumes too.
#
# Usage: nohup ./run_local_text_chain.sh > logs/local_text_chain.out 2>&1 &
# Per-call progress: logs/gemma_local_text_<model>_itc10s[_thinking].log

set -uo pipefail
cd "$(dirname "$0")"
# Never share memory with another local Gemma run.
while pgrep -f "gemma_local.py|gemma_local_text.py" >/dev/null; do sleep 30; done
source .venv-gemma/bin/activate

BATCH="${BATCH:-25}"
MAX_ROUNDS="${MAX_ROUNDS:-40}"   # guard against looping forever on calls that keep erroring
COMMON=(--prompt-path prompt_v8.py --results-tag promptv8 --transcript-source itc --transcript-chunk-seconds 10)

# "<model>|<thinking flag or empty>", run in this order (shortest first)
JOBS=(
  "e4b|"
  "e2b|--thinking"
  "e4b|--thinking"
)

for job in "${JOBS[@]}"; do
  IFS='|' read -r model thinking <<< "$job"
  log="logs/gemma_local_text_${model}_itc10s${thinking:+_thinking}.log"
  echo "=== $(date '+%F %T') START $model ${thinking:-non-thinking} ==="
  for round in $(seq 1 "$MAX_ROUNDS"); do
    python3 gemma_local_text.py --model "$model" $thinking "${COMMON[@]}" --limit "$BATCH"
    # A round with nothing left to do logs "success=0 error=0" -> this job is complete.
    if tail -n 1 "$log" | grep -q "success=0 error=0"; then
      echo "=== $(date '+%F %T') DONE $model ${thinking:-non-thinking} after $round round(s) ==="
      break
    fi
  done
done
echo "=== $(date '+%F %T') ALL TRANSCRIPT JOBS COMPLETE ==="
