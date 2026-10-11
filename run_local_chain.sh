#!/usr/bin/env bash
# Runs Gemma jobs locally on the Mac, one after another, restarting the
# python process every BATCH files. On a 24GB Mac, E4B's swap use creeps up
# over a long run until chunks take 5-10 min instead of ~1 min; a fresh
# process resets that. Finished files are skipped (no --force), so each
# restart just continues where the last one stopped.
#
# Usage: nohup ./run_local_chain.sh > logs/local_chain.out 2>&1 &
# Per-file progress: logs/gemma_local_<model>_<mode>.log

set -uo pipefail
cd "$(dirname "$0")"
# Let a still-running gemma_local.py (e.g. from a previous chain) finish first,
# so two models never share memory.
while pgrep -f "gemma_local.py" >/dev/null; do sleep 30; done
source .venv-gemma/bin/activate

BATCH="${BATCH:-25}"
MAX_ROUNDS="${MAX_ROUNDS:-40}"   # guard against looping forever on files that keep erroring

# "<model>|<thinking flag or empty>|<prompt>|<results tag>"
JOBS=(
  "e4b||prompt_v7.py|promptv7"
)

for job in "${JOBS[@]}"; do
  IFS='|' read -r model thinking prompt tag <<< "$job"
  mode=$([ -n "$thinking" ] && echo thinking || echo nothinking)
  log="logs/gemma_local_${model}_${mode}.log"
  echo "=== $(date '+%F %T') START $model $mode $prompt ($tag) ==="
  for round in $(seq 1 "$MAX_ROUNDS"); do
    python3 gemma_local.py --model "$model" $thinking --prompt-path "$prompt" --results-tag "$tag" --limit "$BATCH"
    # The just-finished process's own summary line: nothing left to do -> job complete.
    if tail -n 1 "$log" | grep -q "success=0 error=0"; then
      echo "=== $(date '+%F %T') DONE $model $mode after $round round(s) ==="
      break
    fi
  done
done
echo "=== $(date '+%F %T') ALL LOCAL JOBS COMPLETE ==="
