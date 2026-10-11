#!/usr/bin/env bash
# One-command, fire-and-forget Gemma run INSIDE a RunPod pod (web terminal,
# no SSH). Clone the repo on the pod, then:
#
#   bash runpod_pod_run.sh <preset> [<preset> ...]
#
# It detaches itself (safe to close the browser tab), then:
#   1. installs deps (pinned to the versions validated locally) + downloads weights
#   2. sizes the shard count from the GPU's VRAM (override with SHARDS=N)
#   3. runs every shard in parallel, re-launching any shard that crashes
#      (finished files are always skipped, so a retry only redoes unfinished work)
#   4. verifies every dataset file has a successful result
#   5. zips results + logs to /workspace/<preset>_results.zip
#   6. optionally pushes results to a GitHub branch (GITHUB_TOKEN) and stops
#      the pod (AUTO_STOP=1) so it doesn't bill while idle
#
# Presets (full Dostt/ dataset) — one per pod, sized for ~10h:
#   e2b_think_v8      e2b thinking, prompt v8      A100/H100 80GB  (~6 shards, ~6h)
#   e4b_think_v7      e4b thinking, prompt v7      H200 141GB      (~6 shards, ~7h)
#   e4b_nothink_v7    e4b non-thinking, prompt v7  L40S 48GB       (~2 shards, ~7h)
#   12b_nothink_v7    12b non-thinking, prompt v7  L40S 48GB       (1 process, ~3-6h)
#   12b_think_v7      12b thinking, prompt v7      3x H200, one per pod, with
#                       SHARDS=12 SHARD_RANGE=0-3 / 4-7 / 8-11 (rough: ~8-9h)
# Several presets on one pod share the GPU (VRAM is split between them).
#
# Optional env:
#   SHARDS=N              total shards per preset (default: auto from VRAM)
#   SHARD_RANGE=a-b       run only shards a..b of SHARDS on this pod (split a job across pods)
#   MAX_RETRIES=5         relaunches per shard before giving up on it
#   GITHUB_TOKEN=...      push results to branch runpod-results-<presets>-<pod> when done
#   AUTO_STOP=1           stop this pod when everything finished AND the push succeeded
#                         (needs runpodctl, preinstalled on RunPod)
#
# Monitor (optional — nothing needs checking): tail -f /workspace/pod_run.log

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
MAIN_LOG="/workspace/pod_run.log"

# --- Re-exec detached so closing the web terminal can't kill the run ---
if [ -z "${_POD_RUN_DETACHED:-}" ]; then
  [ $# -ge 1 ] || { sed -n '2,40p' "$0"; exit 1; }
  export _POD_RUN_DETACHED=1
  nohup setsid bash "$0" "$@" >> "$MAIN_LOG" 2>&1 < /dev/null &
  echo "Started in the background (pid $!). Safe to close this terminal."
  echo "Progress (optional): tail -f $MAIN_LOG"
  exit 0
fi

cd "$REPO_DIR"
log() { echo "[$(date '+%F %T')] $*"; }

# preset -> "<model>|<thinking flag>|<prompt>|<results tag>|<GB per process>"
preset_spec() {
  case "$1" in
    e2b_think_v8)   echo "e2b|--thinking|prompt_v8.py|promptv8|12" ;;
    e4b_think_v7)   echo "e4b|--thinking|prompt_v7.py|promptv7|20" ;;
    e4b_nothink_v7) echo "e4b||prompt_v7.py|promptv7|20" ;;
    12b_think_v7)   echo "12b|--thinking|prompt_v7.py|promptv7|32" ;;
    12b_nothink_v7) echo "12b||prompt_v7.py|promptv7|32" ;;
    *) return 1 ;;
  esac
}
model_id() { case "$1" in e2b) echo google/gemma-4-E2B-it;; e4b) echo google/gemma-4-E4B-it;; 12b) echo google/gemma-4-12B-it;; esac; }

PRESETS=("$@")
for p in "${PRESETS[@]}"; do preset_spec "$p" > /dev/null || { log "ERROR: unknown preset '$p'"; exit 1; }; done
RUN_NAME="$(IFS=_; echo "${PRESETS[*]}")${SHARD_RANGE:+_s$SHARD_RANGE}"
log "=== START $RUN_NAME on $(hostname) ==="

# --- 1. Setup ---
log "Installing system + python deps"
apt-get update -qq && apt-get install -y -qq ffmpeg zip > /dev/null
# Pinned to the versions gemma_local.py was validated with locally; torch
# comes from the RunPod PyTorch image. No torchaudio/torchcodec (see
# requirements-gemma.txt for why torchcodec breaks audio loading).
pip install -q "transformers==5.17.0" "accelerate==1.15.0" pydub pillow "librosa==0.11.0" \
  "pydantic>=2.0" "mutagen>=1.47" huggingface_hub
python3 -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))" \
  || { log "ERROR: CUDA not available — pick a GPU pod with the RunPod PyTorch template"; exit 1; }

for p in "${PRESETS[@]}"; do
  IFS='|' read -r model _ _ _ _ <<< "$(preset_spec "$p")"
  log "Downloading weights for $model"
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$(model_id "$model")')"
done

# --- 2. Shard counts from VRAM ---
TOTAL_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{printf "%d", $1/1024}')
sum_gb=0
for p in "${PRESETS[@]}"; do IFS='|' read -r _ _ _ _ gb <<< "$(preset_spec "$p")"; sum_gb=$((sum_gb + gb)); done
per_pod=$(( TOTAL_GB * 9 / 10 / sum_gb )); [ "$per_pod" -lt 1 ] && per_pod=1
log "GPU ${TOTAL_GB}GB -> $per_pod process(es) per preset on this pod"

# --- 3. Run every shard, supervised ---
mkdir -p logs
run_shard() {  # model thinking prompt tag n s
  local model=$1 thinking=$2 prompt=$3 tag=$4 n=$5 s=$6
  local mode=nothinking; [ -n "$thinking" ] && mode=thinking
  local name="${model}_${mode}_${tag}_s${s}of${n}"
  local out="logs/run_${name}.out"
  for attempt in $(seq 1 "${MAX_RETRIES:-5}"); do
    log "[$name] attempt $attempt"
    python3 gemma_local.py --model "$model" $thinking --prompt-path "$prompt" --results-tag "$tag" \
      --num-shards "$n" --shard-index "$s" >> "$out" 2>&1
    # gemma_local.py's own summary for a run with nothing left: "success=0 error=0".
    # A clean finish of real work ends with error=0 too; re-run once to confirm
    # nothing is left (cheap: every finished file is skipped).
    if tail -n 5 "$out" | grep -q "success=0 error=0"; then log "[$name] complete"; return 0; fi
    sleep 30
  done
  log "[$name] GAVE UP after ${MAX_RETRIES:-5} attempts — see $out"
  return 1
}

PIDS=()
for p in "${PRESETS[@]}"; do
  IFS='|' read -r model thinking prompt tag _ <<< "$(preset_spec "$p")"
  if [ -n "${SHARD_RANGE:-}" ]; then
    n="${SHARDS:?SHARD_RANGE needs SHARDS (the total across all pods)}"
    first="${SHARD_RANGE%%-*}"; last="${SHARD_RANGE##*-}"
  else
    n="${SHARDS:-$per_pod}"; first=0; last=$((n - 1))
  fi
  for s in $(seq "$first" "$last"); do
    run_shard "$model" "$thinking" "$prompt" "$tag" "$n" "$s" &
    PIDS+=($!)
    sleep 20   # stagger model loads
  done
done
log "Launched ${#PIDS[@]} shard(s); waiting"
failed=0
for pid in "${PIDS[@]}"; do wait "$pid" || failed=$((failed + 1)); done

# --- 4. Verify ---
log "Verifying results"
VERIFY=$(python3 - "${PRESETS[@]}" <<'EOF'
import json, sys, zlib, os
from pathlib import Path
import dataset_v2 as d
specs = {"e2b_think_v8": ("e2b", True, "promptv8"), "e4b_think_v7": ("e4b", True, "promptv7"),
         "e4b_nothink_v7": ("e4b", False, "promptv7"), "12b_think_v7": ("12b", True, "promptv7"),
         "12b_nothink_v7": ("12b", False, "promptv7")}
ids = {r.file_id for r in d.load_dataset_v2(Path("Dostt"))}
n = os.environ.get("SHARDS"); rng = os.environ.get("SHARD_RANGE")
if rng and n:
    a, b = map(int, rng.split("-")); ids = {i for i in ids if a <= zlib.crc32(i.encode()) % int(n) <= b}
for p in sys.argv[1:]:
    model, thinking, tag = specs[p]
    rd = Path(f"gemma_results_{tag}/{model}_{'thinking' if thinking else 'nothinking'}/results")
    ok = {f.stem for f in rd.glob("*.json") if json.loads(f.read_text()).get("status") == "success"} if rd.exists() else set()
    missing = ids - ok
    print(f"{p}: {len(ids & ok)}/{len(ids)} files OK" + (f", MISSING {len(missing)}: {sorted(missing)[:5]}" if missing else ""))
EOF
)
log "$VERIFY"

# --- 5. Package ---
ZIP="/workspace/${RUN_NAME}_results.zip"
zip -qr "$ZIP" gemma_results_promptv7 gemma_results_promptv8 logs 2>/dev/null
log "Results zipped: $ZIP"

# --- 6. Push + stop ---
pushed=0
if [ -n "${GITHUB_TOKEN:-}" ]; then
  branch="runpod-results-${RUN_NAME}"
  log "Pushing results to branch $branch"
  git config user.email "runpod@localhost"; git config user.name "runpod"
  git checkout -q -b "$branch"
  git add -f gemma_results_promptv7 gemma_results_promptv8 logs/gemma_local_*shard*.log 2>/dev/null
  git commit -q -m "RunPod results: $RUN_NAME" -m "$VERIFY"
  remote=$(git remote get-url origin | sed "s#https://#https://x-access-token:${GITHUB_TOKEN}@#")
  git push -q "$remote" "$branch" && { pushed=1; log "Pushed $branch"; } || log "ERROR: git push failed — results are still in $ZIP"
fi

if [ "$failed" -gt 0 ] || echo "$VERIFY" | grep -q MISSING; then
  log "=== FINISHED WITH PROBLEMS ($failed shard(s) gave up) — pod left running so nothing is lost ==="
  exit 1
fi
log "=== ALL DONE: $RUN_NAME ==="
if [ "${AUTO_STOP:-0}" = "1" ] && [ "$pushed" = "0" ]; then
  log "AUTO_STOP=1 but results were not pushed (no GITHUB_TOKEN or push failed) — leaving pod running so $ZIP isn't lost"
elif [ "${AUTO_STOP:-0}" = "1" ] && command -v runpodctl > /dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
  log "AUTO_STOP=1 — stopping pod $RUNPOD_POD_ID"
  runpodctl stop pod "$RUNPOD_POD_ID"
fi
