#!/usr/bin/env bash
# Like runpod_run.sh, but runs several (dataset-root, prompt, results-tag,
# models) JOBS sequentially in ONE tmux session on the pod — for when a
# single batch needs to mix full-dataset and dev-set runs, or several
# prompt versions, in one go instead of invoking runpod_run.sh repeatedly.
#
# UNVERIFIED on an actual RunPod pod — same caveat as runpod_run.sh, written
# from the local pipeline's known-working behavior + RunPod's standard SSH
# access pattern.
#
# Usage:
#   RUNPOD_HOST=<pod-ip-or-hostname> RUNPOD_PORT=<ssh-port> \
#   JOBS="Dostt|prompt_v6.py|promptv6|e2b:nothinking;;Dostt_dev|prompt_v6.py|promptv6|e2b:thinking,e4b:thinking;;Dostt_dev|prompt_v7.py|promptv7|e2b:thinking,e4b:thinking;;Dostt_dev|prompt_v8.py|promptv8|e2b:thinking,e4b:thinking" \
#     ./runpod_run_batch.sh
#
# JOBS: jobs separated by ";;", each job is
#   "<dataset_root>|<prompt_path>|<results_tag>|<model:mode,model:mode,...>"
#   - dataset_root: Dostt (full) or Dostt_dev (dev set), etc.
#   - prompt_path: local prompt file, e.g. prompt_v6.py (synced to the pod)
#   - results_tag: suffix for gemma_results[_<dataset-suffix>]_<tag>/
#   - models: comma-separated model:mode pairs, model in {e2b,e4b,12b},
#     mode in {thinking,nothinking}
# All jobs run strictly sequentially, in the order given, in one tmux
# session — same "don't share the GPU between two heavy models" reasoning
# as runpod_run.sh.
#
# Optional connection overrides (same as runpod_run.sh):
#   RUNPOD_USER=root  RUNPOD_KEY=~/.ssh/id_ed25519  REMOTE_DIR=/workspace/os_moderation
#   TMUX_SESSION=gemma_batch

set -euo pipefail

RUNPOD_HOST="${RUNPOD_HOST:?Set RUNPOD_HOST to the pod's IP/hostname (from the RunPod dashboard's Connect tab)}"
RUNPOD_PORT="${RUNPOD_PORT:?Set RUNPOD_PORT to the pod's SSH port (from the RunPod dashboard's Connect tab)}"
RUNPOD_USER="${RUNPOD_USER:-root}"
RUNPOD_KEY="${RUNPOD_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/os_moderation}"
TMUX_SESSION="${TMUX_SESSION:-gemma_batch}"

JOBS="${JOBS:?Set JOBS, e.g. JOBS=\"Dostt|prompt_v6.py|promptv6|e2b:nothinking;;Dostt_dev|prompt_v7.py|promptv7|e2b:thinking,e4b:thinking\"}"

SSH_OPTS=(-p "$RUNPOD_PORT" -i "$RUNPOD_KEY" -o StrictHostKeyChecking=accept-new)
REMOTE="$RUNPOD_USER@$RUNPOD_HOST"

# --- Parse JOBS into parallel arrays, collecting the distinct prompt files
# and dataset roots that need syncing along the way. ---
declare -a JOB_DATASET_ROOTS JOB_PROMPT_PATHS JOB_RESULTS_TAGS JOB_MODELS
declare -A SEEN_PROMPTS SEEN_ROOTS
PROMPT_FILES=()
DATASET_ROOTS=()

IFS=';;' read -ra RAW_JOBS <<< "$JOBS"
for job in "${RAW_JOBS[@]}"; do
  [ -z "$job" ] && continue
  IFS='|' read -r dsroot ppath tag models <<< "$job"
  JOB_DATASET_ROOTS+=("$dsroot")
  JOB_PROMPT_PATHS+=("$ppath")
  JOB_RESULTS_TAGS+=("$tag")
  JOB_MODELS+=("$models")
  if [ -z "${SEEN_PROMPTS[$ppath]:-}" ]; then
    SEEN_PROMPTS[$ppath]=1
    PROMPT_FILES+=("$ppath")
  fi
  if [ -z "${SEEN_ROOTS[$dsroot]:-}" ]; then
    SEEN_ROOTS[$dsroot]=1
    DATASET_ROOTS+=("$dsroot")
  fi
done

echo "=== [1/4] Syncing code + prompts (${PROMPT_FILES[*]}) to $REMOTE:$REMOTE_DIR ==="
ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p $REMOTE_DIR"
rsync -avz --progress -e "ssh ${SSH_OPTS[*]}" \
  gemma_local.py schemas_gemma.py schemas.py schemas_v4.py schemas_v6.py \
  dataset_v2.py config.py prompt_loader.py prompt.py "${PROMPT_FILES[@]}" \
  pipeline_logging.py requirements-gemma.txt \
  "$REMOTE:$REMOTE_DIR/"

echo "=== [2/4] Syncing dataset root(s): ${DATASET_ROOTS[*]} ==="
for dsroot in "${DATASET_ROOTS[@]}"; do
  rsync -avz --progress -e "ssh ${SSH_OPTS[*]}" \
    "$dsroot/" "$REMOTE:$REMOTE_DIR/$dsroot/"
done

echo "=== [3/4] Installing dependencies on the pod ==="
ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s <<REMOTE_SCRIPT
set -euo pipefail
cd $REMOTE_DIR
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — check the pod GPU/template'; print('CUDA OK:', torch.cuda.get_device_name(0))"
grep -viE '^torch$|^torchvision' requirements-gemma.txt > requirements-gemma-runpod.txt
pip install -q -r requirements-gemma-runpod.txt
REMOTE_SCRIPT

echo "=== [4/4] Starting all jobs in tmux session '$TMUX_SESSION' ==="
# Build one big && chain across every job, each job itself a && chain across
# its model:mode pairs — everything strictly sequential, one GPU at a time.
CMD=""
RESULTS_DIRS=()
for i in "${!JOB_DATASET_ROOTS[@]}"; do
  dsroot="${JOB_DATASET_ROOTS[$i]}"
  ppath="${JOB_PROMPT_PATHS[$i]}"
  tag="${JOB_RESULTS_TAGS[$i]}"
  models="${JOB_MODELS[$i]}"

  suffix=""
  if [ "$dsroot" != "Dostt" ]; then
    name="${dsroot#Dostt_}"
    if [ "$name" != "$dsroot" ]; then suffix="_$name"; else suffix="_$dsroot"; fi
  fi
  if [ -n "$tag" ]; then suffix="${suffix}_${tag}"; fi
  RESULTS_DIRS+=("gemma_results${suffix}")

  IFS=',' read -ra PAIRS <<< "$models"
  for pair in "${PAIRS[@]}"; do
    model="${pair%%:*}"
    mode="${pair##*:}"
    thinking_flag=""
    if [ "$mode" == "thinking" ]; then thinking_flag="--thinking"; fi
    run_cmd="echo '>>> $dsroot / $ppath / $tag / $model:$mode' && python3 gemma_local.py --model $model $thinking_flag --dataset-root $dsroot --prompt-path $ppath"
    if [ -n "$tag" ]; then run_cmd="$run_cmd --results-tag $tag"; fi
    if [ -z "$CMD" ]; then CMD="$run_cmd"; else CMD="$CMD && $run_cmd"; fi
  done
done
CMD="$CMD; echo '=== ALL BATCH RUNS COMPLETE ==='"

ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s <<REMOTE_SCRIPT
set -euo pipefail
cd $REMOTE_DIR
tmux kill-session -t $TMUX_SESSION 2>/dev/null || true
tmux new-session -d -s $TMUX_SESSION -c $REMOTE_DIR
tmux send-keys -t $TMUX_SESSION "$CMD" Enter
REMOTE_SCRIPT

echo ""
echo "Started ${#JOB_DATASET_ROOTS[@]} job(s) sequentially in tmux session '$TMUX_SESSION':"
for i in "${!JOB_DATASET_ROOTS[@]}"; do
  echo "  - ${JOB_DATASET_ROOTS[$i]} / ${JOB_PROMPT_PATHS[$i]} / tag=${JOB_RESULTS_TAGS[$i]} / ${JOB_MODELS[$i]}"
done
echo ""
echo "Reattach any time with:"
echo "  ssh -p $RUNPOD_PORT -i $RUNPOD_KEY $REMOTE -t 'tmux attach -t $TMUX_SESSION'"
echo ""
echo "Pull results back to this machine when done with (one per distinct results dir):"
printf '%s\n' "${RESULTS_DIRS[@]}" | sort -u | while read -r rd; do
  echo "  rsync -avz -e \"ssh -p $RUNPOD_PORT -i $RUNPOD_KEY\" $REMOTE:$REMOTE_DIR/$rd/ ./$rd/"
done
