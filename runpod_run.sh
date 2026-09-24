#!/usr/bin/env bash
# One command, run from THIS machine, to: sync code+dataset+prompt to a
# RunPod pod, install dependencies there, and kick off one or more
# gemma_local.py runs (sequentially, in a detachable tmux session, so it
# survives your SSH connection dropping) — generalized from the original
# runpod_e4b_run.sh to take model/dataset-root/results-tag/prompt-path as
# parameters instead of being hardcoded to E4B + the full dataset + prompt.py.
#
# UNVERIFIED on an actual RunPod pod — written from the local pipeline's
# known-working behavior + RunPod's standard SSH access pattern. Expect to
# adjust connection details / debug real errors on first run.
#
# Usage:
#   RUNPOD_HOST=<pod-ip-or-hostname> RUNPOD_PORT=<ssh-port> \
#   MODELS="e2b:thinking,e4b:thinking" \
#   DATASET_ROOT=Dostt_dev \
#   RESULTS_TAG=promptv3 \
#   PROMPT_PATH=prompt_v3.py \
#     ./runpod_run.sh
#
# MODELS: comma-separated "model:mode" pairs, model in {e2b,e4b,12b}, mode
#   in {thinking,nothinking}. Run sequentially in the pod's tmux session,
#   same reasoning as the original script: two heavy models sharing one GPU
#   concurrently risks memory-contention stalls, seen locally on MPS and not
#   worth the risk of rediscovering on unfamiliar hardware.
# DATASET_ROOT: passed straight through to gemma_local.py's --dataset-root.
#   Omit (or leave unset) for the full Dostt/ dataset.
# RESULTS_TAG: passed straight through to --results-tag, keeping this run's
#   outputs (gemma_results[_<dataset-suffix>]_<tag>/) separate from any
#   other run on the same pod or pulled back to the same local checkout.
# PROMPT_PATH: local path to the prompt file this run should use (e.g.
#   prompt_v3.py) — synced to the pod alongside prompt.py; ground truth
#   always used prompt.py regardless of this setting (see gemma_local.py's
#   PROMPT_PATH global for why they're independent).
#
# Optional connection overrides:
#   RUNPOD_USER=root                    (default)
#   RUNPOD_KEY=~/.ssh/id_ed25519         (default; your SSH key for the pod)
#   REMOTE_DIR=/workspace/os_moderation  (default; where code+data land on the pod)

set -euo pipefail

RUNPOD_HOST="${RUNPOD_HOST:?Set RUNPOD_HOST to the pod's IP/hostname (from the RunPod dashboard's Connect tab)}"
RUNPOD_PORT="${RUNPOD_PORT:?Set RUNPOD_PORT to the pod's SSH port (from the RunPod dashboard's Connect tab)}"
RUNPOD_USER="${RUNPOD_USER:-root}"
RUNPOD_KEY="${RUNPOD_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/os_moderation}"

MODELS="${MODELS:?Set MODELS, e.g. MODELS=\"e2b:thinking,e4b:thinking\"}"
DATASET_ROOT="${DATASET_ROOT:-Dostt}"
RESULTS_TAG="${RESULTS_TAG:-}"
PROMPT_PATH="${PROMPT_PATH:-prompt.py}"
TMUX_SESSION="gemma_${RESULTS_TAG:-run}"

SSH_OPTS=(-p "$RUNPOD_PORT" -i "$RUNPOD_KEY" -o StrictHostKeyChecking=accept-new)
REMOTE="$RUNPOD_USER@$RUNPOD_HOST"

echo "=== [1/4] Syncing code + prompt to $REMOTE:$REMOTE_DIR ==="
ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p $REMOTE_DIR"
rsync -avz --progress -e "ssh ${SSH_OPTS[*]}" \
  gemma_local.py schemas_gemma.py schemas.py dataset_v2.py config.py \
  prompt_loader.py prompt.py "$PROMPT_PATH" pipeline_logging.py requirements-gemma.txt \
  "$REMOTE:$REMOTE_DIR/"

echo "=== [2/4] Syncing $DATASET_ROOT/ dataset ==="
rsync -avz --progress -e "ssh ${SSH_OPTS[*]}" \
  "$DATASET_ROOT/" "$REMOTE:$REMOTE_DIR/$DATASET_ROOT/"

echo "=== [3/4] Installing dependencies on the pod ==="
# Skip torch/torchvision from the requirements file — RunPod's PyTorch
# template images ship a CUDA-matched torch already; reinstalling from pip
# risks silently downgrading to a mismatched or CPU-only build. Everything
# else in the file is safe to install fresh.
ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s <<REMOTE_SCRIPT
set -euo pipefail
cd $REMOTE_DIR
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — check the pod GPU/template'; print('CUDA OK:', torch.cuda.get_device_name(0))"
grep -viE '^torch$|^torchvision' requirements-gemma.txt > requirements-gemma-runpod.txt
pip install -q -r requirements-gemma-runpod.txt
REMOTE_SCRIPT

echo "=== [4/4] Starting runs in tmux session '$TMUX_SESSION' ==="
# Build one shell command per model:mode pair, joined with && so they run
# strictly sequentially (see MODELS comment above for why not concurrent).
CMD=""
IFS=',' read -ra PAIRS <<< "$MODELS"
for pair in "${PAIRS[@]}"; do
  model="${pair%%:*}"
  mode="${pair##*:}"
  thinking_flag=""
  if [ "$mode" == "thinking" ]; then
    thinking_flag="--thinking"
  fi
  run_cmd="python3 gemma_local.py --model $model $thinking_flag --dataset-root $DATASET_ROOT --prompt-path $PROMPT_PATH"
  if [ -n "$RESULTS_TAG" ]; then
    run_cmd="$run_cmd --results-tag $RESULTS_TAG"
  fi
  if [ -z "$CMD" ]; then
    CMD="$run_cmd"
  else
    CMD="$CMD && $run_cmd"
  fi
done
CMD="$CMD; echo '=== ALL RUNS COMPLETE ==='"

ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s <<REMOTE_SCRIPT
set -euo pipefail
cd $REMOTE_DIR
tmux kill-session -t $TMUX_SESSION 2>/dev/null || true
tmux new-session -d -s $TMUX_SESSION -c $REMOTE_DIR
tmux send-keys -t $TMUX_SESSION "$CMD" Enter
REMOTE_SCRIPT

RESULTS_SUFFIX=""
if [ "$DATASET_ROOT" != "Dostt" ]; then
  name="${DATASET_ROOT#Dostt_}"
  if [ "$name" != "$DATASET_ROOT" ]; then
    RESULTS_SUFFIX="_$name"
  else
    RESULTS_SUFFIX="_$DATASET_ROOT"
  fi
fi
if [ -n "$RESULTS_TAG" ]; then
  RESULTS_SUFFIX="${RESULTS_SUFFIX}_${RESULTS_TAG}"
fi
RESULTS_DIR="gemma_results${RESULTS_SUFFIX}"

echo ""
echo "Started: $CMD"
echo ""
echo "Reattach any time with:"
echo "  ssh -p $RUNPOD_PORT -i $RUNPOD_KEY $REMOTE -t 'tmux attach -t $TMUX_SESSION'"
echo ""
echo "Pull results back to this machine when done with:"
echo "  rsync -avz -e \"ssh -p $RUNPOD_PORT -i $RUNPOD_KEY\" $REMOTE:$REMOTE_DIR/$RESULTS_DIR/ ./$RESULTS_DIR/"
