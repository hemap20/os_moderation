#!/usr/bin/env bash
# One command, run from THIS machine, to: sync code+dataset to a RunPod pod,
# install dependencies there, and kick off Gemma 4 E4B (both non-thinking
# and thinking, sequentially — not concurrently, to avoid the same
# memory-contention stalls seen running two heavy models at once locally)
# inside a detachable tmux session so it survives your SSH connection
# dropping and keeps running overnight.
#
# UNVERIFIED on an actual RunPod pod — written from the local pipeline's
# known-working behavior + RunPod's standard SSH access pattern. Expect to
# adjust connection details / debug real errors on first run.
#
# Usage:
#   RUNPOD_HOST=<pod-ip-or-hostname> RUNPOD_PORT=<ssh-port> \
#     ./runpod_e4b_run.sh
#
# Optional overrides:
#   RUNPOD_USER=root                  (default)
#   RUNPOD_KEY=~/.ssh/id_ed25519       (default; your SSH key for the pod)
#   REMOTE_DIR=/workspace/os_moderation (default; where code+data land on the pod)

set -euo pipefail

RUNPOD_HOST="${RUNPOD_HOST:?Set RUNPOD_HOST to the pod's IP/hostname (from the RunPod dashboard's Connect tab)}"
RUNPOD_PORT="${RUNPOD_PORT:?Set RUNPOD_PORT to the pod's SSH port (from the RunPod dashboard's Connect tab)}"
RUNPOD_USER="${RUNPOD_USER:-root}"
RUNPOD_KEY="${RUNPOD_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/os_moderation}"

SSH_OPTS=(-p "$RUNPOD_PORT" -i "$RUNPOD_KEY" -o StrictHostKeyChecking=accept-new)
REMOTE="$RUNPOD_USER@$RUNPOD_HOST"

echo "=== [1/4] Syncing code to $REMOTE:$REMOTE_DIR ==="
ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p $REMOTE_DIR"
rsync -avz --progress -e "ssh ${SSH_OPTS[*]}" \
  gemma_local.py schemas_gemma.py schemas.py dataset_v2.py config.py \
  prompt_loader.py prompt.py pipeline_logging.py requirements-gemma.txt \
  "$REMOTE:$REMOTE_DIR/"

echo "=== [2/4] Syncing Dostt/ dataset (this is the slow part, several GB) ==="
rsync -avz --progress -e "ssh ${SSH_OPTS[*]}" \
  Dostt/ "$REMOTE:$REMOTE_DIR/Dostt/"

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

echo "=== [4/4] Starting E4B runs (non-thinking, then thinking) in tmux session 'gemma_e4b' ==="
ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s <<REMOTE_SCRIPT
set -euo pipefail
cd $REMOTE_DIR
tmux kill-session -t gemma_e4b 2>/dev/null || true
tmux new-session -d -s gemma_e4b -c $REMOTE_DIR
tmux send-keys -t gemma_e4b "python3 gemma_local.py --model e4b && python3 gemma_local.py --model e4b --thinking; echo '=== BOTH E4B RUNS COMPLETE ==='" Enter
REMOTE_SCRIPT

echo ""
echo "Started. Reattach any time with:"
echo "  ssh -p $RUNPOD_PORT -i $RUNPOD_KEY $REMOTE -t 'tmux attach -t gemma_e4b'"
echo ""
echo "Pull results back to this machine when done with:"
echo "  rsync -avz -e \"ssh -p $RUNPOD_PORT -i $RUNPOD_KEY\" $REMOTE:$REMOTE_DIR/gemma_results/ ./gemma_results/"
