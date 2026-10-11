#!/usr/bin/env bash
# Runs several Gemma inference jobs CONCURRENTLY on one large RunPod GPU,
# each job split into N disjoint shards (gemma_local.py --num-shards /
# --shard-index), one process per shard, every process in its own tmux
# window. Unlike runpod_run_batch.sh (strictly sequential), this is meant for
# a big card (A100/H100 80GB, H200, ...) where several model copies fit.
#
# UNVERIFIED on an actual RunPod pod — same caveat as runpod_run_batch.sh.
#
# Usage:
#   RUNPOD_HOST=<pod-ip-or-hostname> RUNPOD_PORT=<ssh-port> ./runpod_parallel.sh
#
# JOBS (default below = the full-dataset v7/v8 experiment): jobs separated by
# ";;", each job is
#   "<dataset_root>|<prompt_path>|<results_tag>|<model>:<mode>|<shards>[|<first>-<last>]"
#   - shards: integer, or "auto" to size from the GPU's memory (see GB_*)
#   - <first>-<last> (optional): run only shard indices first..last of the
#     <shards> total on THIS pod, to split one job across several pods, e.g.
#     pod A "...|12b:thinking|12|0-3", pod B "...|12|4-7", pod C "...|12|8-11".
#     Every pod must use the same <shards> total so the splits stay disjoint.
#
# Optional overrides:
#   GB_E2B=12 GB_E4B=20 GB_12B=32  per-process VRAM budget used by "auto" (weights +
#                            KV cache for up to 4096 generated tokens + headroom)
#   VRAM_FRACTION=0.9        share of the GPU "auto" is allowed to plan for
#   SKIP_SYNC=1              skip code/dataset rsync + pip install (relaunch only)
#   RUNPOD_USER=root  RUNPOD_KEY=~/.ssh/id_ed25519  REMOTE_DIR=/workspace/os_moderation
#   TMUX_SESSION=gemma_par
#
# Post-processing (pp_*) is applied by analyze_results.py AFTER inference, so
# it is not part of this script — see the printed follow-up commands.
#
# Re-running is safe: without --force, gemma_local.py skips files whose
# results/<file_id>.json already succeeded, so a crashed/OOM'd shard just
# resumes. Written for macOS's bash 3.2 (no associative arrays).

set -euo pipefail

RUNPOD_HOST="${RUNPOD_HOST:?Set RUNPOD_HOST to the pod's IP/hostname (from the RunPod dashboard's Connect tab)}"
RUNPOD_PORT="${RUNPOD_PORT:?Set RUNPOD_PORT to the pod's SSH port (from the RunPod dashboard's Connect tab)}"
RUNPOD_USER="${RUNPOD_USER:-root}"
RUNPOD_KEY="${RUNPOD_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/os_moderation}"
TMUX_SESSION="${TMUX_SESSION:-gemma_par}"
GB_E2B="${GB_E2B:-12}"
GB_E4B="${GB_E4B:-20}"
GB_12B="${GB_12B:-32}"
VRAM_FRACTION="${VRAM_FRACTION:-0.9}"

JOBS="${JOBS:-Dostt|prompt_v7.py|promptv7|e4b:thinking|auto;;Dostt|prompt_v7.py|promptv7|e4b:nothinking|auto;;Dostt|prompt_v8.py|promptv8|e2b:thinking|auto}"

SSH_OPTS=(-p "$RUNPOD_PORT" -i "$RUNPOD_KEY" -o StrictHostKeyChecking=accept-new)
REMOTE="$RUNPOD_USER@$RUNPOD_HOST"

# --- Parse JOBS ---
J_ROOT=(); J_PROMPT=(); J_TAG=(); J_MODEL=(); J_MODE=(); J_SHARDS=(); J_RANGE=()
for job in $(echo "$JOBS" | sed 's/;;/ /g'); do
  IFS='|' read -r dsroot ppath tag mm shards range <<< "$job"
  J_ROOT+=("$dsroot"); J_PROMPT+=("$ppath"); J_TAG+=("$tag")
  J_MODEL+=("${mm%%:*}"); J_MODE+=("${mm##*:}"); J_SHARDS+=("${shards:-1}"); J_RANGE+=("${range:-}")
  if [ -n "${range:-}" ] && [ "${shards:-1}" = "auto" ]; then
    echo "ERROR: a shard range needs an explicit shard total, not auto ($job)" >&2; exit 1
  fi
done
N_JOBS=${#J_ROOT[@]}
PROMPT_FILES=$(printf '%s\n' "${J_PROMPT[@]}" | sort -u | tr '\n' ' ')
DATASET_ROOTS=$(printf '%s\n' "${J_ROOT[@]}" | sort -u | tr '\n' ' ')

if [ -z "${SKIP_SYNC:-}" ]; then
  echo "=== [1/4] Syncing code + prompts ($PROMPT_FILES) to $REMOTE:$REMOTE_DIR ==="
  ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p $REMOTE_DIR"
  # shellcheck disable=SC2086
  rsync -avz -e "ssh ${SSH_OPTS[*]}" \
    gemma_local.py schemas_gemma.py schemas.py schemas_v4.py schemas_v6.py \
    dataset_v2.py config.py prompt_loader.py prompt.py $PROMPT_FILES \
    pipeline_logging.py requirements-gemma.txt \
    "$REMOTE:$REMOTE_DIR/"

  echo "=== [2/4] Syncing dataset root(s): $DATASET_ROOTS ==="
  for dsroot in $DATASET_ROOTS; do
    rsync -az --info=progress2 -e "ssh ${SSH_OPTS[*]}" "$dsroot/" "$REMOTE:$REMOTE_DIR/$dsroot/" \
      || rsync -az -e "ssh ${SSH_OPTS[*]}" "$dsroot/" "$REMOTE:$REMOTE_DIR/$dsroot/"
  done

  echo "=== [3/4] Installing dependencies on the pod ==="
  ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s <<REMOTE_SCRIPT
set -euo pipefail
cd $REMOTE_DIR
apt-get update -qq && apt-get install -y -qq ffmpeg tmux > /dev/null
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — check the pod GPU/template'; print('CUDA OK:', torch.cuda.get_device_name(0))"
grep -viE '^torch$|^torchvision' requirements-gemma.txt > requirements-gemma-runpod.txt
pip install -q -r requirements-gemma-runpod.txt
# Download weights once up front, so N shards don't all race to fetch the same files.
python3 -c "
from huggingface_hub import snapshot_download
for m in sorted(set('$(printf '%s ' "${J_MODEL[@]}")'.split())):
    snapshot_download({'e2b': 'google/gemma-4-E2B-it', 'e4b': 'google/gemma-4-E4B-it', '12b': 'google/gemma-4-12B-it'}[m])
"
REMOTE_SCRIPT
else
  echo "=== SKIP_SYNC set — skipping sync/install ==="
fi

# --- Resolve "auto" shard counts from total GPU memory ---
TOTAL_GB=$(ssh "${SSH_OPTS[@]}" "$REMOTE" "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1" | awk '{printf "%d", $1/1024}')
gb_for() { case "$1" in e2b) echo "$GB_E2B";; 12b) echo "$GB_12B";; *) echo "$GB_E4B";; esac; }
# Number of processes this pod actually runs for job $1 (its range, else all shards).
local_count() {
  local rg="${J_RANGE[$1]}"
  if [ -n "$rg" ]; then echo $(( ${rg##*-} - ${rg%%-*} + 1 )); else echo "${J_SHARDS[$1]}"; fi
}
fixed_gb=0; auto_gb_per_round=0
for i in $(seq 0 $((N_JOBS - 1))); do
  g=$(gb_for "${J_MODEL[$i]}")
  if [ "${J_SHARDS[$i]}" = "auto" ]; then auto_gb_per_round=$((auto_gb_per_round + g))
  else fixed_gb=$((fixed_gb + g * $(local_count "$i"))); fi
done
budget=$(awk -v t="$TOTAL_GB" -v f="$VRAM_FRACTION" 'BEGIN{printf "%d", t*f}')
auto_n=1
if [ "$auto_gb_per_round" -gt 0 ]; then
  auto_n=$(( (budget - fixed_gb) / auto_gb_per_round ))
  [ "$auto_n" -lt 1 ] && auto_n=1
fi
planned=$fixed_gb
for i in $(seq 0 $((N_JOBS - 1))); do
  [ "${J_SHARDS[$i]}" = "auto" ] && J_SHARDS[$i]=$auto_n && planned=$((planned + $(gb_for "${J_MODEL[$i]}") * auto_n))
done
n_procs=0; for i in $(seq 0 $((N_JOBS - 1))); do n_procs=$((n_procs + $(local_count "$i"))); done
echo "GPU: ${TOTAL_GB} GB total, planning ~${planned} GB across ${n_procs} process(es)"
if [ "$planned" -gt "$TOTAL_GB" ]; then
  echo "WARNING: planned VRAM exceeds the GPU — expect CUDA OOM. Lower the shard counts in JOBS." >&2
fi

echo "=== [4/4] Launching shards in tmux session '$TMUX_SESSION' ==="
LAUNCH=""
RESULTS_DIRS=()
for i in $(seq 0 $((N_JOBS - 1))); do
  dsroot="${J_ROOT[$i]}"; ppath="${J_PROMPT[$i]}"; tag="${J_TAG[$i]}"
  model="${J_MODEL[$i]}"; mode="${J_MODE[$i]}"; n="${J_SHARDS[$i]}"
  suffix=""
  if [ "$dsroot" != "Dostt" ]; then name="${dsroot#Dostt_}"; suffix="_$name"; fi
  [ -n "$tag" ] && suffix="${suffix}_${tag}"
  RESULTS_DIRS+=("gemma_results${suffix}")
  thinking_flag=""; [ "$mode" = "thinking" ] && thinking_flag="--thinking"
  first=0; last=$((n - 1))
  if [ -n "${J_RANGE[$i]}" ]; then first="${J_RANGE[$i]%%-*}"; last="${J_RANGE[$i]##*-}"; fi
  if [ "$last" -ge "$n" ] || [ "$first" -gt "$last" ]; then
    echo "ERROR: shard range ${J_RANGE[$i]} invalid for $n shards" >&2; exit 1
  fi
  for s in $(seq "$first" "$last"); do
    win="${model}_${mode}_${tag}_s${s}"
    cmd="python3 gemma_local.py --model $model $thinking_flag --dataset-root $dsroot --prompt-path $ppath --results-tag $tag --num-shards $n --shard-index $s 2>&1 | tee -a logs/run_${win}.out; echo '=== $win DONE ==='; exec bash"
    # Skip a shard whose window is still open (running or finished), so a
    # SKIP_SYNC relaunch only restarts shards that died, never healthy ones.
    LAUNCH="${LAUNCH}tmux list-windows -t $TMUX_SESSION -F '#W' | grep -qx '$win' || { tmux new-window -t $TMUX_SESSION -n '$win' -c $REMOTE_DIR \"$cmd\"; sleep 20; }
"
    echo "  - $win"
  done
done

# Staggered by 20s so model loads don't all spike CPU RAM/disk at the same moment.
ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s <<REMOTE_SCRIPT
set -euo pipefail
cd $REMOTE_DIR
mkdir -p logs
tmux has-session -t $TMUX_SESSION 2>/dev/null || tmux new-session -d -s $TMUX_SESSION -n monitor -c $REMOTE_DIR "watch -n 30 nvidia-smi"
$LAUNCH
REMOTE_SCRIPT

echo ""
echo "All shards launched. Attach (Ctrl-b w lists windows, Ctrl-b d detaches):"
echo "  ssh -p $RUNPOD_PORT -i $RUNPOD_KEY $REMOTE -t 'tmux attach -t $TMUX_SESSION'"
echo ""
echo "Progress (finished files per results dir):"
echo "  ssh -p $RUNPOD_PORT -i $RUNPOD_KEY $REMOTE 'cd $REMOTE_DIR && for d in gemma_results_*/*/results; do echo \$d \$(ls \$d | wc -l); done; grep -h \"Done in\\|FAILED\\|OutOfMemory\" logs/run_*.out | tail'"
echo ""
echo "If a shard dies (e.g. OOM): close its tmux window (it stays open after exit), then"
echo "  SKIP_SYNC=1 JOBS='<same JOBS, same shard counts>' ./runpod_parallel.sh"
echo "relaunches only the missing windows; finished files are skipped."
echo ""
echo "Pull results back when done:"
printf '%s\n' "${RESULTS_DIRS[@]}" | sort -u | while read -r rd; do
  echo "  rsync -avz -e \"ssh -p $RUNPOD_PORT -i $RUNPOD_KEY\" $REMOTE:$REMOTE_DIR/$rd/ ./$rd/"
done
