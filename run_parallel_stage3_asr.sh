#!/usr/bin/env bash
# Phase 1 (parallel): Indic-Transcribe-Core transcription + Gemma 4 E2B text
#                      classification, run together.
# Phase 2 (sequential, after Phase 1 fully finishes): Gemma 4 E4B text
#                      classification, run alone.
#
# WHY THIS SHAPE: on a disk-constrained pod (30GB seen in practice), E2B +
# Indic-Transcribe-Core's cached weights alone leave too little room for
# E4B's ~16GB download to also fit — this isn't a "simultaneous download"
# problem, it's a total-capacity problem. So between phases, this script
# DELETES E2B's and Indic-Transcribe-Core's Hugging Face cache to free room
# for E4B. Trade-off: rerunning E2B or Indic-Transcribe-Core later means
# re-downloading their weights. Set KEEP_CACHE=1 to skip the cleanup if your
# pod's disk is large enough to hold all three at once.
#
# UNVERIFIED like everything else Gemma/local-ASR in this session.
#
# Usage:
#   ./run_parallel_stage3_asr.sh
#   KEEP_CACHE=1 TRANSCRIPT_CHUNK_SECONDS=5 ./run_parallel_stage3_asr.sh

set -uo pipefail

TRANSCRIPT_CHUNK_SECONDS="${TRANSCRIPT_CHUNK_SECONDS:-5}"
KEEP_CACHE="${KEEP_CACHE:-0}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
# This repo's pod sessions have also seen HF_HOME effectively land at
# /workspace/.cache/huggingface — check both locations when cleaning up.
HF_CACHE_WORKSPACE="/workspace/.cache/huggingface/hub"

mkdir -p logs

echo "=== PHASE 1: Indic-Transcribe-Core + Gemma E2B (parallel) ==="

python3 indic_transcribe_core_pipeline.py --chunk-seconds 2,5,7,10 \
  > logs/parallel_transcribe_core.out 2>&1 &
PID_TRANSCRIBE=$!
echo "Indic-Transcribe-Core sweep: PID $PID_TRANSCRIBE (log: logs/parallel_transcribe_core.out)"

python3 gemma_local_text.py --model e2b --transcript-chunk-seconds "$TRANSCRIPT_CHUNK_SECONDS" \
  > logs/parallel_e2b_text.out 2>&1 &
PID_E2B=$!
echo "E2B text classification: PID $PID_E2B (log: logs/parallel_e2b_text.out)"

echo ""
echo "Watch progress with: tail -f logs/parallel_transcribe_core.out logs/parallel_e2b_text.out"
echo ""

PHASE1_FAILED=0
wait "$PID_TRANSCRIBE" || { echo "Indic-Transcribe-Core sweep FAILED (exit $?)"; PHASE1_FAILED=1; }
wait "$PID_E2B" || { echo "E2B text classification FAILED (exit $?)"; PHASE1_FAILED=1; }

echo ""
echo "=== PHASE 1 DONE $( [ "$PHASE1_FAILED" -eq 0 ] && echo '(both succeeded)' || echo '(at least one failed — check logs above)' ) ==="

if [ "$KEEP_CACHE" -eq 0 ]; then
  echo ""
  echo "Freeing disk for E4B: removing E2B and Indic-Transcribe-Core cached weights..."
  for cache_dir in "$HF_CACHE" "$HF_CACHE_WORKSPACE"; do
    rm -rf "$cache_dir/models--google--gemma-4-E2B-it" 2>/dev/null
    rm -rf "$cache_dir/models--bodhan-ai--indic-transcribe-core" 2>/dev/null
  done
  df -h / 2>/dev/null | tail -1
else
  echo "KEEP_CACHE=1 set — leaving E2B/Indic-Transcribe-Core weights on disk (make sure there's still room for E4B's ~16GB)."
fi

echo ""
echo "=== PHASE 2: Gemma E4B (alone) ==="

python3 gemma_local_text.py --model e4b --transcript-chunk-seconds "$TRANSCRIPT_CHUNK_SECONDS" \
  > logs/parallel_e4b_text.out 2>&1
E4B_STATUS=$?

echo ""
if [ "$E4B_STATUS" -eq 0 ] && [ "$PHASE1_FAILED" -eq 0 ]; then
  echo "=== ALL JOBS COMPLETE ==="
else
  echo "=== DONE, BUT SOMETHING FAILED — check logs/parallel_*.out ==="
fi

echo ""
echo "Next step (not run automatically): semantic WER scoring for the new"
echo "Indic-Transcribe-Core transcripts against the Gemini reference:"
echo "  python3 semantic_wer.py --source indic_transcribe_core --chunk-seconds 2,5,7,10"
