#!/usr/bin/env bash
# Single command: runs all 3 in parallel on ONE GPU —
#   1. Gemma 4 E2B  — text classification of Indic-Conformer transcripts
#   2. Gemma 4 E4B  — text classification of Indic-Conformer transcripts
#   3. Indic-Transcribe-Core — chunked transcription sweep (2/5/7/10s) + metrics
#
# Combined VRAM footprint is roughly E2B (~4-8GB) + E4B (~16GB) +
# Indic-Transcribe-Core (~2-4GB) ≈ 25-30GB. Use a GPU with real headroom
# above that (40GB+ recommended) — if you see CUDA OOM on one of the three,
# that's this constraint, not a bug; rerun that one alone afterward.
#
# UNVERIFIED like everything else Gemma/local-ASR in this session — written
# from the individual scripts' known-working (or documented) behavior, not
# tested running all three concurrently. Watch the three log tails for real
# errors on first run.
#
# Usage:
#   ./run_parallel_stage3_asr.sh
#   TRANSCRIPT_CHUNK_SECONDS=5 ./run_parallel_stage3_asr.sh   # override which
#     Indic-Conformer chunk-size transcripts feed Gemma (default: 5, the
#     best-performing size per the earlier semantic-WER comparison)

set -uo pipefail  # not -e: one job failing shouldn't kill the others

TRANSCRIPT_CHUNK_SECONDS="${TRANSCRIPT_CHUNK_SECONDS:-5}"

mkdir -p logs

echo "=== Starting all 3 jobs in parallel ==="

python3 gemma_local_text.py --model e2b --transcript-chunk-seconds "$TRANSCRIPT_CHUNK_SECONDS" \
  > logs/parallel_e2b_text.out 2>&1 &
PID_E2B=$!
echo "E2B text classification: PID $PID_E2B (log: logs/parallel_e2b_text.out)"

python3 gemma_local_text.py --model e4b --transcript-chunk-seconds "$TRANSCRIPT_CHUNK_SECONDS" \
  > logs/parallel_e4b_text.out 2>&1 &
PID_E4B=$!
echo "E4B text classification: PID $PID_E4B (log: logs/parallel_e4b_text.out)"

python3 indic_transcribe_core_pipeline.py --chunk-seconds 2,5,7,10 \
  > logs/parallel_transcribe_core.out 2>&1 &
PID_TRANSCRIBE=$!
echo "Indic-Transcribe-Core sweep: PID $PID_TRANSCRIBE (log: logs/parallel_transcribe_core.out)"

echo ""
echo "Watch progress with: tail -f logs/parallel_e2b_text.out logs/parallel_e4b_text.out logs/parallel_transcribe_core.out"
echo ""

FAILED=0
wait "$PID_E2B" || { echo "E2B text classification FAILED (exit $?)"; FAILED=1; }
wait "$PID_E4B" || { echo "E4B text classification FAILED (exit $?)"; FAILED=1; }
wait "$PID_TRANSCRIBE" || { echo "Indic-Transcribe-Core sweep FAILED (exit $?)"; FAILED=1; }

echo ""
if [ "$FAILED" -eq 0 ]; then
  echo "=== ALL 3 JOBS COMPLETE ==="
else
  echo "=== DONE, BUT AT LEAST ONE JOB FAILED — check the logs above ==="
fi

echo ""
echo "Next step (not run automatically): semantic WER scoring for the new"
echo "Indic-Transcribe-Core transcripts against the Gemini reference:"
echo "  python3 semantic_wer.py --source indic_transcribe_core --chunk-seconds 2,5,7,10"
