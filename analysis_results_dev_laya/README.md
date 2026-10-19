# Laya raw outputs (dev set)

Layout: `mode_<A|B>/<source>/<defs>/<ctx>/<run>.jsonl`, one JSON line per Laya request.

- source: `conformer` (Indic-Conformer 5 s ASR) or `gt` (Gemini ground-truth transcript)
- defs: `short` (one sentence per category) or `full` (full prompt_v8.py sections); exact text in `questions_<defs>.json` (`{quote_english}` is filled per candidate)
- ctx (Mode B): `quote_only`, `pm10` / `pm30` = transcript ±10 / ±30 s around the candidate's timestamp

Each line: model, prompt_version, file_id, flag_index, candidate (Gemma flag fields incl. `matched`), gemma_pp_full,
ts_unreliable, request settings (max_len, head_max_len, latency_ms), the exact `state` sent, question_ids,
and `laya` = Laya's untouched result (answers with noul / confidence / answer_confidence, usage incl. truncation, routing).

Runs so far:
- `sample12_e2b_nothinking_v6`: 12 random E2B non-thinking (v6) candidates with pp_full ≥ 0.3 (6 matched, 6 not; seed 1), all configs above.
