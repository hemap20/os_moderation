# Full-dataset leaderboard

Per call at confidence ≥ 0.8 unless noted; pp_full confidence (raw for v1). \* = partial run (fewer calls scored). Regenerate: `python3 leaderboard.py`.

| # | Model | Input | Prompt | Calls | Balanced acc. | Recall | Specificity | Flag precision | Precision gap | Right flags ≥0.8 | Ambiguous recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 31b_itc10s_text_nothinking | text (Indic-Transcribe-Core) | v8 | 381 | 0.85 | 0.8 | 0.91 | 0.43 | 0.24 | 0.89 | 0.65 |
| 2 | 31b_itc10s_text_thinking | text (Indic-Transcribe-Core) | v8 | 381 | 0.78 | 0.64 | 0.92 | 0.35 | 0.14 | 0.76 | 0.55 |
| 3 | gemini-3.1-flash-lite | audio | v6 | 381 | 0.77 | 0.59 | 0.95 | 0.47 | 0.35 | 0.87 | 0.2 |
| 4 | e4b_thinking \* | audio | v7 | 127 | 0.77 | 0.7 | 0.84 | 0.29 | 0.15 | 0.67 | 0.25 |
| 5 | e4b_thinking \* | audio | v1 | 183 | 0.73 | 0.65 | 0.81 | 0.42 | 0.15 | 0.76 | 0.06 |
| 6 | e4b_nothinking \* | audio | v7 | 330 | 0.73 | 0.55 | 0.9 | 0.31 | 0.25 | 0.73 | 0.17 |
| 7 | e2b_thinking | audio | v1 | 381 | 0.67 | 0.47 | 0.86 | 0.43 | 0.13 | 0.68 | 0.03 |
| 8 | e2b_nothinking | audio | v1 | 381 | 0.64 | 0.38 | 0.89 | 0.46 | 0.36 | 0.59 | 0.05 |
| 9 | e2b_nothinking | audio | v6 | 380 | 0.64 | 0.5 | 0.77 | 0.23 | 0.21 | 0.6 | 0.17 |
| 10 | e2b_thinking \* | audio | v8 | 165 | 0.64 | 0.51 | 0.77 | 0.26 | 0.11 | 0.67 | 0.18 |
| 11 | gemini-3.5-flash-lite | audio | v1 | 377 | 0.62 | 0.27 | 0.97 | 0.76 | 0.22 | 0.92 | 0.0 |
| 12 | 12b_nothinking | audio | v1 | 381 | 0.61 | 0.33 | 0.9 | 0.4 | 0.27 | 0.97 | 0.05 |
| 13 | e2b_conformer5s_text_nothinking | text (Indic-Conformer) | v1 | 381 | 0.6 | 0.27 | 0.93 | 0.57 | 0.36 | 0.65 | 0.03 |
| 14 | e4b_conformer5s_text_nothinking | text (Indic-Conformer) | v1 | 381 | 0.56 | 0.17 | 0.96 | 0.49 | 0.35 | 0.97 | 0.05 |
| 15 | e4b_nothinking | audio | v1 | 380 | 0.55 | 0.12 | 0.99 | 0.68 | 0.48 | 0.96 | 0.0 |
| 16 | gemini-3.5-flash-lite | audio | v6 | 380 | 0.53 | 0.07 | 0.99 | 0.75 | 0.18 | 0.82 | 0.0 |
