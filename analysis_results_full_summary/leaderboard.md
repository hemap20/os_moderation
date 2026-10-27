# Full-dataset leaderboard

Per call at confidence ≥ 0.8 unless noted (both precisions use all flags, no threshold); pp_full confidence (raw for v1). \* = partial run (fewer calls scored). Regenerate: `python3 leaderboard.py`.

| # | Model | Input | Prompt | Calls | Balanced acc. | Recall | Specificity | Precision per call (all flags) | Precision per flag (all flags) | Precision gap | Right flags ≥0.8 | Ambiguous recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 31b_nothinking_v8 | text (English, Gemini-translated from audio) | v8 | 381 | 0.88 | 0.83 | 0.92 | 0.84 | 0.35 | 0.28 | 0.93 | 0.6 |
| 2 | gemini-3.1-FL_thinking_v8 | text (English, Gemini-translated from audio) | v8 | 381 | 0.88 | 0.86 | 0.9 | 0.81 | 0.35 | 0.33 | 0.94 | 0.62 |
| 3 | 31b_nothinking_v8 | text (Indic-Transcribe-Core) | v8 | 381 | 0.85 | 0.8 | 0.91 | 0.82 | 0.38 | 0.24 | 0.89 | 0.65 |
| 4 | gemini-3.1-FL_thinking_v8 | text (Indic-Transcribe-Core) | v8 | 381 | 0.85 | 0.84 | 0.87 | 0.77 | 0.41 | 0.36 | 0.94 | 0.65 |
| 5 | gemini-3.5-flash-lite_v8 | text (English, Gemini-translated from audio) | v8 | 381 | 0.85 | 0.79 | 0.91 | 0.86 | 0.45 | 0.27 | 0.92 | 0.5 |
| 6 | gemini-3.1-FL_thinking_v8 | text (English, Gemma 31B translation of ITC) | v8 | 381 | 0.84 | 0.76 | 0.92 | 0.8 | 0.37 | 0.29 | 0.9 | 0.42 |
| 7 | gemini-3.1-FL_nothinking_v8 | text (English, Gemini-translated from audio) | v8 | 381 | 0.84 | 0.85 | 0.84 | 0.81 | 0.42 | 0.34 | 0.96 | 0.6 |
| 8 | gemini-3.1-FL_nothinking_v8 | text (Indic-Transcribe-Core) | v8 | 381 | 0.84 | 0.86 | 0.82 | 0.77 | 0.46 | 0.35 | 0.96 | 0.68 |
| 9 | gemini-3.5-flash-lite_v8 | text (Indic-Transcribe-Core) | v8 | 359 | 0.83 | 0.75 | 0.91 | 0.84 | 0.51 | 0.2 | 0.88 | 0.45 |
| 10 | gemini-3.1-FL_nothinking_v8 | text (English, Gemma 31B translation of ITC) | v8 | 381 | 0.82 | 0.82 | 0.83 | 0.77 | 0.41 | 0.3 | 0.94 | 0.62 |
| 11 | gemini-3.1-FL_thinking_v8 | text (English, Gemma 31B-thinking translation of ITC) | v8 | 381 | 0.82 | 0.75 | 0.9 | 0.78 | 0.35 | 0.26 | 0.9 | 0.45 |
| 12 | gemini-3.1-FL_nothinking_v8 | text (English, Gemma 31B-thinking translation of ITC) | v8 | 381 | 0.82 | 0.81 | 0.84 | 0.77 | 0.42 | 0.29 | 0.94 | 0.55 |
| 13 | 31b_thinking_v8 | text (English, Gemini-translated from audio) | v8 | 381 | 0.8 | 0.67 | 0.94 | 0.82 | 0.29 | 0.18 | 0.78 | 0.47 |
| 14 | 31b_thinking_v8 | text (Indic-Transcribe-Core) | v8 | 381 | 0.78 | 0.64 | 0.92 | 0.79 | 0.3 | 0.14 | 0.76 | 0.55 |
| 15 | e4b_nothinking_v8 | text (Indic-Transcribe-Core) | v8 | 381 | 0.78 | 0.63 | 0.93 | 0.89 | 0.38 | 0.22 | 0.89 | 0.23 |
| 16 | gemini-3.1-FL_thinking_v8 | audio | v8 | 380 | 0.78 | 0.68 | 0.87 | 0.7 | 0.24 | 0.39 | 0.91 | 0.35 |
| 17 | gemini-3.1-FL_nothinking_v8 | audio | v8 | 380 | 0.78 | 0.69 | 0.87 | 0.69 | 0.3 | 0.39 | 0.93 | 0.3 |
| 18 | gemini-3.1-FL_thinking_v6 | audio | v6 | 381 | 0.77 | 0.59 | 0.95 | 0.78 | 0.33 | 0.35 | 0.87 | 0.2 |
| 19 | e4b_thinking_v7 \* | audio | v7 | 127 | 0.77 | 0.7 | 0.84 | 0.75 | 0.22 | 0.15 | 0.67 | 0.25 |
| 20 | e4b_thinking_v1 \* | audio | v1 | 183 | 0.73 | 0.65 | 0.81 | 0.76 | 0.37 | 0.15 | 0.76 | 0.06 |
| 21 | e4b_nothinking_v7 \* | audio | v7 | 330 | 0.73 | 0.55 | 0.9 | 0.73 | 0.14 | 0.25 | 0.73 | 0.17 |
| 22 | gemini-3.1-FL_thinking_v1 | audio | v1 | 381 | 0.71 | 0.49 | 0.92 | 0.87 | 0.66 | 0.03 | 0.87 | 0.0 |
| 23 | gemini-3.1-FL_nothinking_v1 | audio | v1 | 380 | 0.71 | 0.52 | 0.89 | 0.83 | 0.6 | 0.29 | 0.86 | 0.0 |
| 24 | e2b_nothinking_v8 | text (Indic-Transcribe-Core) | v8 | 381 | 0.69 | 0.52 | 0.86 | 0.67 | 0.34 | 0.29 | 0.66 | 0.15 |
| 25 | e2b_thinking_v1 | audio | v1 | 381 | 0.67 | 0.47 | 0.86 | 0.78 | 0.38 | 0.13 | 0.68 | 0.03 |
| 26 | e2b_nothinking_v1 | audio | v1 | 381 | 0.64 | 0.38 | 0.89 | 0.67 | 0.19 | 0.36 | 0.59 | 0.05 |
| 27 | e2b_nothinking_v6 | audio | v6 | 380 | 0.64 | 0.5 | 0.77 | 0.6 | 0.04 | 0.21 | 0.6 | 0.17 |
| 28 | e2b_thinking_v8 \* | audio | v8 | 165 | 0.64 | 0.51 | 0.77 | 0.66 | 0.21 | 0.11 | 0.67 | 0.18 |
| 29 | gemini-3.5-flash-lite_v1 | audio | v1 | 377 | 0.62 | 0.27 | 0.97 | 0.92 | 0.74 | 0.22 | 0.92 | 0.0 |
| 30 | 12b_nothinking_v1 | audio | v1 | 381 | 0.61 | 0.33 | 0.9 | 0.79 | 0.38 | 0.27 | 0.97 | 0.05 |
| 31 | e2b_nothinking_v1 | text (Indic-Conformer) | v1 | 381 | 0.6 | 0.27 | 0.93 | 0.69 | 0.36 | 0.36 | 0.65 | 0.03 |
| 32 | e4b_nothinking_v1 | text (Indic-Conformer) | v1 | 381 | 0.56 | 0.17 | 0.96 | 0.86 | 0.45 | 0.35 | 0.97 | 0.05 |
| 33 | e4b_nothinking_v1 | audio | v1 | 380 | 0.55 | 0.12 | 0.99 | 0.87 | 0.62 | 0.48 | 0.96 | 0.0 |
| 34 | gemini-3.5-flash-lite_v6 | audio | v6 | 380 | 0.53 | 0.07 | 0.99 | 0.95 | 0.71 | 0.18 | 0.82 | 0.0 |
| 35 | gemini-3.5-flash-lite_v8 | audio | v8 | 378 | 0.52 | 0.05 | 0.99 | 0.93 | 0.59 | -0.08 | 0.88 | 0.03 |
