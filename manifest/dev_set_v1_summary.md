# Dev set v1 — 50-file stratified dev set

Frozen. Do not overwrite — a rebuild goes to `dev_set_v2`.

## Deviations from the original spec (both explicitly authorized)

- **`e4b_thinking` excluded from all signals** — its inference run is only 48% complete (183/381 files) on the full dataset, unrelated to the matching fix. `n_models_flagged`/`n_models_correct`/`n_models_fp_flags` are computed from the remaining **7** models.

- **Disagreement metric adjusted accordingly**: "closest to 4 of 8" -> "closest to 3.5 of 7" (the natural half-point for 7 models).


## Counts per language x slot

| language | pos_PlatformMove | pos_SuspiciousActivity | pos_ExplicitFlirting | pos_hard_miss | neg_hard_clean | neg_easy_clean | total |
|---|---|---|---|---|---|---|---|
| hindi | 1 | 1 | 1 | 2 | 3 | 2 | 10 |
| tamil | 1 | 1 | 1 | 2 | 3 | 2 | 10 |
| telugu | 1 | 1 | 1 | 2 | 3 | 2 | 10 |
| kannada | 1 | 1 | 1 | 2 | 3 | 2 | 10 |
| malayalam | 1 | 1 | 1 | 2 | 3 | 2 | 10 |

## Counts per category (positives)

| category | count |
|---|---|
| PlatformMove | 18 |
| SuspiciousActivity | 10 |
| Explicit-Flirting | 17 |

## Validation checks

- Positives per category >= 5: PlatformMove=18, SuspiciousActivity=10, Explicit-Flirting=17  -> PASS
- Total GT flags >= 30: 73  -> PASS
- Positives with >1 GT flag (Multi) >= 10: 23  -> PASS

## Other stats

- Total files: 50
- Total audio duration: 8413.5s (140.2 min)
- Substitutions: 2
  - telugu: `rec_default_427305885_audio_1789058218067.mp3` -> slot `neg_hard_clean` (substituted from neg_easy_clean (slot neg_hard_clean had a shortfall))
  - telugu: `rec_default_427307149_audio_1789058314884.mp3` -> slot `neg_hard_clean` (substituted from neg_easy_clean (slot neg_hard_clean had a shortfall))
