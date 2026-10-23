# Task: Wire up prompt_v6 (final prompt iteration) and run it on the dev set

Copy `prompt_v6.py` and `schemas_v6.py` into the repo root.

## 1. Prompt → schema mapping that cannot fail silently

The v5 run used the wrong schema because `prompt_loader.py` falls back to `schemas` for any prompt it doesn't list. Fix this permanently:

- Each prompt file may declare a module-level `SCHEMA` constant (`prompt_v6.py` has `SCHEMA = "v6"`). In `prompt_loader.schema_module_for_prompt()`:
  - read that constant;
  - map `"v4"` → `schemas_v4` and `"v6"` → `schemas_v6`;
  - fall back to the existing filename map only when the constant is absent.
- Add `"prompt_v5.py": schemas_v4` to the filename map as well, for the record.
- **Raise an error** if a prompt file named `prompt_v<N>.py` with N ≥ 4 resolves to the default `schemas` module. Only `prompt.py` and versions 1–3 may use the default schema.
- Keep the existing ground-truth assertion: `stage2_classify.py` / `stage12_v2.py` must still resolve to `schemas`.

## 2. Plumbing for the new fields

- Add optional `model_speaker` (`spk`) and `model_context` (`ctx`) to `GemmaChunkFlag` in `schemas_gemma.py`, and carry them through to the per-file analysis JSON and `fp_flags.csv`.
- `quote_type` no longer exists in v6. `model_quote_type` stays None for v6 results; that must not break any analysis.
- The `violation` key name is unchanged, so `find_violation_token_index` and the `p_violation_yes` computation should work as they are. **Verify on a dry run**, especially for a thinking model, now that the thinking-offset bug is fixed.
- `gemini_model_eval.py` should pick up the v6 schema through the same mapping.

## 3. Hard check on field compliance

At the end of every run, compute the per-field missing rate across all flags. If any required field (`t, seg, tr, f, spk, ctx, speech_act, j, violation, c`) is missing on more than 20% of flags:
- print a loud warning;
- write `"compliance_warning": true` into the run summary.

Also report how many flags use a category other than the three valid ones (e.g. "Not a violation").

## 4. Analysis additions for v6

In the v6 analysis folder, add:
- the TP rate by `spk` (expert / user / unclear / missing);
- the TP rate by `speech_act` (already exists; make sure v6 is included);
- for every hard-negative file in `manifest/dev_set_v1.csv`, the highest-confidence flag, its `speech_act`, and its `ctx`. This lets me check the safety-advice files directly (`424416192`, `427308167`).

Also run the existing threshold-by-category table, AUROC, FP classification (with the stronger classifier model) and entropy outputs, exactly as for v4.

## 5. Dry run, then the full dev run

1. Dry run on 3 dev files, including `424416192` and `421768524`, with e2b_nothinking and e4b_thinking. Show me:
   - the raw output;
   - the parsed flags with `spk`, `ctx` and `speech_act`;
   - the located violation token and `p_violation_yes`, for both models.
2. If that's clean, run the full 50-file dev set on **e2b_nothinking, e4b_nothinking, e2b_thinking and e4b_thinking**, plus gemini-3.5-flash-lite if possible.
3. Then compare v4 and v6 per model:
   - recall, precision, specificity and FP files at thresholds none / 0.3 / 0.6 / 0.8 (all categories, and per category);
   - TP flags and FN files;
   - AUROC of `model_confidence` and `p_violation_yes`;
   - flags per file;
   - `speech_act` on the hard-negative files.
