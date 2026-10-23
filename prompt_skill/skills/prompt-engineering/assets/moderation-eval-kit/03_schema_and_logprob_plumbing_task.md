# Task: Support prompt_v4's new output fields, without touching ground truth

`prompt_v4.py` (in the repo root) asks the model for three new fields in each flag, in this order:

```
t, seg, tr, f, speech_act, quote_type, j, violation, c
```

- `speech_act`: "direct" | "reported" | "hypothetical" | "denial"
- `quote_type`: "verbatim" | "paraphrase"
- `violation`: "yes" | "no"

## 1. Keep ground truth isolated (most important)

`schemas.py`'s `RawModelOutput` / `raw_model_output_json_schema()` is shared with `stage2_classify.py`, which generates ground truth. **Do not modify it.** Ground truth must stay exactly as generated with the v1 prompt and schema.

- Create a new `schemas_v4.py` with `FlagResultV4` and `RawModelOutputV4`. List the fields in the order above; the field order in the JSON schema is part of the prompt.
- Add a prompt → schema mapping, for example in `config.py` or `prompt_loader.py`: `prompt_v4.py` uses the v4 schema, and every other prompt uses the current schema.
- Use that mapping in `gemma_local.py` and `gemini_model_eval.py`, both for the `{json_schema_str}` substitution and for parsing. `stage2_classify.py` must keep using `schemas.py` whatever the setting. Add an assertion for this.
- For `gemini_model_eval.py`: if it passes a `response_schema` to Gemini, pass the v4 one. Otherwise just substitute the v4 schema into the prompt.

## 2. Store the new fields

Add to `GemmaChunkFlag` in `schemas_gemma.py` (as optional fields, so old results still load):

- `model_speech_act`, `model_quote_type`, `model_violation`;
- `logprob_violation`: the logprob of the first token of the `violation` value;
- `p_violation_yes`:
  - if the generated value is "yes", this is `exp(logprob_violation)`;
  - if it is "no", compute `exp(logprob of " yes"-equivalent token)` from the top-k at that position when it's available. Otherwise use `1 - exp(logprob_violation)`, and record which method was used in `p_violation_method`.

If `gemma_local.py` only stores the chosen token's logprob, extend `token_logprobs_and_entropy` to also keep the logprobs of the top 5 alternatives at each position. Store them for the violation-token position only, to keep the files small.

- Locate the violation token the same way the category span is found now: the value span after the `"violation"` key, within that flag's own JSON object.
- Check that the locator finds the right token on 3 real outputs, and print the token text and the logprob for each.

Keep the existing `logprob_decision`, `logprob_category` and `logprob_derived_confidence` unchanged, so v1–v3 results can be compared directly.

For flags where a field is missing (`c` has been missing in about 10% of e2b_nothinking flags), store `None`. Never impute values at parse time, and count the missing fields per field in the run summary.

## 3. Analysis changes (`analyze_results.py` and the AUPRC script)

- **Every flag counts as a model flag in TP/FP/FN, including `violation: "no"` ones.** The recall ceiling is the recall when every flag is kept.
- **Rank on these signals:** `model_confidence`, `logprob_derived_confidence`, negative entropy, and the new `p_violation_yes`.
  - For each, report AUROC, AUPRC, and recall at an FP budget of 10, 25, 50 or 100.
  - Report these overall, per language and per category.
- **Missing values:** rank missing-score flags *below* all others (treat them as the lowest score) instead of dropping them. Report `n_missing` per signal alongside the metrics. Also re-run v1–v3 with this rule so every version uses the same method.
- **New breakdowns for v4:**
  - TP rate by `speech_act`, by `quote_type`, and by `violation`;
  - a 2×2 of `violation` (yes/no) against matched (TP/FP).

  These show whether the model's own labels carry information.

## 4. Before running

Run `--dry-run` on 3 dev files with e2b_nothinking. Show me:

- the raw output;
- the parsed flags with the new fields;
- the located violation token, its logprob and `p_violation_yes`;
- confirmation that `stage2_classify.py` still renders the old schema.

Then run v4 on the dev set with e2b_nothinking and e4b_nothinking, and with gemini-3.5-flash-lite if it's cheap. Gemini has no logprobs, so only its categorical fields can be analysed.
