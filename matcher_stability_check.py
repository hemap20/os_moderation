"""mv1 matcher stability check: runs the FULL global matcher (embeddings +
Hungarian assignment + LLM verify + retry, via analyze_results.resolve_matches)
twice, independently, on EVERY file of a dataset (no pre-pass filtering,
unlike analyze_results.py --stability-check), and reports how often the two
runs produce the exact same final (GT idx, model idx) pair-set per file.

Read-only: writes nothing to analysis_results/ and never touches the per-file
caches. Prints a summary and writes a JSON report (per-file pair-sets for both
runs) next to the log, so any disagreement can be inspected.

Usage:
    .venv/bin/python3 matcher_stability_check.py --dataset-root Dostt_dev --results-tag promptv6 \
        --model e2b_nothinking 2>&1 | tee logs/mv1_stability_check_dev_v6.log
"""
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import analyze_results as ar
import config
import gemini_client
from pipeline_logging import StageLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="Dostt_dev")
    parser.add_argument("--results-tag", type=str, default="promptv6")
    parser.add_argument("--model", type=str, default="e2b_nothinking", help="Model key whose flags are matched")
    parser.add_argument("--match-model", type=str, default=config.GEMINI_MATCH_MODEL, help="LLM used for the verify step")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--report", type=str, default="logs/mv1_stability_check_report.json")
    args = parser.parse_args()

    ar.configure_dataset_root(args.dataset_root, args.results_tag)
    logger = StageLogger("matcher_stability_check")
    client = gemini_client.get_client()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    model_dir = ar.MODEL_DIRS[args.model]
    records = list(ar.unique_records())
    items = [it for it in (ar.build_score_input(r, model_dir) for r in records) if it is not None]
    n_skipped = len(records) - len(items)

    print(f"mv1 stability check  {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"commit={commit}  matcher_version=mv1  match_model={args.match_model}")
    print(f"dataset_root={args.dataset_root}  results_tag={args.results_tag}  model={args.model}")
    print(f"files: {len(records)} total, {len(items)} with GT + model results, {n_skipped} skipped")

    # One file per call, same as production (resolve_matches with a 1-item batch).
    def run_once(item):
        pairs, _ = ar.resolve_matches(client, [item], args.match_model, 1, logger)
        return frozenset(pairs.get(item["file_id"], []))

    runs = []
    for run_idx in (1, 2):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            runs.append(dict(zip((it["file_id"] for it in items), pool.map(run_once, items))))
        logger.info(f"[matcher-stability] run {run_idx} done")

    run1, run2 = runs
    agree, disagreements = 0, []
    n_with_pairs = 0
    for item in items:
        fid = item["file_id"]
        s1, s2 = run1[fid], run2[fid]
        if s1 or s2:
            n_with_pairs += 1
        if s1 == s2:
            agree += 1
        else:
            disagreements.append({"file_id": fid, "run1": sorted(s1), "run2": sorted(s2)})

    n = len(items)
    rate = 100 * agree / n if n else 100.0
    pairs1 = sum(len(s) for s in run1.values())
    pairs2 = sum(len(s) for s in run2.values())
    print(f"\nFinal pair-set agreement: {agree}/{n} files ({rate:.1f}%)")
    print(f"Files with at least one pair in either run: {n_with_pairs}")
    print(f"Total pairs: run1={pairs1}  run2={pairs2}")
    for d in disagreements:
        print(f"  DISAGREE {d['file_id']}: run1={d['run1']} run2={d['run2']}")
    logger.info(f"[matcher-stability] agreement={rate:.1f}% over {n} file(s), {len(disagreements)} disagreement(s)")

    with open(args.report, "w") as f:
        json.dump({
            "commit": commit, "matcher_version": "mv1", "match_model": args.match_model,
            "dataset_root": args.dataset_root, "results_tag": args.results_tag, "model": args.model,
            "n_files": n, "n_skipped": n_skipped, "n_agree": agree, "agreement_pct": rate,
            "n_files_with_pairs": n_with_pairs, "pairs_run1": pairs1, "pairs_run2": pairs2,
            "disagreements": disagreements,
            "run1": {k: sorted(v) for k, v in run1.items()},
            "run2": {k: sorted(v) for k, v in run2.items()},
        }, f, indent=1)
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
