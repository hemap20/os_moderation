"""Semantic WER for the local Gemma transcripts (transcripts_local/gemma_<run>/),
using semantic_wer.compute_semantic_wer unchanged (same judge, prompt and
scoring as the Indic-Transcribe-Core / Indic-Conformer semantic WER), and a
comparison table against those ASR results on the same calls.

Output: semantic_wer_results/gemma_<run>/<file_id>.json and
semantic_wer_results/local_transcription_semantic_wer.csv.

Usage: python3 semantic_wer_local.py [--runs e2b,e2b_thinking,e4b,e4b_thinking]
"""
import argparse
import csv
import json
import statistics as st
from concurrent.futures import ThreadPoolExecutor

import config
import dataset_v2 as dsv2
import gemini_client
import semantic_wer as sw
from pipeline_logging import StageLogger

ROOT = config.PROJECT_ROOT
BASELINES = {"indic_transcribe_core_10s": ROOT / "semantic_wer_results" / "indic_transcribe_core" / "chunk_10s",
             "indic_conformer_5s": ROOT / "semantic_wer_results" / "indic_conformer" / "chunk_5s"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="e2b,e2b_thinking,e4b,e4b_thinking")
    args = ap.parse_args()
    runs = args.runs.split(",")
    logger = StageLogger("semantic_wer_local")
    client = gemini_client.get_client()
    recs = {r.file_id: r for r in dsv2.load_dataset_v2(config.DOSTT_DIR)}
    fids = sorted(p.stem for p in (ROOT / "transcripts_local" / f"gemma_{runs[0]}").glob("*.json"))

    def score(run, fid):
        out = ROOT / "semantic_wer_results" / f"gemma_{run}" / f"{fid}.json"
        if out.exists():
            return json.loads(out.read_text())
        hyp = json.loads((ROOT / "transcripts_local" / f"gemma_{run}" / f"{fid}.json").read_text())["full_text"]
        ref = sw.load_reference(recs[fid])
        res = sw.compute_semantic_wer(client, fid, 28.0, ref or "", hyp, logger)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(res.model_dump_json(indent=2))
        return json.loads(out.read_text())

    jobs = [(r, f) for r in runs for f in fids]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = dict(zip(jobs, pool.map(lambda j: score(*j), jobs)))

    rows = []
    for name in list(BASELINES) + [f"gemma_{r}" for r in runs]:
        for fid in fids:
            if name in BASELINES:
                d = json.loads((BASELINES[name] / f"{fid}.json").read_text())
            else:
                d = results[(name.removeprefix("gemma_"), fid)]
            cats = {}
            for x in d.get("discrepancies", []):
                cats[x["category"]] = cats.get(x["category"], 0) + 1
            rows.append({"transcriber": name, "file_id": fid, "language": recs[fid].language,
                         "reference_word_count": d["reference_word_count"], "affected_word_count": d["affected_word_count"],
                         "semantic_wer": round(d["semantic_wer"], 4) if d.get("semantic_wer") is not None else "",
                         "n_discrepancies": len(d.get("discrepancies", [])), **{f"n_{c}": cats.get(c, 0) for c in
                         ("number_or_id_altered", "negation_flipped", "named_entity_altered", "key_content_missing",
                          "other_meaning_change")}})
    path = ROOT / "semantic_wer_results" / "local_transcription_semantic_wer.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{'transcriber':28} median  mean   micro")
    for name in list(BASELINES) + [f"gemma_{r}" for r in runs]:
        r = [x for x in rows if x["transcriber"] == name and x["semantic_wer"] != ""]
        s = [x["semantic_wer"] for x in r]
        micro = sum(x["affected_word_count"] for x in r) / sum(x["reference_word_count"] for x in r)
        print(f"{name:28} {st.median(s):.3f}  {st.mean(s):.3f}  {micro:.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
