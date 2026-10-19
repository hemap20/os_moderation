"""Semantic error rate of the English translations against the Gemini
ground-truth transcript (original language), with a focus on whether
explicit TONE and INTENT survive translation.

Same idea as semantic_wer.py (LLM-judged, only meaning-changing errors count,
semantic error rate = reference words affected / reference words), but
cross-lingual: REFERENCE = ground-truth transcript in the original language,
HYPOTHESIS = an English translation of the same call. Two hypotheses:
  audio  translations/audio_gemini31fl/   (Gemini translated straight from audio)
  itc    translations/itc10s_gemini31fl/  (Gemini translated the ITC ASR transcript)

The judge (gemini-3.5-flash) is a different, stronger model than the one that
translated (gemini-3.1-flash-lite), so it isn't grading its own output.

Per call and source it saves:
  - discrepancies: meaning-changing errors with category and reference words affected
    (categories: semantic_wer.py's five + tone_softened, intent_changed, added_content)
  - policy_moments: every policy-relevant moment in the reference (sexual/explicit,
    money/payment, moving off the app, meeting) rated preserved / softened /
    intent_changed / missing in the translation
  - tone_preserved / intent_preserved: overall yes / partly / no

Output: semantic_translation_results/<source>/<file_id>.json, plus
semantic_translation_results/summary_*.csv (written by `summary`).

Usage:
  python3 semantic_translation_eval.py run [--sources audio,itc] [--limit N] [--files a,b] [--workers 6]
  python3 semantic_translation_eval.py summary
"""
import argparse
import csv
import json
import statistics as st
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
import dataset_v2 as dsv2
import gemini_client as gc

JUDGE = "gemini-3.5-flash"
OUT = config.PROJECT_ROOT / "semantic_translation_results"
SOURCES = {"audio": config.PROJECT_ROOT / "translations" / "audio_gemini31fl",
           "itc": config.PROJECT_ROOT / "translations" / "itc10s_gemini31fl",
           # Local Gemma translations of the same ITC transcripts (10-call speed test)
           **{f"itc_gemma_{m}": config.PROJECT_ROOT / "translations" / f"itc10s_gemma_{m}"
              for m in ("e2b", "e2b_thinking", "e4b", "e4b_thinking")}}
CATS = ["number_or_id_altered", "negation_flipped", "named_entity_altered", "key_content_missing",
        "other_meaning_change", "tone_softened", "intent_changed", "added_content"]
MOMENT_TYPES = ["sexual_or_explicit", "money_or_payment", "move_off_app", "meeting_in_person"]
MOMENT_STATUS = ["preserved", "softened", "intent_changed", "missing"]

PROMPT = """
You are checking an ENGLISH TRANSLATION of a phone call against the REFERENCE transcript of the same call.
The REFERENCE is in {language} (native script) and is assumed accurate. The TRANSLATION was produced
separately and may be incomplete, softened, or wrong. Both have MM:SS timestamps; they may not line up
exactly, so compare by content and approximate time, not line by line.

The call is from "Friends", an app where a user talks to a paid expert. What matters most is whether
explicit content and the speakers' intent survive translation.

1. List every discrepancy where the translation changes the MEANING of the reference. IGNORE: wording or
   style differences, literal vs idiomatic phrasing that keeps the meaning, filler words, repetitions,
   hesitations, and lines that are split or merged differently. Categories:
   - number_or_id_altered: a number, phone number, ID, amount or username is changed or missing
   - negation_flipped: a negation is added, dropped or reversed
   - named_entity_altered: a person, place or app/platform name is changed
   - key_content_missing: reference content that carries meaning is absent from the translation
   - other_meaning_change: the translation says something that misrepresents the reference
   - tone_softened: explicit, sexual, crude or aggressive language is toned down, censored or made vague
   - intent_changed: what the speaker is trying to do changes (e.g. a request becomes a statement, a
     proposal becomes a refusal, a joke or warning becomes a real request, or vice versa)
   - added_content: the translation contains meaning that is not in the reference
   For each: reference_text (the exact reference span, empty for added_content), translation_text (what
   the translation says, empty if missing), category, word_count_affected (number of REFERENCE words
   affected; 0 for added_content), explanation (one short sentence).

2. List every POLICY-RELEVANT moment in the reference, i.e. any line that involves:
   - sexual_or_explicit: sexual talk, sexual acts, nudity, showing the body
   - money_or_payment: money, payment, gifts, recharges
   - move_off_app: WhatsApp, Instagram, phone numbers, phone calls or any other app or contact outside this app
   - meeting_in_person: meeting or visiting in person
   For each: t (reference timestamp), type, reference_text, translation_text (the corresponding English,
   empty if missing), status:
   - preserved: same meaning AND same tone/explicitness AND same intent
   - softened: meaning kept but the explicitness or crudeness is reduced
   - intent_changed: the speaker's intent differs (request vs refusal, real vs joke/warning/story)
   - missing: not present in the translation

3. Overall: tone_preserved (yes / partly / no) = is the explicitness and emotional tone of the call kept?
   intent_preserved (yes / partly / no) = are the speakers' intentions kept? One-sentence summary.

[REFERENCE]
{reference}

[TRANSLATION]
{translation}
""".strip()

SCHEMA = {
    "type": "object",
    "properties": {
        "discrepancies": {"type": "array", "items": {"type": "object", "properties": {
            "reference_text": {"type": "string"}, "translation_text": {"type": "string"},
            "category": {"type": "string", "enum": CATS}, "word_count_affected": {"type": "integer"},
            "explanation": {"type": "string"}},
            "required": ["reference_text", "translation_text", "category", "word_count_affected", "explanation"]}},
        "policy_moments": {"type": "array", "items": {"type": "object", "properties": {
            "t": {"type": "string"}, "type": {"type": "string", "enum": MOMENT_TYPES},
            "reference_text": {"type": "string"}, "translation_text": {"type": "string"},
            "status": {"type": "string", "enum": MOMENT_STATUS}},
            "required": ["t", "type", "reference_text", "translation_text", "status"]}},
        "tone_preserved": {"type": "string", "enum": ["yes", "partly", "no"]},
        "intent_preserved": {"type": "string", "enum": ["yes", "partly", "no"]},
        "summary": {"type": "string"},
    },
    "required": ["discrepancies", "policy_moments", "tone_preserved", "intent_preserved", "summary"],
}


def reference(rec):
    p = rec.transcripts_dir / f"{rec.file_id}.json"
    if not p.exists():
        return None, 0
    d = json.loads(p.read_text())
    if d.get("status") != "success":
        return None, 0
    t = d["transcript"]
    lines = "\n".join(f"[{s.get('t')}] {s.get('text', '')}" for s in t.get("segments", []) if s.get("text"))
    return lines, len(t.get("full_text", "").split())


def translation(source, file_id):
    p = SOURCES[source] / f"{file_id}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return "\n".join(f"[{s['t']}] {s['en']}" for s in d.get("segments", []))


def evaluate(client, rec, source):
    ref, ref_words = reference(rec)
    hyp = translation(source, rec.file_id)
    if ref is None or hyp is None:
        raise FileNotFoundError(f"missing {'reference' if ref is None else 'translation'}")
    t0 = time.time()
    prompt = PROMPT.format(language=rec.language.capitalize(), reference=ref, translation=hyp)
    text = gc.call_with_retries(lambda: gc.generate_text(client, JUDGE, [prompt], response_json_schema=SCHEMA))
    j = gc.parse_json_lenient(text)
    affected = sum(max(0, int(d.get("word_count_affected") or 0)) for d in j.get("discrepancies", []))
    out = {
        "file_id": rec.file_id, "language": rec.language, "source": source, "judge": JUDGE,
        "reference_word_count": ref_words, "affected_word_count": affected,
        "semantic_error_rate": round(min(1.0, affected / ref_words), 4) if ref_words else None,
        "seconds": round(time.time() - t0, 1), **j,
    }
    d = OUT / source
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{rec.file_id}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def run(args):
    seen, recs = set(), []
    for r in dsv2.load_dataset_v2(config.DOSTT_DIR):
        if r.file_id not in seen:
            seen.add(r.file_id)
            recs.append(r)
    if args.files:
        want = set(args.files.split(","))
        recs = [r for r in recs if r.file_id in want]
    jobs = [(r, s) for s in args.sources.split(",") for r in recs if not (OUT / s / f"{r.file_id}.json").exists()]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} evaluation(s) with judge {JUDGE}", flush=True)
    client = gc.get_client()
    done, failed, lock = 0, [], threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(evaluate, client, r, s): (r, s) for r, s in jobs}
        for fut in as_completed(futs):
            r, s = futs[fut]
            try:
                o = fut.result()
                with lock:
                    done += 1
                print(f"[{done}/{len(jobs)}] {s:5} {r.file_id}: SER={o['semantic_error_rate']} "
                      f"moments={len(o['policy_moments'])} tone={o['tone_preserved']} intent={o['intent_preserved']} "
                      f"({o['seconds']}s)", flush=True)
            except Exception as exc:  # noqa: BLE001
                failed.append((s, r.file_id, str(exc)[:200]))
                print(f"FAILED {s} {r.file_id}: {str(exc)[:200]}", flush=True)
    print(f"done={done} failed={len(failed)}")
    for f in failed:
        print("  failed:", *f)


def summary(_args=None):
    rows = []
    for s in SOURCES:
        for p in sorted((OUT / s).glob("*.json")):
            rows.append(json.loads(p.read_text()))
    if not rows:
        print("no results yet")
        return
    per_call, by_group, cats, moments = [], defaultdict(list), Counter(), Counter()
    for r in rows:
        mc = Counter(m["status"] for m in r["policy_moments"])
        per_call.append({"source": r["source"], "file_id": r["file_id"], "language": r["language"],
                         "reference_word_count": r["reference_word_count"], "affected_word_count": r["affected_word_count"],
                         "semantic_error_rate": r["semantic_error_rate"], "tone_preserved": r["tone_preserved"],
                         "intent_preserved": r["intent_preserved"], "n_policy_moments": len(r["policy_moments"]),
                         **{f"moments_{k}": mc[k] for k in MOMENT_STATUS},
                         **{f"err_{c}": sum(1 for d in r["discrepancies"] if d["category"] == c) for c in CATS}})
        for key in (("ALL",), (r["language"],)):
            by_group[(r["source"],) + key].append(r)
        for d in r["discrepancies"]:
            cats[(r["source"], d["category"])] += 1
        for m in r["policy_moments"]:
            moments[(r["source"], m["type"], m["status"])] += 1
    write(OUT / "summary_per_call.csv", per_call)

    agg = []
    for (src, lang), rs in sorted(by_group.items()):
        ser = [r["semantic_error_rate"] for r in rs if r["semantic_error_rate"] is not None]
        ref_w = sum(r["reference_word_count"] for r in rs)
        aff = sum(r["affected_word_count"] for r in rs)
        ms = [m for r in rs for m in r["policy_moments"]]
        agg.append({"source": src, "language": lang, "n_calls": len(rs),
                    "micro_semantic_error_rate": round(aff / ref_w, 4) if ref_w else None,
                    "mean_semantic_error_rate": round(st.mean(ser), 4) if ser else None,
                    "median_semantic_error_rate": round(st.median(ser), 4) if ser else None,
                    "tone_preserved_yes": round(sum(r["tone_preserved"] == "yes" for r in rs) / len(rs), 3),
                    "tone_preserved_no": round(sum(r["tone_preserved"] == "no" for r in rs) / len(rs), 3),
                    "intent_preserved_yes": round(sum(r["intent_preserved"] == "yes" for r in rs) / len(rs), 3),
                    "intent_preserved_no": round(sum(r["intent_preserved"] == "no" for r in rs) / len(rs), 3),
                    "policy_moments": len(ms),
                    **{f"moments_{k}_rate": round(sum(m["status"] == k for m in ms) / len(ms), 3) if ms else None
                       for k in MOMENT_STATUS}})
    write(OUT / "summary_by_source_language.csv", agg)
    write(OUT / "summary_error_categories.csv",
          [{"source": s, "category": c, "count": cats[(s, c)]} for s in SOURCES for c in CATS])
    write(OUT / "summary_policy_moments.csv",
          [{"source": s, "type": t, **{k: moments[(s, t, k)] for k in MOMENT_STATUS}} for s in SOURCES for t in MOMENT_TYPES])
    for a in agg:
        if a["language"] == "ALL":
            print(a)


def write(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--sources", default="audio,itc")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--files", default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else summary(a)
