"""Shared pieces for the Laya zero-shot moderation experiment (dev set only):
transcript loading for both sources, question building from prompt_v8.py at
runtime, transcript windowing, and the Laya router. See laya_experiment.py."""
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import config

DEV_MANIFEST = config.PROJECT_ROOT / "manifest" / "dev_set_v1.csv"
PROMPT_FILE = config.PROJECT_ROOT / "prompt_v8.py"
CONFORMER_DIR = config.PROJECT_ROOT / "indic_conformer_results" / "chunk_10s" / "transcripts"
GT_ROOT = config.PROJECT_ROOT / "Dostt_dev"
OUT_ROOT = config.PROJECT_ROOT / "analysis_results_dev_laya"
CATEGORIES = {"platform_move": "PlatformMove", "suspicious_activity": "SuspiciousActivity",
              "explicit_flirting": "Explicit-Flirting"}
MODEL = "multilingual"


def dev_files() -> List[dict]:
    with DEV_MANIFEST.open() as fh:
        return list(csv.DictReader(fh))


def mmss_to_sec(t: str) -> Optional[float]:
    m = re.fullmatch(r"\s*(\d+):(\d{1,2})(?::(\d{1,2}))?\s*", t or "")
    if not m:
        return None
    a, b, c = m.groups()
    return int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)


def load_segments(source: str, file_id: str) -> List[dict]:
    """[{t_sec, text}] for a file. conformer = Indic-Conformer 5 s chunks;
    gt = the Gemini ground-truth transcript under Dostt_dev/<cell>/transcripts/."""
    if source == "conformer":
        d = json.loads((CONFORMER_DIR / f"{file_id}.json").read_text())
        return [{"t_sec": mmss_to_sec(s.get("t")), "text": s.get("text", "")} for s in d.get("segments", [])]
    hits = list(GT_ROOT.glob(f"*/transcripts/{file_id}.json")) + list(GT_ROOT.glob(f"*/*/transcripts/{file_id}.json"))
    if not hits:
        raise FileNotFoundError(f"no GT transcript for {file_id}")
    d = json.loads(hits[0].read_text())
    return gt_segments(d)


def gt_segments(d) -> List[dict]:
    """GT transcript JSON -> [{t_sec, text}]; tolerant of the field names used
    by stage1_transcribe.py's output."""
    inner = d.get("transcript")
    if isinstance(inner, dict):
        d = inner
    segs = d.get("segments") or d.get("utterances") or (d.get("transcript") if not isinstance(inner, dict) else None) or []
    out = []
    if isinstance(segs, str):
        return [{"t_sec": None, "text": segs}]
    for s in segs:
        if isinstance(s, str):
            out.append({"t_sec": None, "text": s})
            continue
        t = s.get("t") or s.get("timestamp") or s.get("start")
        t_sec = mmss_to_sec(t) if isinstance(t, str) else (float(t) if isinstance(t, (int, float)) else None)
        text = s.get("text") or s.get("native") or s.get("transcript") or ""
        spk = s.get("speaker") or s.get("spk")
        out.append({"t_sec": t_sec, "text": f"{spk}: {text}" if spk else text})
    return out


def full_text(segs: List[dict]) -> str:
    return "\n".join(s["text"] for s in segs if s["text"])


def window_text(segs: List[dict], center: float, half: float) -> str:
    return "\n".join(s["text"] for s in segs
                     if s["t_sec"] is not None and center - half <= s["t_sec"] <= center + half and s["text"])


def fixed_windows(segs: List[dict], width: float) -> List[str]:
    """Non-overlapping windows of `width` seconds by segment timestamp."""
    buckets: Dict[int, List[str]] = {}
    for s in segs:
        if s["t_sec"] is None or not s["text"]:
            continue
        buckets.setdefault(int(s["t_sec"] // width), []).append(s["text"])
    return ["\n".join(buckets[k]) for k in sorted(buckets)]


# ---------------------------------------------------------------- questions

def prompt_sections() -> Dict[str, str]:
    """The category sections and the About section of prompt_v8.py, verbatim,
    from its rendered text — sections start at a **[Name]** marker and run to
    the next **[...]** marker or the next top-level heading."""
    import prompt_loader
    text = prompt_loader.load_raw_prompt(PROMPT_FILE)
    marks = [(m.start(), m.group(1)) for m in re.finditer(r"\*\*\[([^\]]+)\]\*\*", text)]
    out = {}
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[pos:end]
        nxt = re.search(r"\n\s*(#{1,3} |\*\*[A-Z][A-Z ]+\*\*)", body[10:])
        if nxt:
            body = body[:10 + nxt.start()]
        out[name] = re.sub(r"\n{3,}", "\n\n", body.strip())
    return out


FRAMING = ("Using this policy definition, does a speaker in this transcript commit this violation "
           "(a strong or weaker case, not an allowed item)?\n\n")
ABOUT_FRAMING = "Is the relevant speech only about a violation rather than doing it?\n\n"


def _first_line(section: str) -> str:
    lines = [l.strip() for l in section.splitlines() if l.strip()]
    return lines[1] if len(lines) > 1 else lines[0]


def build_questions(defs: str, mode: str, quote_english: str = "") -> dict:
    """defs = full | short | full_ctx | short_ctx (the *_ctx variants add prompt_v8's [Context]
    section, and full_ctx also its [About a Violation, Not Doing It] rules). mode = A (category questions) | B (+ about_not_doing,
    quote_supported)."""
    secs = prompt_sections()

    def body(name):
        s = secs[name]
        if defs == "full":
            return s
        lines = [l.strip() for l in s.splitlines() if l.strip()]
        return lines[1] if len(lines) > 1 else lines[0]  # first line after the [Name] marker

    if defs in ("full_ctx", "short_ctx"):
        # Same policy text Gemma gets: app context + the category definition
        # (+ the "about, not doing" rules for full_ctx), question last.
        ctx = secs["Context"]
        about = secs["About a Violation, Not Doing It"]

        def cat_q(cat):
            d = secs[cat] if defs == "full_ctx" else _first_line(secs[cat])
            parts = [ctx, d] + ([about] if defs == "full_ctx" else [])
            return ("\n\n".join(parts) + "\n\nQuestion: does a speaker in this transcript commit this "
                    f"{cat} violation (a strong or weaker case, not an allowed item and not just talking about it)?")
        q = {qid: {"type": "noul", "instructions": cat_q(cat)} for qid, cat in CATEGORIES.items()}
        if mode == "B":
            q["about_not_doing"] = {"type": "noul", "instructions": ctx + "\n\n" + about +
                                    "\n\nQuestion: is the relevant speech only about a violation rather than doing it?"}
            q["quote_supported"] = {"type": "noul", "instructions":
                                    f"Does the transcript contain words with the same meaning as this quote: «{quote_english}»?"}
        return q

    q = {qid: {"type": "noul", "instructions": FRAMING + body(cat)} for qid, cat in CATEGORIES.items()}
    if mode == "B":
        q["about_not_doing"] = {"type": "noul", "instructions": ABOUT_FRAMING + body("About a Violation, Not Doing It")}
        q["quote_supported"] = {"type": "noul", "instructions":
                                f"Does the transcript contain words with the same meaning as this quote: «{quote_english}»?"}
    return q


_ROUTER = None


def router():
    global _ROUTER
    if _ROUTER is None:
        from laya import Router
        _ROUTER = Router(preload=True, default=MODEL)
    return _ROUTER


def tokenizer():
    return router().load(MODEL).tok


def ntok(text: str) -> int:
    return len(tokenizer()(text, add_special_tokens=False)["input_ids"])


def save_raw(mode: str, source: str, defs: str, ctx: str, record: dict, run_name: str = "raw") -> Path:
    """Append one request's full Laya output to
    analysis_results_dev_laya/<mode>/<source>/<defs>/<ctx>/<run_name>.jsonl.
    `record` should carry the identifiers (file_id, candidate fields), the
    state sent, the request settings and Laya's untouched result dict."""
    d = OUT_ROOT / f"mode_{mode}" / source / defs / ctx
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{run_name}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path
