"""Dataset ingestion for the new Dostt layout.

    Dostt/
      {language}_{category}/
        TP/
          Single/*.mp3
          Multi/*.mp3
        FP/*.mp3                 (flat, no Single/Multi split)
        FN/                      leftover, empty — ignored
      {language}_fn/*.mp3        NEW: per-language pool, flat, NO category
                                 (category is unknown until classified)
      TN/                        currently empty — TN is decided during the
                                 pipeline run and tracked only in the
                                 manifest, never physically populated here.

Filenames are opaque unique ids, same convention as before.
"""
import dataclasses
from pathlib import Path
from typing import List, Optional

import config

_IGNORED = config.IGNORED_FILENAMES
_EXTS = config.AUDIO_EXTENSIONS


@dataclasses.dataclass(frozen=True)
class FileRecordV2:
    file_id: str
    path: Path
    language: str
    category: Optional[str]  # "pm"/"sa"/"ef" code; None for FN (unknown until classified)
    original_bucket: str  # "TP" / "FP" / "FN"
    violation_multiplicity: Optional[str]  # "single"/"multi" for TP; None otherwise

    @property
    def folder_dir(self) -> Path:
        """The TP/FP/{lang}_fn directory this file lives directly under (TP
        files live one level deeper, in Single/Multi, so folder_dir climbs
        back up to TP itself — that's where transcripts/ and
        raw_responses/ are created, per spec)."""
        if self.original_bucket == "TP":
            return self.path.parent.parent
        return self.path.parent

    @property
    def transcripts_dir(self) -> Path:
        return self.folder_dir / "transcripts"

    @property
    def raw_responses_dir(self) -> Path:
        return self.folder_dir / "raw_responses"


def _iter_audio_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _EXTS and p.name not in _IGNORED
    )


def load_dataset_v2(dataset_dir: Path = config.DOSTT_DIR) -> List[FileRecordV2]:
    records: List[FileRecordV2] = []

    for language in config.LANGUAGES:
        for category in config.CATEGORIES:
            cell_dir = dataset_dir / f"{language}_{category}"
            if not cell_dir.is_dir():
                continue

            tp_dir = cell_dir / "TP"
            for mult_name, mult_label in (("Single", "single"), ("Multi", "multi")):
                for f in _iter_audio_files(tp_dir / mult_name):
                    records.append(FileRecordV2(f.name, f, language, category, "TP", mult_label))

            for f in _iter_audio_files(cell_dir / "FP"):
                records.append(FileRecordV2(f.name, f, language, category, "FP", None))

        fn_dir = dataset_dir / f"{language}_fn"
        for f in _iter_audio_files(fn_dir):
            records.append(FileRecordV2(f.name, f, language, None, "FN", None))

    return records


def summarize(records: Optional[List[FileRecordV2]] = None):
    records = records if records is not None else load_dataset_v2()
    counts = {}
    for r in records:
        key = f"{r.language}_{r.category or 'fn'}_{r.original_bucket}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    recs = load_dataset_v2()
    print(f"Total files: {len(recs)}")
    for k, v in summarize(recs).items():
        print(f"  {k}: {v}")
