"""Dataset ingestion for Dostt_lite.

Actual on-disk layout (verified directly, not assumed):

    Dostt_lite/
      {language}_{category}/        e.g. hindi_pm, tamil_sa, malayalam_ef
        TP/
          Single/*.mp3
          Multi/*.mp3
        FP/*.mp3
        FN/*.mp3
      TN/*.mp3                      pooled across all languages AND categories,
                                     no further split.

Filenames carry no meaning (confirmed by the user) — they are opaque unique
ids. The full filename is used as `file_id` everywhere (including in any
ground-truth CSV/JSON), and all metadata (language, category, outcome,
single/multi) is derived purely from the folder path.

Only TP is split into Single/Multi on disk; FP and FN are not, so
`violation_multiplicity` is None for those (and for TN, which has no
violations by definition).
"""
import dataclasses
from pathlib import Path
from typing import Dict, List, Optional

import config


@dataclasses.dataclass(frozen=True)
class FileRecord:
    file_id: str  # full filename, e.g. "rec_audio_call_425655492_audio_1788895962923.mp3"
    path: Path
    language: Optional[str]  # None for TN
    category: Optional[str]  # "pm" / "sa" / "ef", None for TN
    outcome: str  # "TP" / "FP" / "FN" / "TN"
    violation_multiplicity: Optional[str]  # "single" / "multi" / None

    @property
    def cell_key(self) -> str:
        if self.outcome == "TN":
            return "TN"
        return f"{self.language}_{self.category}_{self.outcome}"


def _iter_audio_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in config.AUDIO_EXTENSIONS
        and p.name not in config.IGNORED_FILENAMES
    )


def load_dataset(dataset_dir: Path = config.DATASET_DIR) -> List[FileRecord]:
    records: List[FileRecord] = []

    for language in config.LANGUAGES:
        for category in config.CATEGORIES:
            cell_dir = dataset_dir / f"{language}_{category}"
            if not cell_dir.is_dir():
                continue

            tp_dir = cell_dir / "TP"
            for mult_name, mult_label in (("Single", "single"), ("Multi", "multi")):
                for f in _iter_audio_files(tp_dir / mult_name):
                    records.append(
                        FileRecord(f.name, f, language, category, "TP", mult_label)
                    )

            for outcome in ("FP", "FN"):
                for f in _iter_audio_files(cell_dir / outcome):
                    records.append(
                        FileRecord(f.name, f, language, category, outcome, None)
                    )

    for f in _iter_audio_files(dataset_dir / "TN"):
        records.append(FileRecord(f.name, f, None, None, "TN", None))

    return records


def expected_cells() -> List[str]:
    """All (language, category, outcome) cells the dataset is supposed to cover."""
    cells = [
        f"{lang}_{cat}_{outcome}"
        for lang in config.LANGUAGES
        for cat in config.CATEGORIES
        for outcome in config.OUTCOMES
    ]
    cells.append("TN")
    return cells


def find_empty_cells(records: Optional[List[FileRecord]] = None) -> List[str]:
    """Cells with zero files. Generic — does not assume there is exactly one."""
    records = records if records is not None else load_dataset()
    counts: Dict[str, int] = {c: 0 for c in expected_cells()}
    for r in records:
        counts[r.cell_key] = counts.get(r.cell_key, 0) + 1
    return sorted(k for k, v in counts.items() if v == 0)


def summarize(records: Optional[List[FileRecord]] = None) -> Dict[str, int]:
    records = records if records is not None else load_dataset()
    counts: Dict[str, int] = {}
    for r in records:
        counts[r.cell_key] = counts.get(r.cell_key, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    recs = load_dataset()
    print(f"Total files: {len(recs)}")
    for cell, n in summarize(recs).items():
        print(f"  {cell}: {n}")
    empty = find_empty_cells(recs)
    print(f"\nEmpty cells ({len(empty)}):")
    for c in empty:
        print(f"  {c}")
