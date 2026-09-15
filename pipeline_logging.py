"""Minimal timestamped logging shared by all stages: writes to stdout and to
an append-only log file under logs/, so unattended (Stage 3/4) runs still
leave a persistent trail."""
import datetime
import sys
from pathlib import Path

import config


def _timestamp() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class StageLogger:
    def __init__(self, stage_name: str):
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.path = config.LOGS_DIR / f"{stage_name}.log"
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, message: str, level: str = "INFO"):
        line = f"[{_timestamp()}] [{level}] {message}"
        print(line, file=sys.stdout if level != "ERROR" else sys.stderr)
        self._fh.write(line + "\n")
        self._fh.flush()

    def info(self, message: str):
        self.log(message, "INFO")

    def warn(self, message: str):
        self.log(message, "WARN")

    def error(self, message: str):
        self.log(message, "ERROR")

    def close(self):
        self._fh.close()
