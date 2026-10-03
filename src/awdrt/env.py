"""Minimal .env loader (no extra dependency).

Reads KEY=VALUE lines from the repo-root .env into os.environ. Variables already set in the
environment win, so `SOME_KEY=... python ...` still overrides the file.
"""

import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_env(path=ROOT / ".env"):
    path = pathlib.Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)
