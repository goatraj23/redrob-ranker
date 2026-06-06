"""I/O helpers: stream candidates from .jsonl or .jsonl.gz with low memory."""
from __future__ import annotations
import gzip
import json
from typing import Iterator, Dict, Any


def open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def stream_candidates(path: str) -> Iterator[Dict[str, Any]]:
    """Yield one candidate dict per line. Skips blank/garbled lines safely."""
    with open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_candidates(path: str):
    return list(stream_candidates(path))
