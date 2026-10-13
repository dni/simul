"""The lab notebook: an append-only JSON-lines record of every run.

Each entry is a full certificate (source, raw data, device, verdict) plus
who ran it and why. Nothing is ever rewritten: a later run that contradicts
an earlier one is a new entry, and the disagreement is itself data.
"""
from __future__ import annotations

import json
import pathlib
import time

DEFAULT = pathlib.Path("lab/notebook.jsonl")


def append(cert: dict, path: pathlib.Path = DEFAULT, **meta) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = 0
    if path.exists():
        with path.open(encoding="utf-8") as f:
            entries = sum(1 for _ in f)
    entry = {"id": entries + 1, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **meta, **cert}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry["id"]


def entries(path: pathlib.Path = DEFAULT) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summary(path: pathlib.Path = DEFAULT, limit: int = 50) -> list[dict]:
    """One line per run, without the raw data."""
    out = []
    for e in entries(path)[-limit:]:
        law = e.get("law") or {}
        out.append({"id": e["id"], "time": e["time"], "claim": e["claim"], "verdict": e["verdict"],
                    "by": e.get("by", "?"), "why": e.get("why", ""), "rows": len(e.get("data", [])),
                    "witness": law.get("witness"), "ratio": law.get("ratio")})
    return out
