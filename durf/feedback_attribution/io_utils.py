"""Small JSONL and CSV helpers for attribution scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def as_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def as_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return value.lower() in {"1", "true", "yes"}


def as_json(value: str | None):
    if value is None or value == "":
        return None
    return json.loads(value)
