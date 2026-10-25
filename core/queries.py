"""Load named SQL blocks from sql/analytics_queries.sql (aiosql-style: `-- name: foo`)."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_SQL_FILE = Path(__file__).resolve().parents[1] / "sql" / "analytics_queries.sql"


@lru_cache(maxsize=1)
def load_queries(path: str | None = None) -> dict[str, str]:
    text = Path(path or _SQL_FILE).read_text(encoding="utf-8")
    queries: dict[str, str] = {}
    for block in re.split(r"^-- name:\s*", text, flags=re.M)[1:]:
        name, _, body = block.partition("\n")
        queries[name.strip()] = body.strip().rstrip(";")
    return queries


def get_query(name: str) -> str:
    return load_queries()[name]
