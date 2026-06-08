"""Shared paths and helpers for Section B."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

STUDENT_ROOT = Path(__file__).resolve().parent
DATA_DIR = STUDENT_ROOT / "data"
ENTRIES_DIR = DATA_DIR / "Wikipedia Entries"
PUBLIC_QUERIES_PATH = DATA_DIR / "public_queries.json"
ARTIFACTS_DIR = STUDENT_ROOT / "artifacts"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_EVAL = 10

DENSE_INDEX_NAME = "dense_chunks.faiss"
DENSE_META_NAME = "dense_chunks_meta.json"
BM25_BUNDLE_NAME = "bm25_chunks_bundle.json"
BM25_ARRAYS_NAME = "bm25_chunks_arrays.npz"
BM25_META_NAME = "bm25_chunks_meta.json"

CHUNK_TOKEN_TARGET_MIN = 140
CHUNK_TOKEN_TARGET_MAX = 200
CHUNK_SUBCHUNK_OVERLAP = 30


def normalize_page_id(value: Any) -> int:
    """Coerce page_id from JSON (int or numeric string) to int."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"Invalid page_id: {value!r}")


def load_public_queries(path: Path | None = None) -> List[Dict[str, Any]]:
    path = path or PUBLIC_QUERIES_PATH
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        row["relevant_page_ids"] = [
            normalize_page_id(pid) for pid in row["relevant_page_ids"]
        ]
    return rows


def list_entry_paths(entries_dir: Path | None = None) -> List[Path]:
    """Return corpus JSON files in deterministic order."""
    root = entries_dir or ENTRIES_DIR
    if not root.is_dir():
        raise FileNotFoundError(
            f"Corpus directory not found: {root}. "
            "Expected student/data/Wikipedia Entries/ with one JSON file per page."
        )
    return sorted(root.glob("*.json"))


def iter_entries(entries_dir: Path | None = None) -> Iterator[Dict[str, Any]]:
    """Yield one record per JSON file in the corpus directory."""
    for path in list_entry_paths(entries_dir):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["page_id"] = normalize_page_id(data.get("page_id", path.stem))
        yield data


def entry_text(record: Dict[str, Any]) -> str:
    title = record.get("title", "")
    content = record.get("content", "")
    if title:
        return f"{title}\n\n{content}".strip()
    return str(content).strip()


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def tokenize_text(text: str) -> List[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    return normalized.split(" ")


def normalize_tokens(text: str) -> List[str]:
    return [token.lower() for token in tokenize_text(text) if token]


def untokenize_text(tokens: Iterable[str]) -> str:
    return " ".join(token for token in tokens if token)


def split_paragraphs(text: str) -> List[str]:
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    parts = re.split(r"\n\s*\n+", normalized)
    return [normalize_whitespace(part) for part in parts if normalize_whitespace(part)]


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR
