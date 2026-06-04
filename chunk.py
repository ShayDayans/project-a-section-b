"""Optional preprocessing and chunking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 40
DEFAULT_TITLE_REPEATS = 2
DEFAULT_MAX_CHUNKS_PER_PAGE = 8


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def _chunk_text(title: str, words: List[str]) -> str:
    prefix = " ".join([title] * DEFAULT_TITLE_REPEATS).strip()
    body = " ".join(words).strip()
    if prefix and body:
        return f"{prefix}\n\n{body}"
    return body or prefix


def _window_starts(num_words: int, chunk_size: int, overlap: int, max_chunks: int) -> List[int]:
    if num_words <= chunk_size:
        return [0]
    step = chunk_size - overlap
    starts = list(range(0, num_words - chunk_size + 1, step))
    last = num_words - chunk_size
    if starts[-1] != last:
        starts.append(last)
    if len(starts) <= max_chunks:
        return starts
    spacing = last / float(max_chunks - 1)
    sampled = [int(round(i * spacing)) for i in range(max_chunks)]
    return sorted(set(sampled))


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    page_id = int(record["page_id"])
    title = str(record.get("title", "")).strip()
    words = str(record.get("content", "")).split()

    if not words:
        return [Chunk(page_id=page_id, chunk_id=0, text=title)]

    chunks: List[Chunk] = []
    for chunk_id, start in enumerate(
        _window_starts(len(words), DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_MAX_CHUNKS_PER_PAGE)
    ):
        window = words[start: start + DEFAULT_CHUNK_SIZE]
        chunks.append(Chunk(
            page_id=page_id,
            chunk_id=chunk_id,
            text=_chunk_text(title, window),
        ))
    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
