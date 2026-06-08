"""Paragraph-first preprocessing and chunking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from utils import (
    CHUNK_SUBCHUNK_OVERLAP,
    CHUNK_TOKEN_TARGET_MAX,
    CHUNK_TOKEN_TARGET_MIN,
    normalize_whitespace,
    split_paragraphs,
    tokenize_text,
    untokenize_text,
)


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    title: str
    chunk_text: str
    paragraph_index: int
    subchunk_index: int

    @property
    def text(self) -> str:
        if self.title and self.chunk_text:
            return f"{self.title}\n\n{self.chunk_text}"
        return self.title or self.chunk_text


def _split_oversized_paragraph(tokens: Sequence[str]) -> List[List[str]]:
    if len(tokens) <= CHUNK_TOKEN_TARGET_MAX:
        return [list(tokens)]

    overlap = CHUNK_SUBCHUNK_OVERLAP
    max_tokens = CHUNK_TOKEN_TARGET_MAX
    total_tokens = len(tokens)

    num_chunks = 1
    while total_tokens > num_chunks * max_tokens - (num_chunks - 1) * overlap:
        num_chunks += 1

    expanded_total = total_tokens + (num_chunks - 1) * overlap
    base_size, remainder = divmod(expanded_total, num_chunks)
    chunk_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_chunks)]

    subchunks: List[List[str]] = []
    start = 0
    for size in chunk_sizes:
        end = min(start + size, total_tokens)
        subchunks.append(list(tokens[start:end]))
        start = end - overlap
    return subchunks


def _merge_paragraph_groups(
    paragraphs: Sequence[Tuple[int, str, Sequence[str]]],
) -> List[Tuple[int, str]]:
    groups: List[Tuple[int, str]] = []
    current_start = -1
    current_parts: List[str] = []
    current_tokens = 0

    for paragraph_index, paragraph_text, tokens in paragraphs:
        token_count = len(tokens)
        if current_start == -1:
            current_start = paragraph_index
            current_parts = [paragraph_text]
            current_tokens = token_count
            continue

        should_merge = (
            current_tokens < CHUNK_TOKEN_TARGET_MIN
            and current_tokens + token_count <= CHUNK_TOKEN_TARGET_MAX
        )
        if should_merge:
            current_parts.append(paragraph_text)
            current_tokens += token_count
            continue

        groups.append((current_start, "\n\n".join(current_parts)))
        current_start = paragraph_index
        current_parts = [paragraph_text]
        current_tokens = token_count

    if current_start != -1:
        groups.append((current_start, "\n\n".join(current_parts)))

    return groups


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    page_id = int(record["page_id"])
    title = normalize_whitespace(record.get("title", ""))
    paragraphs = split_paragraphs(record.get("content", ""))

    if not paragraphs:
        return [
            Chunk(
                page_id=page_id,
                chunk_id=0,
                title=title,
                chunk_text="",
                paragraph_index=0,
                subchunk_index=0,
            )
        ]

    chunks: List[Chunk] = []
    paragraph_buffer: List[Tuple[int, str, List[str]]] = []
    next_chunk_id = 0

    for paragraph_index, paragraph_text in enumerate(paragraphs):
        tokens = tokenize_text(paragraph_text)
        if len(tokens) <= CHUNK_TOKEN_TARGET_MAX:
            paragraph_buffer.append((paragraph_index, paragraph_text, tokens))
            continue

        for start_index, merged_text in _merge_paragraph_groups(paragraph_buffer):
            chunks.append(
                Chunk(
                    page_id=page_id,
                    chunk_id=next_chunk_id,
                    title=title,
                    chunk_text=merged_text,
                    paragraph_index=start_index,
                    subchunk_index=0,
                )
            )
            next_chunk_id += 1

        paragraph_buffer = []

        for subchunk_index, subchunk_tokens in enumerate(_split_oversized_paragraph(tokens)):
            chunks.append(
                Chunk(
                    page_id=page_id,
                    chunk_id=next_chunk_id,
                    title=title,
                    chunk_text=untokenize_text(subchunk_tokens),
                    paragraph_index=paragraph_index,
                    subchunk_index=subchunk_index,
                )
            )
            next_chunk_id += 1

    for start_index, merged_text in _merge_paragraph_groups(paragraph_buffer):
        chunks.append(
            Chunk(
                page_id=page_id,
                chunk_id=next_chunk_id,
                title=title,
                chunk_text=merged_text,
                paragraph_index=start_index,
                subchunk_index=0,
            )
        )
        next_chunk_id += 1

    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
