"""Offline dense index build and runtime load helpers."""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

import faiss
import numpy as np

from bm25 import BM25ArtifactBuilder
from chunk import Chunk, chunk_entry
from embed import embed_texts, get_model_device
from utils import (
    ARTIFACTS_DIR,
    BM25_ARRAYS_NAME,
    BM25_META_NAME,
    DENSE_INDEX_NAME,
    DENSE_META_NAME,
    EMBEDDING_MODEL_NAME,
    ensure_artifacts_dir,
    iter_entries,
    list_entry_paths,
    normalize_page_id,
    normalize_tokens,
)

_DENSE_EMBED_BATCH_SIZE = 256
_EMPTY_EMBEDDING_DIM = 384
_PROGRESS_PAGE_INTERVAL = 500
_PROGRESS_TIME_INTERVAL_SECONDS = 5.0


def _chunk_to_metadata(chunk: Chunk) -> Dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "page_id": chunk.page_id,
        "title": chunk.title,
        "chunk_text": chunk.chunk_text,
        "paragraph_index": chunk.paragraph_index,
        "subchunk_index": chunk.subchunk_index,
    }


def _build_dense_faiss_index(vectors: np.ndarray) -> faiss.Index:
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape={vectors.shape!r}")

    dense_vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(dense_vectors.shape[1])
    if dense_vectors.shape[0] > 0:
        index.add(dense_vectors)
    return index


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_eta(processed: int, total: int, elapsed_seconds: float) -> str:
    if processed <= 0 or elapsed_seconds <= 0.0 or processed >= total:
        return "--:--:--"
    rate = processed / elapsed_seconds
    if rate <= 0.0:
        return "--:--:--"
    remaining_seconds = (total - processed) / rate
    return _format_duration(remaining_seconds)


@dataclass(frozen=True)
class _BuildChunkRecord:
    chunk: Chunk
    dense_text: str
    tokens: List[str]
    term_counts: Dict[str, int]
    token_count: int


class _DenseMetadataWriter:
    """Stream dense metadata JSON to disk while preserving the existing schema."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: TextIO | None = None
        self._count = 0
        self._first_chunk = True
        self._count_width = 12
        self._count_offset = 0

    def __enter__(self) -> "_DenseMetadataWriter":
        self._file = self._path.open("w", encoding="utf-8")
        prefix = (
            '{"model":'
            + json.dumps(EMBEDDING_MODEL_NAME, ensure_ascii=False)
            + ',"num_vectors":'
        )
        self._file.write(prefix)
        self._count_offset = self._file.tell()
        self._file.write(" " * self._count_width)
        self._file.write(',"chunks":[')
        return self

    def append(self, chunk: Chunk) -> None:
        if self._file is None:
            raise RuntimeError("Metadata writer is not open")
        if not self._first_chunk:
            self._file.write(",")
        json.dump(
            _chunk_to_metadata(chunk),
            self._file,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._first_chunk = False
        self._count += 1

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._file is None:
            return
        try:
            self._file.write("]}")
            if exc_type is not None:
                return
            if self._count >= 10**self._count_width:
                raise ValueError("Dense metadata count exceeded header field width")
            self._file.flush()
            self._file.seek(self._count_offset)
            self._file.write(f"{self._count:>{self._count_width}d}")
        finally:
            self._file.close()
            self._file = None
        if exc_type is not None:
            return


class _DenseIndexBuilder:
    """Streaming dense builder that keeps only one embedding batch in memory."""

    def __init__(
        self,
        *,
        metadata_path: Path,
        batch_size: int = _DENSE_EMBED_BATCH_SIZE,
    ) -> None:
        self._metadata_path = metadata_path
        self._batch_size = int(batch_size)
        self._index: faiss.Index | None = None
        self._page_ids: List[int] = []
        self._batch_records: List[_BuildChunkRecord] = []
        self._metadata_writer: _DenseMetadataWriter | None = None
        self.embedded_chunks = 0
        self.embedding_seconds = 0.0

    def __enter__(self) -> "_DenseIndexBuilder":
        self._metadata_writer = _DenseMetadataWriter(self._metadata_path).__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            self.flush_batch()
        finally:
            if self._metadata_writer is not None:
                self._metadata_writer.__exit__(exc_type, exc, tb)
                self._metadata_writer = None

    def add_record(self, record: _BuildChunkRecord) -> Tuple[int, float]:
        self._batch_records.append(record)
        if len(self._batch_records) < self._batch_size:
            return 0, 0.0
        return self.flush_batch()

    def flush_batch(self) -> Tuple[int, float]:
        if not self._batch_records:
            return 0, 0.0
        if self._metadata_writer is None:
            raise RuntimeError("Dense metadata writer is not open")

        batch_records = self._batch_records
        self._batch_records = []

        t0 = time.perf_counter()
        vectors = embed_texts(
            [record.dense_text for record in batch_records],
            batch_size=self._batch_size,
        )
        elapsed = time.perf_counter() - t0
        self.embedding_seconds += elapsed

        if self._index is None:
            self._index = _build_dense_faiss_index(vectors)
        elif vectors.shape[0] > 0:
            self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))

        for record in batch_records:
            self._metadata_writer.append(record.chunk)
            self._page_ids.append(record.chunk.page_id)

        flushed = len(batch_records)
        self.embedded_chunks += flushed
        return flushed, elapsed

    def finalize(self, artifacts_dir: Path) -> Tuple[faiss.Index, List[int]]:
        index = self._index
        if index is None:
            index = faiss.IndexFlatIP(_EMPTY_EMBEDDING_DIM)
        faiss.write_index(index, str(artifacts_dir / DENSE_INDEX_NAME))
        return index, list(self._page_ids)


class _BuildProgressReporter:
    """Plain-text progress output for long-running offline builds."""

    def __init__(
        self,
        *,
        total_pages: int,
        artifacts_dir: Path,
        dense_batch_size: int,
        embedding_device: str,
    ) -> None:
        self.total_pages = total_pages
        self.artifacts_dir = artifacts_dir
        self.dense_batch_size = dense_batch_size
        self.embedding_device = embedding_device
        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time
        self.last_logged_pages = 0

    def log_start(self) -> None:
        print(
            "[start] "
            f"pages={self.total_pages} "
            f"model={EMBEDDING_MODEL_NAME} "
            f"device={self.embedding_device} "
            f"dense_batch={self.dense_batch_size} "
            f"artifacts={self.artifacts_dir}",
            flush=True,
        )

    def maybe_log(
        self,
        *,
        pages_processed: int,
        chunks_produced: int,
        dense_chunks_embedded: int,
        bm25_vocab_size: int,
    ) -> None:
        now = time.perf_counter()
        if (
            pages_processed - self.last_logged_pages < _PROGRESS_PAGE_INTERVAL
            and now - self.last_log_time < _PROGRESS_TIME_INTERVAL_SECONDS
            and pages_processed < self.total_pages
        ):
            return

        elapsed = now - self.start_time
        pages_per_minute = (pages_processed / elapsed * 60.0) if elapsed > 0.0 else 0.0
        chunks_per_second = chunks_produced / elapsed if elapsed > 0.0 else 0.0
        percent = (100.0 * pages_processed / self.total_pages) if self.total_pages else 100.0
        eta = _format_eta(pages_processed, self.total_pages, elapsed)

        print(
            "[build] "
            f"pages={pages_processed}/{self.total_pages} "
            f"pct={percent:.1f} "
            f"chunks={chunks_produced} "
            f"dense_embedded={dense_chunks_embedded} "
            f"bm25_vocab={bm25_vocab_size} "
            f"elapsed={_format_duration(elapsed)} "
            f"pages_per_min={pages_per_minute:.1f} "
            f"chunks_per_sec={chunks_per_second:.1f} "
            f"eta={eta}",
            flush=True,
        )
        self.last_log_time = now
        self.last_logged_pages = pages_processed

    def log_dense_flush(self, *, batch_size: int, embedded_total: int, elapsed_seconds: float) -> None:
        rate = batch_size / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        print(
            "[dense] "
            f"flushed={batch_size} "
            f"embedded_total={embedded_total} "
            f"elapsed={elapsed_seconds:.2f}s "
            f"rate={rate:.1f} chunks/s",
            flush=True,
        )

    def log_phase(self, label: str, **fields: object) -> None:
        suffix = " ".join(f"{key}={value}" for key, value in fields.items())
        if suffix:
            print(f"[{label}] {suffix}", flush=True)
        else:
            print(f"[{label}]", flush=True)

    def log_artifact(self, path: Path) -> None:
        size_mb = path.stat().st_size / (1024.0 * 1024.0)
        print(f"[write] file={path.name} size_mb={size_mb:.2f}", flush=True)

    def log_summary(
        self,
        *,
        pages_processed: int,
        chunks_produced: int,
        dense_chunks_embedded: int,
        avg_chunk_length: float,
        bm25_vocab_size: int,
        phase_seconds: Dict[str, float],
    ) -> None:
        total_elapsed = time.perf_counter() - self.start_time
        print(
            "[done] "
            f"pages={pages_processed} "
            f"chunks={chunks_produced} "
            f"avg_chunk_length={avg_chunk_length:.2f} "
            f"bm25_vocab={bm25_vocab_size} "
            f"dense_vectors={dense_chunks_embedded} "
            f"elapsed={_format_duration(total_elapsed)} "
            f"chunking_s={phase_seconds.get('chunking', 0.0):.2f} "
            f"embedding_s={phase_seconds.get('embedding', 0.0):.2f} "
            f"bm25_finalize_s={phase_seconds.get('bm25_finalize', 0.0):.2f} "
            f"artifact_write_s={phase_seconds.get('artifact_write', 0.0):.2f}",
            flush=True,
        )


def _load_entry(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["page_id"] = normalize_page_id(data.get("page_id", path.stem))
    return data


def _make_build_record(chunk: Chunk) -> _BuildChunkRecord:
    tokens = normalize_tokens(chunk.text)
    term_counts = dict(Counter(tokens))
    return _BuildChunkRecord(
        chunk=chunk,
        dense_text=chunk.text,
        tokens=tokens,
        term_counts=term_counts,
        token_count=len(tokens),
    )


def build_artifacts(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    dense_batch_size: int = _DENSE_EMBED_BATCH_SIZE,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> Tuple[faiss.Index, List[int]]:
    """Build dense and BM25 artifacts from one shared corpus/chunk pass."""
    out_dir = artifacts_dir or ensure_artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    entry_paths = list_entry_paths(entries_dir)
    progress = _BuildProgressReporter(
        total_pages=len(entry_paths),
        artifacts_dir=out_dir,
        dense_batch_size=dense_batch_size,
        embedding_device=get_model_device(),
    )
    progress.log_start()

    bm25_builder = BM25ArtifactBuilder(k1=bm25_k1, b=bm25_b)
    phase_seconds = {
        "chunking": 0.0,
        "embedding": 0.0,
        "bm25_finalize": 0.0,
        "artifact_write": 0.0,
    }

    pages_processed = 0
    chunks_produced = 0
    metadata_path = out_dir / DENSE_META_NAME

    with _DenseIndexBuilder(metadata_path=metadata_path, batch_size=dense_batch_size) as dense_builder:
        for entry_path in entry_paths:
            t0 = time.perf_counter()
            record = _load_entry(entry_path)
            chunks = chunk_entry(record)
            chunk_records = [_make_build_record(chunk) for chunk in chunks]
            phase_seconds["chunking"] += time.perf_counter() - t0

            for build_record in chunk_records:
                bm25_builder.add_chunk(
                    build_record.chunk,
                    tokens=build_record.tokens,
                    term_counts=build_record.term_counts,
                    token_count=build_record.token_count,
                )
                flushed, flush_seconds = dense_builder.add_record(build_record)
                if flushed:
                    phase_seconds["embedding"] += flush_seconds
                    progress.log_dense_flush(
                        batch_size=flushed,
                        embedded_total=dense_builder.embedded_chunks,
                        elapsed_seconds=flush_seconds,
                    )
                chunks_produced += 1

            pages_processed += 1
            progress.maybe_log(
                pages_processed=pages_processed,
                chunks_produced=chunks_produced,
                dense_chunks_embedded=dense_builder.embedded_chunks,
                bm25_vocab_size=bm25_builder.vocab_size,
            )

        flushed, flush_seconds = dense_builder.flush_batch()
        if flushed:
            phase_seconds["embedding"] += flush_seconds
            progress.log_dense_flush(
                batch_size=flushed,
                embedded_total=dense_builder.embedded_chunks,
                elapsed_seconds=flush_seconds,
            )

    progress.log_phase("bm25", status="finalize_start")
    t0 = time.perf_counter()
    bm25_bundle = bm25_builder.finalize_bundle()
    phase_seconds["bm25_finalize"] = time.perf_counter() - t0
    progress.log_phase(
        "bm25",
        status="finalize_done",
        vocab=bm25_builder.vocab_size,
        postings=bm25_builder.total_postings,
    )

    progress.log_phase("write", status="start")
    t0 = time.perf_counter()
    index, page_ids = dense_builder.finalize(out_dir)
    progress.log_artifact(out_dir / DENSE_META_NAME)
    progress.log_artifact(out_dir / DENSE_INDEX_NAME)

    bm25_arrays = bm25_bundle["arrays"]
    bm25_metadata = {key: value for key, value in bm25_bundle.items() if key != "arrays"}
    np.savez(
        out_dir / BM25_ARRAYS_NAME,
        vocab_terms=bm25_arrays["vocab_terms"],
        posting_starts=bm25_arrays["posting_starts"],
        posting_doc_freqs=bm25_arrays["posting_doc_freqs"],
        posting_idfs=bm25_arrays["posting_idfs"],
        doc_indices=bm25_arrays["doc_indices"],
        term_frequencies=bm25_arrays["term_frequencies"],
        chunk_lengths=bm25_arrays["chunk_lengths"],
    )
    (out_dir / BM25_META_NAME).write_text(
        json.dumps(bm25_metadata, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    phase_seconds["artifact_write"] = time.perf_counter() - t0
    progress.log_artifact(out_dir / BM25_ARRAYS_NAME)
    progress.log_artifact(out_dir / BM25_META_NAME)
    progress.log_phase("write", status="done")

    avg_chunk_length = 0.0
    chunk_lengths = bm25_bundle["arrays"]["chunk_lengths"]
    if chunk_lengths.size > 0:
        avg_chunk_length = float(np.asarray(chunk_lengths, dtype=np.float32).mean())

    progress.maybe_log(
        pages_processed=pages_processed,
        chunks_produced=chunks_produced,
        dense_chunks_embedded=chunks_produced,
        bm25_vocab_size=bm25_builder.vocab_size,
    )
    progress.log_summary(
        pages_processed=pages_processed,
        chunks_produced=chunks_produced,
        dense_chunks_embedded=chunks_produced,
        avg_chunk_length=avg_chunk_length,
        bm25_vocab_size=bm25_builder.vocab_size,
        phase_seconds=phase_seconds,
    )
    return index, page_ids


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[faiss.Index, List[int]]:
    """
    Embed the full chunked corpus and persist a dense FAISS artifact bundle.

    Returns the built FAISS index together with the per-chunk page_id list.
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / DENSE_META_NAME

    with _DenseIndexBuilder(metadata_path=metadata_path) as dense_builder:
        for record in iter_entries(entries_dir):
            for chunk in chunk_entry(record):
                dense_builder.add_record(_make_build_record(chunk))
        dense_builder.flush_batch()
    return dense_builder.finalize(out_dir)


def load_dense_index(artifacts_dir: Optional[Path] = None) -> faiss.Index:
    """Load the saved exact dense FAISS index from artifacts/."""
    root = artifacts_dir or ARTIFACTS_DIR
    return faiss.read_index(str(root / DENSE_INDEX_NAME))


def load_dense_metadata(
    artifacts_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load dense chunk metadata from artifacts/ without rebuilding anything."""
    root = artifacts_dir or ARTIFACTS_DIR
    payload = json.loads((root / DENSE_META_NAME).read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError("Dense metadata is malformed: 'chunks' must be a list")
    return chunks


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Backward-compatible runtime adapter.

    Loads the FAISS artifact bundle and reconstructs the stored chunk embedding
    matrix together with the chunk -> page_id mapping expected by retrieve.py.
    """
    index = load_dense_index(artifacts_dir)
    metadata = load_dense_metadata(artifacts_dir)

    ntotal = index.ntotal
    if ntotal != len(metadata):
        raise ValueError(
            f"Dense artifact mismatch: index has {ntotal} vectors but metadata has "
            f"{len(metadata)} chunks"
        )

    if ntotal == 0:
        vectors = np.zeros((0, 0), dtype=np.float32)
    else:
        vectors = faiss.vector_to_array(index.reconstruct_n(0, ntotal)).reshape(
            ntotal, index.d
        )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    page_ids = [int(chunk["page_id"]) for chunk in metadata]
    return vectors, page_ids
