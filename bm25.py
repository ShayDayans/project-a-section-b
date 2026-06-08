"""Offline-built chunk-level BM25 artifacts and runtime scoring helpers."""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from chunk import Chunk, chunk_entry
from utils import (
    ARTIFACTS_DIR,
    BM25_ARRAYS_NAME,
    BM25_BUNDLE_NAME,
    BM25_META_NAME,
    ensure_artifacts_dir,
    iter_entries,
    normalize_tokens,
)


def _chunk_reference(chunk: Chunk) -> Dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "page_id": chunk.page_id,
        "title": chunk.title,
        "paragraph_index": chunk.paragraph_index,
        "subchunk_index": chunk.subchunk_index,
    }


def _idf(num_docs: int, doc_freq: int) -> float:
    return math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)


@dataclass(frozen=True)
class PostingList:
    doc_indices: np.ndarray
    term_frequencies: np.ndarray
    doc_freq: int
    idf: float


class PostingStore(Mapping[str, PostingList]):
    """Lazy term -> posting adapter over compact contiguous BM25 arrays."""

    def __init__(
        self,
        *,
        vocab_terms: np.ndarray,
        posting_starts: np.ndarray,
        posting_doc_freqs: np.ndarray,
        posting_idfs: np.ndarray,
        doc_indices: np.ndarray,
        term_frequencies: np.ndarray,
    ):
        self._vocab_terms = np.asarray(vocab_terms)
        self._posting_starts = np.asarray(posting_starts, dtype=np.int64)
        self._posting_doc_freqs = np.asarray(posting_doc_freqs, dtype=np.int32)
        self._posting_idfs = np.asarray(posting_idfs, dtype=np.float32)
        self._doc_indices = np.asarray(doc_indices, dtype=np.int32)
        self._term_frequencies = np.asarray(term_frequencies, dtype=np.float32)
        self._term_to_index = {
            str(term): idx for idx, term in enumerate(self._vocab_terms.tolist())
        }

    def __len__(self) -> int:
        return int(self._vocab_terms.shape[0])

    def __iter__(self):
        for term in self._vocab_terms.tolist():
            yield str(term)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._term_to_index

    def _posting_slice(self, term_index: int) -> slice:
        start = int(self._posting_starts[term_index])
        end = int(self._posting_starts[term_index + 1])
        return slice(start, end)

    def _make_posting(self, term_index: int) -> PostingList:
        posting_slice = self._posting_slice(term_index)
        return PostingList(
            doc_indices=self._doc_indices[posting_slice],
            term_frequencies=self._term_frequencies[posting_slice],
            doc_freq=int(self._posting_doc_freqs[term_index]),
            idf=float(self._posting_idfs[term_index]),
        )

    def __getitem__(self, key: str) -> PostingList:
        term_index = self._term_to_index[key]
        return self._make_posting(term_index)

    def get(self, key: str, default: Optional[PostingList] = None) -> Optional[PostingList]:
        term_index = self._term_to_index.get(key)
        if term_index is None:
            return default
        return self._make_posting(term_index)


class BM25:
    """Runtime BM25 scorer backed by a prebuilt chunk-level artifact bundle."""

    def __init__(
        self,
        *,
        postings: PostingStore,
        chunk_lengths: np.ndarray,
        chunk_metadata: List[Dict[str, Any]],
        avgdl: float,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = float(k1)
        self.b = float(b)
        self.postings = postings
        self.chunk_lengths = np.ascontiguousarray(chunk_lengths, dtype=np.float32)
        self.chunk_metadata = chunk_metadata
        self.n = len(chunk_metadata)
        self.avgdl = float(avgdl)

        if self.chunk_lengths.shape != (self.n,):
            raise ValueError(
                "BM25 artifact is malformed: chunk_lengths shape does not match metadata"
            )

    @classmethod
    def from_serialized_bundle(cls, payload: Dict[str, Any]) -> "BM25":
        version = int(payload.get("version", 1))
        if version >= 2:
            return cls._from_compact_bundle(payload)
        return cls._from_legacy_bundle(payload)

    @classmethod
    def _from_compact_bundle(cls, payload: Dict[str, Any]) -> "BM25":
        arrays = payload.get("arrays")
        if arrays is None:
            raise ValueError("BM25 artifact is malformed: missing compact arrays bundle")

        postings = PostingStore(
            vocab_terms=np.asarray(arrays["vocab_terms"]),
            posting_starts=np.asarray(arrays["posting_starts"], dtype=np.int64),
            posting_doc_freqs=np.asarray(arrays["posting_doc_freqs"], dtype=np.int32),
            posting_idfs=np.asarray(arrays["posting_idfs"], dtype=np.float32),
            doc_indices=np.asarray(arrays["doc_indices"], dtype=np.int32),
            term_frequencies=np.asarray(arrays["term_frequencies"], dtype=np.float32),
        )

        return cls(
            postings=postings,
            chunk_lengths=np.asarray(arrays["chunk_lengths"], dtype=np.float32),
            chunk_metadata=list(payload["chunks"]),
            avgdl=float(payload["avgdl"]),
            k1=float(payload.get("k1", 1.5)),
            b=float(payload.get("b", 0.75)),
        )

    @classmethod
    def _from_legacy_bundle(cls, payload: Dict[str, Any]) -> "BM25":
        postings_payload = payload.get("postings", {})
        if not isinstance(postings_payload, dict):
            raise ValueError("BM25 artifact is malformed: 'postings' must be an object")

        vocab_terms = sorted(postings_payload.keys())
        posting_starts = np.zeros(len(vocab_terms) + 1, dtype=np.int64)
        posting_doc_freqs = np.zeros(len(vocab_terms), dtype=np.int32)
        posting_idfs = np.zeros(len(vocab_terms), dtype=np.float32)

        all_doc_indices: List[np.ndarray] = []
        all_term_frequencies: List[np.ndarray] = []
        offset = 0
        for term_index, term in enumerate(vocab_terms):
            posting = postings_payload[term]
            doc_indices = np.asarray(posting["doc_indices"], dtype=np.int32)
            term_frequencies = np.asarray(posting["term_frequencies"], dtype=np.float32)
            all_doc_indices.append(doc_indices)
            all_term_frequencies.append(term_frequencies)
            posting_doc_freqs[term_index] = int(posting["doc_freq"])
            posting_idfs[term_index] = float(posting["idf"])
            offset += int(doc_indices.shape[0])
            posting_starts[term_index + 1] = offset

        if all_doc_indices:
            flat_doc_indices = np.concatenate(all_doc_indices)
            flat_term_frequencies = np.concatenate(all_term_frequencies)
        else:
            flat_doc_indices = np.zeros(0, dtype=np.int32)
            flat_term_frequencies = np.zeros(0, dtype=np.float32)

        postings = PostingStore(
            vocab_terms=np.asarray(vocab_terms),
            posting_starts=posting_starts,
            posting_doc_freqs=posting_doc_freqs,
            posting_idfs=posting_idfs,
            doc_indices=flat_doc_indices,
            term_frequencies=flat_term_frequencies,
        )

        return cls(
            postings=postings,
            chunk_lengths=np.asarray(payload["chunk_lengths"], dtype=np.float32),
            chunk_metadata=list(payload["chunks"]),
            avgdl=float(payload["avgdl"]),
            k1=float(payload.get("k1", 1.5)),
            b=float(payload.get("b", 0.75)),
        )

    def get_scores(self, query: str) -> np.ndarray:
        query_terms = set(normalize_tokens(query))
        scores = np.zeros(self.n, dtype=np.float32)
        if not query_terms or self.n == 0:
            return scores

        avgdl = self.avgdl if self.avgdl > 0.0 else 1.0
        norm = self.k1 * (1.0 - self.b + self.b * self.chunk_lengths / avgdl)

        for term in query_terms:
            posting = self.postings.get(term)
            if posting is None or posting.doc_indices.size == 0:
                continue

            doc_indices = posting.doc_indices
            tfs = posting.term_frequencies
            denom = tfs + norm[doc_indices]
            scores[doc_indices] += posting.idf * ((tfs * (self.k1 + 1.0)) / denom)

        return scores

    def search(self, query: str, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if top_k <= 0 or self.n == 0:
            return (
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
            )

        scores = self.get_scores(query)
        limit = min(top_k, scores.shape[0])
        if limit == 0:
            return (
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
            )

        if limit == scores.shape[0]:
            ranked_indices = np.argsort(-scores)
        else:
            candidate_indices = np.argpartition(-scores, limit - 1)[:limit]
            ranked_indices = candidate_indices[np.argsort(-scores[candidate_indices])]

        return ranked_indices.astype(np.int32, copy=False), scores[ranked_indices]

    def search_batch(self, queries: List[str], top_k: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [self.search(query, top_k=top_k) for query in queries]


class BM25ArtifactBuilder:
    """Incremental offline BM25 builder aligned with dense chunk order."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self._chunk_metadata: List[Dict[str, Any]] = []
        self._chunk_lengths: List[int] = []
        self._term_to_index: Dict[str, int] = {}
        self._posting_doc_indices: List[List[int]] = []
        self._posting_term_frequencies: List[List[float]] = []

    @property
    def num_chunks(self) -> int:
        return len(self._chunk_metadata)

    @property
    def vocab_size(self) -> int:
        return len(self._term_to_index)

    @property
    def total_postings(self) -> int:
        return sum(len(doc_indices) for doc_indices in self._posting_doc_indices)

    def add_chunk(
        self,
        chunk: Chunk,
        *,
        tokens: List[str],
        term_counts: Dict[str, int] | Counter[str],
        token_count: int,
    ) -> None:
        chunk_index = len(self._chunk_metadata)
        self._chunk_metadata.append(_chunk_reference(chunk))
        self._chunk_lengths.append(int(token_count))

        for term, tf in term_counts.items():
            term_index = self._term_to_index.get(term)
            if term_index is None:
                term_index = len(self._posting_doc_indices)
                self._term_to_index[term] = term_index
                self._posting_doc_indices.append([])
                self._posting_term_frequencies.append([])
            self._posting_doc_indices[term_index].append(chunk_index)
            self._posting_term_frequencies[term_index].append(float(tf))

    def finalize_arrays(self) -> Dict[str, np.ndarray]:
        sorted_terms = sorted(self._term_to_index)
        if sorted_terms:
            vocab_terms = np.asarray(sorted_terms)
        else:
            vocab_terms = np.asarray([], dtype="<U1")

        posting_starts = np.zeros(len(sorted_terms) + 1, dtype=np.int64)
        posting_doc_freqs = np.zeros(len(sorted_terms), dtype=np.int32)
        posting_idfs = np.zeros(len(sorted_terms), dtype=np.float32)

        flat_doc_indices: List[np.ndarray] = []
        flat_term_frequencies: List[np.ndarray] = []
        offset = 0
        num_docs = len(self._chunk_metadata)

        for sorted_term_index, term in enumerate(sorted_terms):
            source_index = self._term_to_index[term]
            doc_indices = np.asarray(self._posting_doc_indices[source_index], dtype=np.int32)
            term_frequencies = np.asarray(
                self._posting_term_frequencies[source_index],
                dtype=np.float32,
            )
            doc_freq = int(doc_indices.shape[0])

            posting_starts[sorted_term_index] = offset
            posting_doc_freqs[sorted_term_index] = doc_freq
            posting_idfs[sorted_term_index] = _idf(num_docs, doc_freq)
            offset += doc_freq

            flat_doc_indices.append(doc_indices)
            flat_term_frequencies.append(term_frequencies)

        posting_starts[-1] = offset

        if flat_doc_indices:
            doc_indices = np.concatenate(flat_doc_indices)
            term_frequencies = np.concatenate(flat_term_frequencies)
        else:
            doc_indices = np.zeros(0, dtype=np.int32)
            term_frequencies = np.zeros(0, dtype=np.float32)

        return {
            "vocab_terms": vocab_terms,
            "posting_starts": posting_starts,
            "posting_doc_freqs": posting_doc_freqs,
            "posting_idfs": posting_idfs,
            "doc_indices": doc_indices,
            "term_frequencies": term_frequencies,
            "chunk_lengths": np.asarray(self._chunk_lengths, dtype=np.int32),
        }

    def finalize_metadata(self, *, chunk_lengths: np.ndarray) -> Dict[str, Any]:
        avgdl = float(chunk_lengths.mean()) if chunk_lengths.size > 0 else 0.0
        return {
            "version": 2,
            "k1": self.k1,
            "b": self.b,
            "num_chunks": len(self._chunk_metadata),
            "avgdl": avgdl,
            "chunks": list(self._chunk_metadata),
        }

    def finalize_bundle(self) -> Dict[str, Any]:
        arrays = self.finalize_arrays()
        metadata = self.finalize_metadata(chunk_lengths=arrays["chunk_lengths"])
        return {**metadata, "arrays": arrays}

    def write_artifacts(self, artifacts_dir: Path) -> Dict[str, Any]:
        bundle = self.finalize_bundle()
        arrays = bundle["arrays"]
        metadata = {key: value for key, value in bundle.items() if key != "arrays"}
        _write_compact_bundle(arrays, metadata, artifacts_dir=artifacts_dir)
        return bundle


def _write_compact_bundle(
    arrays: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    *,
    artifacts_dir: Path,
) -> None:
    np.savez(
        artifacts_dir / BM25_ARRAYS_NAME,
        vocab_terms=arrays["vocab_terms"],
        posting_starts=arrays["posting_starts"],
        posting_doc_freqs=arrays["posting_doc_freqs"],
        posting_idfs=arrays["posting_idfs"],
        doc_indices=arrays["doc_indices"],
        term_frequencies=arrays["term_frequencies"],
        chunk_lengths=arrays["chunk_lengths"],
    )
    (artifacts_dir / BM25_META_NAME).write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _load_compact_metadata(root: Path) -> Optional[Dict[str, Any]]:
    meta_path = root / BM25_META_NAME
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def build_bm25_artifacts(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25:
    """
    Build and persist chunk-level BM25 artifacts from the canonical chunk pipeline.

    The saved chunk order matches dense chunk order for the same corpus traversal.
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    builder = BM25ArtifactBuilder(k1=k1, b=b)
    for record in iter_entries(entries_dir):
        for chunk in chunk_entry(record):
            tokens = normalize_tokens(chunk.text)
            builder.add_chunk(
                chunk,
                tokens=tokens,
                term_counts=Counter(tokens),
                token_count=len(tokens),
            )

    bundle = builder.write_artifacts(out_dir)
    return BM25.from_serialized_bundle(bundle)


def load_bm25_bundle(artifacts_dir: Optional[Path] = None) -> Dict[str, Any]:
    root = artifacts_dir or ARTIFACTS_DIR
    arrays_path = root / BM25_ARRAYS_NAME
    meta_path = root / BM25_META_NAME
    metadata = _load_compact_metadata(root)
    if arrays_path.is_file() and metadata is not None:
        with np.load(arrays_path, allow_pickle=False) as arrays_payload:
            arrays = {
                key: np.ascontiguousarray(arrays_payload[key])
                for key in arrays_payload.files
            }
        metadata["arrays"] = arrays
        return metadata

    bundle_path = root / BM25_BUNDLE_NAME
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"BM25 artifact not found: {arrays_path} / {meta_path} or legacy {bundle_path}. "
            "Build it offline under artifacts/."
        )
    return json.loads(bundle_path.read_text(encoding="utf-8"))


def load_bm25(artifacts_dir: Optional[Path] = None) -> BM25:
    """Load chunk-level BM25 runtime state from the offline artifact bundle."""
    return BM25.from_serialized_bundle(load_bm25_bundle(artifacts_dir))


def load_bm25_metadata(artifacts_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load persisted chunk references aligned with BM25 score indices."""
    root = artifacts_dir or ARTIFACTS_DIR
    metadata = _load_compact_metadata(root)
    if metadata is not None:
        return list(metadata.get("chunks", []))
    return list(load_bm25_bundle(root).get("chunks", []))


def search_bm25(
    query: str,
    *,
    top_k: int = 10,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience API for one-shot BM25 chunk retrieval from saved artifacts."""
    return load_bm25(artifacts_dir).search(query, top_k=top_k)
