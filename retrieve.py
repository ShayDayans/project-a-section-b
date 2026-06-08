"""Query-time hybrid retrieval and page aggregation."""
from __future__ import annotations

import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from bm25 import BM25, load_bm25
from embed import embed_queries
from index import load_dense_index, load_dense_metadata
from utils import K_EVAL, tokenize_text

DENSE_TOP_K = 100
BM25_TOP_K = 200
RRF_K = 60

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

_CACHE: dict[Path, dict[str, object]] = {}


def _normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def _tokenize_match(text: str) -> List[str]:
    tokens: List[str] = []
    for token in tokenize_text(text):
        cleaned = token.lower().translate(_PUNCT_TABLE)
        if cleaned:
            tokens.append(cleaned)
    return tokens


def _normalize_feature(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _extract_quoted_phrases(query: str) -> List[str]:
    return [
        phrase.strip().lower()
        for phrase in re.findall(r'"([^"]+)"', query)
        if phrase.strip()
    ]


def _has_code_like_token(tokens: Sequence[str]) -> bool:
    for token in tokens:
        if any(ch in token for ch in ("_", "/", "\\", ".", "::", "(", ")", "[", "]")):
            return True
        if "-" in token and any(ch.isalpha() for ch in token) and any(ch.isdigit() for ch in token):
            return True
        if re.search(r"[a-z][A-Z]|[A-Z][a-z].*\d|\d.*[A-Za-z]", token):
            return True
    return False


def _load_runtime(artifacts_dir: Optional[Path] = None) -> dict[str, object]:
    root = (artifacts_dir or Path(__file__).resolve().parent / "artifacts").resolve()
    cached = _CACHE.get(root)
    if cached is not None:
        return cached

    dense_index = load_dense_index(root)
    dense_metadata = load_dense_metadata(root)
    bm25 = load_bm25(root)

    if len(dense_metadata) != len(bm25.chunk_metadata):
        raise ValueError(
            "Dense and BM25 artifacts are misaligned: chunk counts differ at query time"
        )

    chunk_lookup: List[Dict[str, object]] = []
    for idx, dense_chunk in enumerate(dense_metadata):
        bm25_chunk = bm25.chunk_metadata[idx]
        page_id = int(dense_chunk["page_id"])
        bm25_page_id = int(bm25_chunk["page_id"])
        if page_id != bm25_page_id:
            raise ValueError(
                "Dense and BM25 artifacts are misaligned: page_id mismatch in chunk order"
            )

        title = str(dense_chunk.get("title", bm25_chunk.get("title", "")))
        chunk_text = str(dense_chunk.get("chunk_text", ""))
        full_text = _normalize_text(f"{title} {chunk_text}")
        chunk_lookup.append(
            {
                "page_id": page_id,
                "title": title,
                "chunk_text": chunk_text,
                "title_tokens": set(_tokenize_match(title)),
                "chunk_tokens": set(_tokenize_match(chunk_text)),
                "full_text": full_text,
            }
        )

    runtime = {
        "dense_index": dense_index,
        "bm25": bm25,
        "chunks": chunk_lookup,
    }
    _CACHE[root] = runtime
    return runtime


def _query_lexical_profile(query: str, bm25: BM25) -> dict[str, object]:
    raw_tokens = [token.lower() for token in tokenize_text(query) if token.strip()]
    match_tokens = _tokenize_match(query)
    unique_match_tokens = list(dict.fromkeys(match_tokens))
    quoted_phrases = _extract_quoted_phrases(query)
    digit_tokens = [token for token in unique_match_tokens if any(ch.isdigit() for ch in token)]

    idfs = [
        float(bm25.postings[token].idf)
        for token in unique_match_tokens
        if token in bm25.postings
    ]
    if idfs:
        rare_signal = min(float(np.mean(sorted(idfs, reverse=True)[:3])) / 6.0, 1.0)
    else:
        rare_signal = 0.0

    short_query = len(unique_match_tokens) <= 3 and len(raw_tokens) <= 4
    code_like = _has_code_like_token(raw_tokens)
    has_digits = bool(digit_tokens)
    has_quotes = bool(quoted_phrases)

    lexical_boost = 0.0
    lexical_boost += 0.18 * rare_signal
    lexical_boost += 0.05 if has_digits else 0.0
    lexical_boost += 0.07 if has_quotes else 0.0
    lexical_boost += 0.05 if short_query else 0.0
    lexical_boost += 0.07 if code_like else 0.0
    lexical_boost = min(lexical_boost, 0.30)

    return {
        "query_text": _normalize_text(query),
        "query_tokens": set(unique_match_tokens),
        "quoted_phrases": quoted_phrases,
        "digit_tokens": digit_tokens,
        "bm25_weight": 1.20 + lexical_boost,
        "dense_weight": 1.0,
    }


def _dense_search(
    query_vectors: np.ndarray,
    dense_index: object,
) -> Tuple[np.ndarray, np.ndarray]:
    if query_vectors.size == 0:
        return (
            np.zeros((0, DENSE_TOP_K), dtype=np.float32),
            np.zeros((0, DENSE_TOP_K), dtype=np.int64),
        )
    scores, indices = dense_index.search(
        np.ascontiguousarray(query_vectors, dtype=np.float32),
        DENSE_TOP_K,
    )
    return scores, indices


def _rrf_scores(
    dense_indices: np.ndarray,
    dense_scores: np.ndarray,
    bm25_indices: np.ndarray,
    bm25_scores: np.ndarray,
    *,
    dense_weight: float,
    bm25_weight: float,
) -> Tuple[List[int], Dict[int, float], Dict[int, float], Dict[int, float]]:
    candidate_ids = set()
    dense_score_map: Dict[int, float] = {}
    bm25_score_map: Dict[int, float] = {}
    fused_scores: Dict[int, float] = defaultdict(float)

    for rank, chunk_idx in enumerate(dense_indices.tolist(), start=1):
        if chunk_idx < 0:
            continue
        candidate_ids.add(chunk_idx)
        dense_score_map[chunk_idx] = float(dense_scores[rank - 1])
        fused_scores[chunk_idx] += dense_weight / float(RRF_K + rank)

    for rank, chunk_idx in enumerate(bm25_indices.tolist(), start=1):
        if chunk_idx < 0:
            continue
        candidate_ids.add(chunk_idx)
        bm25_score_map[chunk_idx] = float(bm25_scores[rank - 1])
        fused_scores[chunk_idx] += bm25_weight / float(RRF_K + rank)

    ordered_candidates = sorted(candidate_ids)
    return ordered_candidates, fused_scores, dense_score_map, bm25_score_map


def _chunk_feature_score(
    chunk: Dict[str, object],
    profile: dict[str, object],
) -> Tuple[float, float, float, float]:
    query_tokens = profile["query_tokens"]
    title_tokens = chunk["title_tokens"]
    chunk_tokens = chunk["chunk_tokens"]

    if query_tokens:
        title_overlap = len(query_tokens & title_tokens) / len(query_tokens)
        chunk_overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
    else:
        title_overlap = 0.0
        chunk_overlap = 0.0

    full_text = chunk["full_text"]
    quoted_phrases = profile["quoted_phrases"]
    exact_phrase_match = 0.0
    if quoted_phrases:
        if any(phrase in full_text for phrase in quoted_phrases):
            exact_phrase_match = 1.0
    else:
        query_text = profile["query_text"]
        if len(query_text.split()) >= 2 and query_text in full_text:
            exact_phrase_match = 1.0

    digit_tokens = profile["digit_tokens"]
    if digit_tokens:
        numeric_match = sum(token in full_text for token in digit_tokens) / len(digit_tokens)
    else:
        numeric_match = 0.0

    return title_overlap, chunk_overlap, exact_phrase_match, numeric_match


def _rerank_candidates(
    candidate_indices: List[int],
    fused_scores: Dict[int, float],
    dense_score_map: Dict[int, float],
    bm25_score_map: Dict[int, float],
    chunks: Sequence[Dict[str, object]],
    profile: dict[str, object],
) -> List[Tuple[int, float]]:
    if not candidate_indices:
        return []

    dense_norm = _normalize_feature([dense_score_map.get(idx, 0.0) for idx in candidate_indices])
    bm25_norm = _normalize_feature([bm25_score_map.get(idx, 0.0) for idx in candidate_indices])
    fused_norm = _normalize_feature([fused_scores.get(idx, 0.0) for idx in candidate_indices])

    reranked: List[Tuple[int, float]] = []
    for pos, chunk_idx in enumerate(candidate_indices):
        title_overlap, chunk_overlap, phrase_match, numeric_match = _chunk_feature_score(
            chunks[chunk_idx],
            profile,
        )
        linear_score = (
            0.24 * float(dense_norm[pos])
            + 0.24 * float(bm25_norm[pos])
            + 0.14 * title_overlap
            + 0.18 * chunk_overlap
            + 0.12 * phrase_match
            + 0.08 * numeric_match
        )
        final_score = 0.55 * float(fused_norm[pos]) + 0.45 * linear_score
        reranked.append((chunk_idx, final_score))

    reranked.sort(key=lambda item: (-item[1], item[0]))
    return reranked


def _aggregate_pages(
    reranked_chunks: Sequence[Tuple[int, float]],
    chunks: Sequence[Dict[str, object]],
    *,
    top_k: int,
) -> List[int]:
    page_to_scores: Dict[int, List[float]] = defaultdict(list)
    for chunk_idx, score in reranked_chunks:
        page_id = int(chunks[chunk_idx]["page_id"])
        page_to_scores[page_id].append(score)

    page_scores: List[Tuple[int, float]] = []
    for page_id, scores in page_to_scores.items():
        scores.sort(reverse=True)
        aggregate = scores[0]
        if len(scores) > 1:
            aggregate += 0.40 * scores[1]
        if len(scores) > 2:
            aggregate += 0.20 * scores[2]
        page_scores.append((page_id, aggregate))

    page_scores.sort(key=lambda item: (-item[1], item[0]))
    return [page_id for page_id, _ in page_scores[:top_k]]


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """Hybrid dense + BM25 chunk retrieval with page-level score aggregation."""
    runtime = _load_runtime(artifacts_dir)
    dense_index = runtime["dense_index"]
    bm25: BM25 = runtime["bm25"]  # type: ignore[assignment]
    chunks = runtime["chunks"]

    if not queries:
        return []

    query_vectors = embed_queries(queries)
    dense_scores_batch, dense_indices_batch = _dense_search(query_vectors, dense_index)

    ranked_pages: List[List[int]] = []
    for query_idx, query in enumerate(queries):
        profile = _query_lexical_profile(query, bm25)

        bm25_indices, bm25_scores = bm25.search(query, top_k=BM25_TOP_K)
        dense_indices = dense_indices_batch[query_idx]
        dense_scores = dense_scores_batch[query_idx]

        candidate_indices, fused_scores, dense_score_map, bm25_score_map = _rrf_scores(
            dense_indices=dense_indices,
            dense_scores=dense_scores,
            bm25_indices=bm25_indices,
            bm25_scores=bm25_scores,
            dense_weight=float(profile["dense_weight"]),
            bm25_weight=float(profile["bm25_weight"]),
        )

        reranked_chunks = _rerank_candidates(
            candidate_indices,
            fused_scores,
            dense_score_map,
            bm25_score_map,
            chunks,
            profile,
        )
        ranked_pages.append(_aggregate_pages(reranked_chunks, chunks, top_k=top_k))

    return ranked_pages
