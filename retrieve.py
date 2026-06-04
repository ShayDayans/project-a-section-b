"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from embed import embed_queries
from index import load_index
from utils import K_EVAL

TOP_K_CHUNKS = 3


def _normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """Dense retrieval with mean-of-top-k chunk aggregation per page."""
    corpus_vectors, chunk_page_ids = load_index(artifacts_dir)
    query_vectors = embed_queries(queries)

    page_to_chunk_indices: dict[int, list[int]] = {}
    for i, pid in enumerate(chunk_page_ids):
        page_to_chunk_indices.setdefault(pid, []).append(i)

    all_pages = list(page_to_chunk_indices.keys())
    scores_matrix = query_vectors @ corpus_vectors.T  # (Q, num_chunks)

    ranked: List[List[int]] = []
    for chunk_row in scores_matrix:
        page_scores = np.array([
            float(np.mean(np.sort(chunk_row[page_to_chunk_indices[p]])[-TOP_K_CHUNKS:]))
            for p in all_pages
        ])
        order = np.argsort(-page_scores)
        ranked.append([all_pages[i] for i in order[:top_k]])

    return ranked
