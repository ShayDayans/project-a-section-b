"""
Section B entry point.

The autograder calls run(queries) once with all evaluation queries (batch of 50).
Query embedding + retrieval must complete within the time limit (GPU available).
"""
from __future__ import annotations

from typing import List

from embed import get_model
from index import build_artifacts
from retrieve import DEFAULT_RETRIEVAL_CONFIG, _load_cross_encoder, _load_runtime, search_batch

# Warm the embedding model at import time so local eval scripts don't
# count the first lazy model load inside query-phase timing.
get_model()
_load_runtime()
_load_cross_encoder(DEFAULT_RETRIEVAL_CONFIG)


def run(queries: List[str]) -> List[List[int]]:
    """
    Rank corpus pages for each query.

    Parameters
    ----------
    queries : list[str]
        Batch of query strings (e.g. 50 hidden queries at grading time).

    Returns
    -------
    list[list[int]]
        One ranked list of page_id per query (most relevant first).
        Only the first 10 IDs per list are scored.
    """
    return search_batch(queries)


def build_offline_index() -> None:
    """Run once locally to create artifacts/ (not timed at grading)."""
    build_artifacts()


if __name__ == "__main__":
    build_offline_index()
    print("Index built under artifacts/. Run: python scripts/eval_public.py")
    
