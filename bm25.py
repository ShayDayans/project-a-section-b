"""BM25Okapi implemented with numpy + standard library only."""
from __future__ import annotations

import math
from collections import Counter
from typing import List

import numpy as np


class BM25:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        tokenized = [doc.lower().split() for doc in corpus]
        self.k1 = k1
        self.b = b
        self.n = len(tokenized)
        self.avgdl = sum(len(d) for d in tokenized) / max(self.n, 1)

        # term -> list of (doc_idx, tf)
        self.doc_freqs: dict[str, int] = {}
        self.tf: List[dict[str, int]] = []
        self.doc_lengths = np.array([len(d) for d in tokenized], dtype=np.float32)

        for doc in tokenized:
            counts = Counter(doc)
            self.tf.append(counts)
            for term in counts:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def get_scores(self, query: str) -> np.ndarray:
        tokens = query.lower().split()
        scores = np.zeros(self.n, dtype=np.float32)
        for term in set(tokens):
            idf = self._idf(term)
            for i, doc in enumerate(self.tf):
                tf = doc.get(term, 0)
                if tf == 0:
                    continue
                dl = self.doc_lengths[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        return scores
