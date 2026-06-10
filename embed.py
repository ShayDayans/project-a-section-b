"""Embedding utilities for the bundled all-MiniLM-L6-v2 model."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import ARTIFACTS_DIR, EMBEDDING_MODEL_NAME

_model: SentenceTransformer | None = None

_LOCAL_MODEL_DIR = ARTIFACTS_DIR / "models" / "all-MiniLM-L6-v2"


def _resolve_model_device() -> str:
    preferred = os.environ.get("SBERT_DEVICE")
    if preferred:
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        model_source = str(_resolve_model_path())
        _model = SentenceTransformer(
            model_source,
            device=_resolve_model_device(),
        )
    return _model


def _resolve_model_path() -> Path:
    if _LOCAL_MODEL_DIR.exists():
        return _LOCAL_MODEL_DIR
    raise FileNotFoundError(
        "Bundled embedding model not found at "
        f"{_LOCAL_MODEL_DIR}. Save {EMBEDDING_MODEL_NAME} there before running."
    )


def get_model_device() -> str:
    model = get_model()
    device = getattr(model, "device", None)
    if device is None:
        device = getattr(model, "_target_device", "unknown")
    return str(device)


def embed_texts(texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
    """Return L2-normalized embeddings, shape (n, dim)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_queries(queries: List[str], *, batch_size: int = 64) -> np.ndarray:
    return embed_texts(queries, batch_size=batch_size)
