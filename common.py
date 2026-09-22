"""Shared local embedding model for today's kata.

One model (all-MiniLM-L6-v2, 384-dim) reused across parts A/B/C so every
embedding stays comparable. Model is cached to disk by sentence-transformers.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Return (n, 384) float32 array of normalized-style embeddings."""
    return model().encode(texts)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Plain-NumPy cosine similarity: dot(a, b) / (||a|| * ||b||)."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))