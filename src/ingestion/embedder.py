"""src/ingestion/embedder.py
Unified text embedding for all chunk types.
Image chunks use their VLM text summary as input -- never raw pixels.
"""
from __future__ import annotations
import threading
from typing import Optional
import numpy as np
from src.config import get_settings
from src.ingestion.pdf_parser import ParsedChunk

_lock = threading.Lock()
_model = None


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(get_settings().embedding_model)
    return _model


class Embedder:
    def embed_chunks(self, chunks: list[ParsedChunk],
                     batch_size: int = 32,
                     show_progress: bool = False) -> np.ndarray:
        if not chunks:
            return np.empty((0, get_settings().embedding_dim))
        return _get_model().encode(
            [c.text for c in chunks],
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True)

    def embed_query(self, query: str) -> np.ndarray:
        return _get_model().encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True)[0]

    @property
    def dim(self) -> int:
        return get_settings().embedding_dim


_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder