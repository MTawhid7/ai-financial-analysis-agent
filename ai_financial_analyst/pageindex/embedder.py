"""Gemini text-embedding-004 service for PageIndex.

Uses langchain-google-genai (already installed).  Caches results via the
existing ResultCache to avoid re-embedding identical text across runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)

_EMBED_MODEL = settings.llm_embedding_model
_EMBED_DIMS  = 768


def _make_embedder(task_type: str) -> Any:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(
        model=_EMBED_MODEL,
        google_api_key=settings.google_api_key,
        task_type=task_type,
    )


_doc_embedder:   Any = None
_query_embedder: Any = None


def _get_doc_embedder() -> Any:
    global _doc_embedder
    if _doc_embedder is None:
        _doc_embedder = _make_embedder("retrieval_document")
    return _doc_embedder


def _get_query_embedder() -> Any:
    global _query_embedder
    if _query_embedder is None:
        _query_embedder = _make_embedder("retrieval_query")
    return _query_embedder


def _cache_key(text: str) -> str:
    return "embed:" + hashlib.sha256(text.encode()).hexdigest()


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of document texts in batches. Results are cached.

    Returns a list of 768-dim float vectors in the same order as `texts`.
    """
    from ..core.cache import ResultCache
    cache = ResultCache()

    results: list[list[float] | None] = [None] * len(texts)
    uncached_indices: list[int] = []
    uncached_texts:   list[str] = []

    # Check cache first
    for i, text in enumerate(texts):
        cached = cache.get("embed", {"text_hash": _cache_key(text)})
        if cached is not None:
            results[i] = json.loads(cached) if isinstance(cached, str) else cached
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    # Embed uncached texts in batches
    if uncached_texts:
        embedder = _get_doc_embedder()
        batch_size = settings.pageindex_embed_batch_size
        for batch_start in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[batch_start: batch_start + batch_size]
            try:
                batch_vectors = await embedder.aembed_documents(batch)
            except Exception as exc:
                logger.warning("Embedding batch failed, using zeros: %s", exc)
                batch_vectors = [[0.0] * _EMBED_DIMS] * len(batch)

            for j, (orig_idx, vector) in enumerate(
                zip(uncached_indices[batch_start: batch_start + batch_size], batch_vectors)
            ):
                results[orig_idx] = vector
                # Cache for future runs
                key_args = {"text_hash": _cache_key(uncached_texts[batch_start + j])}
                cache.set("embed", key_args, json.dumps(vector))

    return [r or ([0.0] * _EMBED_DIMS) for r in results]


async def embed_query(text: str) -> list[float]:
    """Embed a single query string (uses retrieval_query task type)."""
    embedder = _get_query_embedder()
    try:
        return await embedder.aembed_query(text)
    except Exception as exc:
        logger.warning("Query embedding failed: %s", exc)
        return [0.0] * _EMBED_DIMS
