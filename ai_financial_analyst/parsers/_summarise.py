"""Hierarchical Flash-Lite summarisation — covers entire documents regardless of size."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_CHUNK_CHARS   = 3_000
_CHUNK_OVERLAP = 200


def _sliding_chunks(text: str, chunk_size: int = _CHUNK_CHARS, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + chunk_size])
        start += chunk_size - overlap
    return chunks


async def _summarise_chunk(text: str, subllm, task: str = "") -> str:
    from langchain_core.messages import HumanMessage
    from ..core.llm import content_to_str
    prompt = (
        f"{task}\n\n" if task else
        "Summarise the following text in 3-5 sentences. Focus on key topics, "
        "important figures, company names, and financial data.\n\n"
    ) + f"Text:\n{text}"
    try:
        resp = await subllm.ainvoke([HumanMessage(content=prompt)])
        return content_to_str(resp.content if hasattr(resp, "content") else resp).strip()
    except Exception as exc:
        logger.warning("Chunk summarisation failed: %s", exc)
        return text[:500]


async def hierarchical_summarise(text: str, subllm) -> str:
    """Summarise any-length document without truncation.

    Short text → direct summarisation.
    Long text  → sliding-chunk summaries → combined final summary.
    """
    if not text.strip():
        return ""
    if len(text) <= _CHUNK_CHARS:
        return await _summarise_chunk(text, subllm)
    summaries = [await _summarise_chunk(c, subllm) for c in _sliding_chunks(text)]
    return await _summarise_chunk(
        "\n\n".join(summaries), subllm,
        task=(
            "The following are section-by-section summaries of a longer document. "
            "Combine them into a single coherent 4-6 sentence overview, preserving "
            "the most important financial figures and company names:"
        ),
    )
