"""Plain text and Markdown file parser."""
from __future__ import annotations

from typing import Any


async def parse_text(file_bytes: bytes, filename: str, subllm=None) -> dict[str, Any]:
    """Read plain text or Markdown; summarise hierarchically when large."""
    try:
        text = file_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        return {"error": f"Could not read file: {exc}", "filename": filename}

    ext        = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    char_count = len(text)
    word_count = len(text.split())
    excerpt    = text[:500].strip()
    summary    = excerpt

    if subllm and text.strip() and char_count > 500:
        from ._summarise import hierarchical_summarise
        summary = await hierarchical_summarise(text, subllm)

    return {
        "filename":   filename,
        "file_type":  ext,
        "char_count": char_count,
        "word_count": word_count,
        "excerpt":    excerpt,
        "summary":    summary,
    }
