"""PDF file parser using pdfplumber + hierarchical summarisation."""
from __future__ import annotations

import io
from typing import Any


async def parse_pdf(file_bytes: bytes, filename: str, subllm=None) -> dict[str, Any]:
    """Extract text from every page and summarise the full document."""
    try:
        import pdfplumber
        pages: list[str] = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text.strip())
    except Exception as exc:
        return {"error": f"Could not parse PDF: {exc}", "filename": filename}

    full_text  = "\n\n".join(pages)
    char_count = len(full_text)
    summary    = full_text[:2000]

    if subllm and full_text.strip():
        from ._summarise import hierarchical_summarise
        summary = await hierarchical_summarise(full_text, subllm)

    return {
        "filename":   filename,
        "file_type":  "pdf",
        "pages":      page_count,
        "char_count": char_count,
        "summary":    summary,
    }
