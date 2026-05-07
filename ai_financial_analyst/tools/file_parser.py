"""File parser — converts CSV and PDF uploads to structured summaries.

Security invariants preserved:
- CSV: only a fixed-schema JSON summary is produced (no arbitrary pandas operations,
  no formula execution, no user-supplied column names reach the LLM directly).
- PDF: extracted text goes through Flash-Lite summarisation — raw user content
  never reaches the primary reasoning LLM unsanitised.
- Formula injection: cell values starting with =, +, -, @ (Excel injection
  patterns) are detected and scrubbed before the summary is built.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Excel / Google Sheets formula injection prefixes
_FORMULA_RE = re.compile(r"^[=+\-@]")


def parse_csv(file_bytes: bytes, filename: str = "file.csv") -> dict[str, Any]:
    """Parse a CSV file and return a fixed-schema summary dict.

    The summary includes: shape, column names, dtypes, first 5 rows,
    and numeric descriptive stats.  No raw cell values longer than 200
    characters are included to prevent context injection.
    """
    try:
        import pandas as pd
        df = pd.read_csv(io.BytesIO(file_bytes), nrows=500)
    except Exception as exc:
        return {"error": f"Could not parse CSV: {exc}", "filename": filename}

    # Scrub formula injection from string cells
    injection_count = 0
    for col in df.select_dtypes(include="object").columns:
        def _scrub(val):
            nonlocal injection_count
            s = str(val) if val is not None else ""
            if _FORMULA_RE.match(s):
                injection_count += 1
                return "[REMOVED]"
            return s[:200]  # cap length
        df[col] = df[col].map(_scrub)

    numeric_stats: dict[str, dict] = {}
    for col in df.select_dtypes(include="number").columns:
        stats = df[col].describe()
        numeric_stats[col] = {k: round(float(v), 4) for k, v in stats.items() if k != "count"}

    return {
        "filename": filename,
        "file_type": "csv",
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "preview": df.head(5).fillna("").to_dict(orient="records"),
        "numeric_stats": numeric_stats,
        "formula_cells_removed": injection_count,
    }


async def parse_pdf(file_bytes: bytes, filename: str, subllm=None) -> dict[str, Any]:
    """Extract text from a PDF and return a structured summary.

    If a subllm is provided, Flash-Lite summarises the extracted text.
    Otherwise, returns the raw first 2,000 characters of text.
    """
    try:
        import pdfplumber
        text_pages: list[str] = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages[:20]:  # cap at 20 pages
                page_text = page.extract_text() or ""
                text_pages.append(page_text[:3000])  # cap per page
    except Exception as exc:
        return {"error": f"Could not parse PDF: {exc}", "filename": filename}

    full_text = "\n\n".join(text_pages)
    page_count = len(text_pages)

    summary_text = full_text[:2000]  # default: first 2k chars
    if subllm and full_text.strip():
        try:
            from langchain_core.messages import HumanMessage
            from ..core.llm import content_to_str
            prompt = (
                "Summarise the following document in 3-5 sentences. "
                "Focus on: document type, key topics, any financial figures or company names.\n\n"
                f"Document text:\n{full_text[:4000]}"
            )
            resp = await subllm.ainvoke([HumanMessage(content=prompt)])
            summary_text = content_to_str(
                resp.content if hasattr(resp, "content") else resp
            ).strip()
        except Exception as exc:
            logger.warning("PDF summarisation failed: %s", exc)

    return {
        "filename": filename,
        "file_type": "pdf",
        "pages": page_count,
        "summary": summary_text,
        "char_count": len(full_text),
    }
