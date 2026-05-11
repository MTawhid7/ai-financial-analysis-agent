"""File upload and export endpoints.

POST /files/upload               Accept CSV, PDF, XLSX, XLS, DOCX, TXT, MD, JSON.
POST /export/pdf/{report_id}     Generate and return a PDF of the report.
POST /export/docx/{report_id}    Generate and return a Word document.
POST /export/xlsx/{report_id}    Generate and return an Excel workbook.
GET  /export/available           Return which export formats are available.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from ..core.database import get_db_path
from ..core.deps import CurrentUser, get_current_user
from ..core import session_manager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["files"])

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
_ALLOWED_EXTENSIONS = {".csv", ".pdf", ".xlsx", ".xls", ".docx", ".txt", ".md", ".json"}
_WORKSPACE = os.getenv("WORKSPACE_DIR", "workspace")


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------


@router.post("/files/upload")
async def upload_file(
    file: UploadFile,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Accept a financial document and return a structured summary."""
    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {_MAX_UPLOAD_BYTES // 1024 // 1024} MB limit.",
        )

    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()

    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type: '{ext}'. "
                f"Supported: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
            ),
        )

    agent = session_manager.get_or_create(user.id)
    subllm = agent._subllm

    from ai_financial_analyst.tools.file_parser import (
        parse_csv, parse_docx, parse_json, parse_pdf, parse_text, parse_xlsx,
    )

    if ext == ".csv":
        summary = parse_csv(content, filename)
    elif ext == ".pdf":
        summary = await parse_pdf(content, filename, subllm=subllm)
    elif ext in (".xlsx", ".xls"):
        summary = parse_xlsx(content, filename)
    elif ext == ".docx":
        summary = await parse_docx(content, filename, subllm=subllm)
    elif ext in (".txt", ".md"):
        summary = await parse_text(content, filename, subllm=subllm)
    elif ext == ".json":
        summary = parse_json(content, filename)
    else:
        # Should not reach here given the extension check above
        raise HTTPException(status_code=415, detail=f"Unsupported: {ext}")

    return summary


# ---------------------------------------------------------------------------
# Export availability check
# ---------------------------------------------------------------------------


@router.get("/reports/{report_id}/sources")
async def get_report_sources(
    report_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Return citations + web source URLs for a report."""
    async with aiosqlite.connect(get_db_path()) as db:
        async with db.execute(
            "SELECT tickers, analysis_json, raw_data_json FROM reports"
            " WHERE id = ? AND user_id = ?",
            (report_id, user.id),
        ) as cursor:
            row = await cursor.fetchone()

    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    analysis = json.loads(row[1] or "{}")
    raw_data = json.loads(row[2] or "{}")

    # Build citations per ticker
    citations_by_ticker: dict = {}
    for ticker, ta in analysis.items():
        cit = ta.get("citations", {})
        metrics: dict = {}
        for metric, source in cit.items():
            metrics[metric] = {
                "value": ta.get(metric),
                "source_tool": source.get("source_tool"),
                "observation_step": source.get("observation_step"),
            }
        if metrics:
            citations_by_ticker[ticker] = metrics

    # Collect web search results (include title + URL for citation links)
    web_sources: list[dict] = []
    for ticker, ticker_data in raw_data.items():
        news = ticker_data.get("news_search", {})
        if isinstance(news, dict):
            for item in news.get("summaries", []):
                url = item.get("url", "")
                title = item.get("headline", "")
                if url:
                    web_sources.append({
                        "ticker": ticker,
                        "title": title,
                        "url": url,
                        "score": item.get("score", 0),
                    })

    return {
        "tickers": row[0],
        "analysis": citations_by_ticker,
        "web_sources": web_sources,
    }


@router.get("/export/available")
async def export_available() -> dict:
    """Return which export formats are supported on this server."""
    from ai_financial_analyst.tools.pdf_exporter import is_available as pdf_available
    return {
        "pdf": pdf_available(),
        "docx": True,
        "xlsx": True,
    }


# ---------------------------------------------------------------------------
# Helpers: load report from DB
# ---------------------------------------------------------------------------


async def _load_report(report_id: str, user_id: str) -> dict:
    async with aiosqlite.connect(get_db_path()) as db:
        async with db.execute(
            "SELECT tickers, report_markdown, raw_data_json, analysis_json"
            " FROM reports WHERE id = ? AND user_id = ?",
            (report_id, user_id),
        ) as cursor:
            row = await cursor.fetchone()

    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    return {
        "tickers": row[0],
        "report_markdown": row[1],
        "raw_data": json.loads(row[2] or "{}"),
        "analysis": json.loads(row[3] or "{}"),
    }


def _tmp_path(suffix: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix="fin_export_"))
    return d / f"report{suffix}"


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------


@router.post("/export/pdf/{report_id}")
async def export_pdf(
    report_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> FileResponse:
    data = await _load_report(report_id, user.id)
    out = _tmp_path(".pdf")

    try:
        from ai_financial_analyst.tools.pdf_exporter import export_to_pdf
        export_to_pdf(data["report_markdown"], out)
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    tickers = data["tickers"].replace(", ", "_")
    return FileResponse(
        path=str(out),
        media_type="application/pdf",
        filename=f"analysis_{tickers}.pdf",
    )


# ---------------------------------------------------------------------------
# Word export
# ---------------------------------------------------------------------------


@router.post("/export/docx/{report_id}")
async def export_docx(
    report_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> FileResponse:
    data = await _load_report(report_id, user.id)
    out = _tmp_path(".docx")

    try:
        from ai_financial_analyst.tools.docx_exporter import export_to_docx
        export_to_docx(data["report_markdown"], out)
    except ImportError as exc:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc))

    tickers = data["tickers"].replace(", ", "_")
    return FileResponse(
        path=str(out),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"analysis_{tickers}.docx",
    )


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------


@router.post("/export/xlsx/{report_id}")
async def export_xlsx(
    report_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> FileResponse:
    data = await _load_report(report_id, user.id)
    out = _tmp_path(".xlsx")

    try:
        from ai_financial_analyst.tools.xlsx_exporter import export_to_xlsx
        export_to_xlsx(data["raw_data"], data["analysis"], out)
    except ImportError as exc:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc))

    tickers = data["tickers"].replace(", ", "_")
    return FileResponse(
        path=str(out),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"analysis_{tickers}.xlsx",
    )
