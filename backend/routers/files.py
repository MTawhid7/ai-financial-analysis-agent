"""File upload and export endpoints.

POST /files/upload               Accept CSV or PDF, return parsed summary.
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

_MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
_ALLOWED_TYPES = {"text/csv", "application/pdf", "application/octet-stream"}
_WORKSPACE = os.getenv("WORKSPACE_DIR", "workspace")


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------


@router.post("/files/upload")
async def upload_file(
    file: UploadFile,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Accept a CSV or PDF file and return a structured summary."""
    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {_MAX_UPLOAD_BYTES // 1024 // 1024} MB limit.",
        )

    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()

    if ext == ".csv":
        from ai_financial_analyst.tools.file_parser import parse_csv
        summary = parse_csv(content, filename)
    elif ext == ".pdf":
        agent = session_manager.get_or_create(user.id)
        from ai_financial_analyst.tools.file_parser import parse_pdf
        summary = await parse_pdf(content, filename, subllm=agent._subllm)
    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {ext}. Supported: .csv, .pdf",
        )

    return summary


# ---------------------------------------------------------------------------
# Export availability check
# ---------------------------------------------------------------------------


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
