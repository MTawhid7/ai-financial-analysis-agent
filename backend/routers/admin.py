"""Admin-only endpoints for system document management.

System documents (scope='system') are visible to all authenticated users.
Only users listed in ADMIN_USER_IDS (comma-separated env var) can call these.

POST /admin/documents/upload   — Index a system document
GET  /admin/documents          — List all system documents
DELETE /admin/documents/{id}   — Remove a system document
PATCH /admin/documents/{id}    — Update metadata (title, tags, source_url)
GET  /admin/documents/users    — List all user-uploaded documents (admin oversight)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel

from ai_financial_analyst.config import settings
from ..core.deps import CurrentUser, get_current_user
from ..core import session_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

_MAX_UPLOAD_BYTES   = 50 * 1024 * 1024
_ALLOWED_EXTENSIONS = {".csv", ".pdf", ".xlsx", ".xls", ".docx", ".txt", ".md", ".json", ".html", ".htm"}
_ADMIN_USER_IDS     = set(settings.admin_user_ids)


# ---------------------------------------------------------------------------
# Admin role guard
# ---------------------------------------------------------------------------

async def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Dependency: allows only users listed in ADMIN_USER_IDS env var."""
    if not _ADMIN_USER_IDS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No admin users configured. Set ADMIN_USER_IDS in .env.",
        )
    if user.id not in _ADMIN_USER_IDS and user.email not in _ADMIN_USER_IDS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return user


# ---------------------------------------------------------------------------
# Upload a system document
# ---------------------------------------------------------------------------

@router.post("/documents/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_system_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    source_url: str | None = None,
    admin: CurrentUser = Depends(require_admin),
) -> dict:
    """Index a document as a system-level resource (visible to all users).

    The file is indexed in the background; returns 202 Accepted immediately.
    Pass source_url to record the document's original URL (e.g. SEC EDGAR link).
    """
    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {_MAX_UPLOAD_BYTES // 1024 // 1024} MB limit.",
        )

    filename = file.filename or "system_doc"
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: '{ext}'.",
        )

    # Get a shared subllm instance (reuse admin's session for Flash-Lite)
    agent = session_manager.get_or_create(admin.id)
    subllm = agent._subllm

    background_tasks.add_task(
        _index_system_document, content, filename, subllm, source_url
    )
    return {
        "message": f"System document '{filename}' queued for indexing.",
        "filename": filename,
        "scope": "system",
        "indexing": True,
    }


async def _index_system_document(
    file_bytes: bytes,
    filename: str,
    subllm: Any,
    source_url: str | None,
) -> None:
    try:
        from ai_financial_analyst.pageindex import index_document
        doc_id = await index_document(
            file_bytes=file_bytes,
            filename=filename,
            user_id=None,       # NULL → system document
            subllm=subllm,
            scope="system",
            source_url=source_url,
        )
        logger.info("Admin: system document indexed — %s → doc_id=%s", filename, doc_id)
    except Exception as exc:
        logger.error("Admin: system document indexing failed — %s: %s", filename, exc)


# ---------------------------------------------------------------------------
# List system documents
# ---------------------------------------------------------------------------

@router.get("/documents")
async def list_system_documents(
    admin: CurrentUser = Depends(require_admin),
) -> list[dict]:
    """Return all indexed system documents."""
    from backend.core.database import async_session_factory
    from backend.core.models import Document
    from sqlalchemy import select

    async with async_session_factory() as session:
        result = await session.execute(
            select(Document)
            .where(Document.scope == "system")
            .order_by(Document.created_at.desc())
        )
        docs = result.scalars().all()
        return [
            {
                "id":          d.id,
                "filename":    d.filename,
                "title":       d.title or d.filename,
                "file_type":   d.file_type,
                "status":      d.status,
                "total_pages": d.total_pages,
                "source_url":  d.source_url,
                "created_at":  d.created_at,
                "metadata":    d.doc_metadata,
            }
            for d in docs
        ]


# ---------------------------------------------------------------------------
# Update system document metadata
# ---------------------------------------------------------------------------

class UpdateDocumentRequest(BaseModel):
    title: str | None = None
    source_url: str | None = None
    metadata: dict | None = None


@router.patch("/documents/{document_id}")
async def update_system_document(
    document_id: str,
    body: UpdateDocumentRequest,
    admin: CurrentUser = Depends(require_admin),
) -> dict:
    """Update title, source_url, or metadata for a system document."""
    from backend.core.database import async_session_factory
    from backend.core.models import Document
    from sqlalchemy import select
    import time

    async with async_session_factory() as session:
        result = await session.execute(
            select(Document).where(Document.id == document_id, Document.scope == "system")
        )
        doc = result.scalar_one_or_none()
        if not doc:
            raise HTTPException(status_code=404, detail="System document not found.")
        if body.title      is not None: doc.title      = body.title
        if body.source_url is not None: doc.source_url = body.source_url
        if body.metadata   is not None: doc.doc_metadata = body.metadata
        doc.updated_at = time.time()
        await session.commit()
        return {"id": doc.id, "title": doc.title, "updated_at": doc.updated_at}


# ---------------------------------------------------------------------------
# Delete system document
# ---------------------------------------------------------------------------

@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_system_document(
    document_id: str,
    admin: CurrentUser = Depends(require_admin),
) -> None:
    """Remove a system document and all its pages (CASCADE)."""
    from backend.core.database import async_session_factory
    from backend.core.models import Document
    from sqlalchemy import select, delete

    async with async_session_factory() as session:
        result = await session.execute(
            select(Document).where(Document.id == document_id, Document.scope == "system")
        )
        doc = result.scalar_one_or_none()
        if not doc:
            raise HTTPException(status_code=404, detail="System document not found.")
        await session.execute(delete(Document).where(Document.id == document_id))
        await session.commit()


# ---------------------------------------------------------------------------
# Admin: overview of all user-uploaded documents
# ---------------------------------------------------------------------------

@router.get("/documents/users")
async def list_all_user_documents(
    admin: CurrentUser = Depends(require_admin),
) -> list[dict]:
    """Admin view: all user-uploaded documents across all users."""
    from backend.core.database import async_session_factory
    from backend.core.models import Document
    from sqlalchemy import select

    async with async_session_factory() as session:
        result = await session.execute(
            select(Document)
            .where(Document.scope == "user")
            .order_by(Document.created_at.desc())
            .limit(500)
        )
        docs = result.scalars().all()
        return [
            {
                "id":          d.id,
                "user_id":     d.user_id,
                "filename":    d.filename,
                "file_type":   d.file_type,
                "status":      d.status,
                "total_pages": d.total_pages,
                "created_at":  d.created_at,
            }
            for d in docs
        ]
