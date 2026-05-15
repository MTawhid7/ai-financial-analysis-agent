"""End-to-end PageIndex ingestion pipeline.

index_document(file_bytes, filename, user_id, subllm, scope='user')
  → parse pages → summarise → embed → store → return document_id
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_UPLOAD_DIR = os.getenv("UPLOAD_DIR", ".uploads")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def index_document(
    file_bytes: bytes,
    filename: str,
    user_id: str | None,
    subllm: Any,
    scope: str = "user",
    source_url: str | None = None,
    extra_metadata: dict | None = None,
) -> str:
    """Index a document into the PageIndex store.

    Returns the document_id.  For 'user' scope, user_id is required.
    For 'system' scope, user_id must be None.
    """
    from backend.core.database import async_session_factory
    from backend.core.models import Document, DocumentPage, PageLink, HAS_PGVECTOR
    from ..parsers._page_extractor import extract_pages, RawPage
    from .embedder import embed_texts
    from .ocr import is_scanned_pdf, ocr_pdf_pages

    file_type = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    checksum  = hashlib.sha256(file_bytes).hexdigest()

    async with async_session_factory() as session:
        # ── Dedup: return existing document_id if already indexed ──────────
        from sqlalchemy import select
        existing_q = select(Document).where(
            Document.checksum == checksum,
            Document.scope    == scope,
        )
        if scope == "user":
            existing_q = existing_q.where(Document.user_id == user_id)

        result = await session.execute(existing_q)
        existing = result.scalar_one_or_none()
        if existing and existing.status == "ready":
            logger.info("Document %s already indexed (%s) — skipping", filename, existing.id)
            return existing.id

        # ── Create Document row ─────────────────────────────────────────────
        doc_id = existing.id if existing else str(uuid.uuid4())
        title  = _extract_title(filename, file_bytes, file_type)

        if not existing:
            doc = Document(
                id=doc_id, user_id=user_id, scope=scope,
                filename=filename, file_type=file_type, title=title,
                source_url=source_url,
                status="indexing",
                checksum=checksum,
                doc_metadata=extra_metadata or {},
                created_at=time.time(), updated_at=time.time(),
            )
            session.add(doc)
        else:
            existing.status = "indexing"
            existing.updated_at = time.time()
        await session.commit()

    # ── Extract pages ───────────────────────────────────────────────────────
    try:
        raw_pages = extract_pages(file_bytes, file_type)
    except Exception as exc:
        logger.error("Page extraction failed for %s: %s", filename, exc)
        raw_pages = []

    # ── OCR fallback for scanned PDFs ───────────────────────────────────────
    if file_type == "pdf" and raw_pages:
        avg_chars = sum(len(p.content) for p in raw_pages) / max(len(raw_pages), 1)
        if avg_chars < 50:
            logger.info("Scanned PDF detected for %s, running OCR…", filename)
            ocr_texts = ocr_pdf_pages(file_bytes)
            for i, page in enumerate(raw_pages):
                if i < len(ocr_texts) and ocr_texts[i]:
                    page.content = ocr_texts[i]

    if not raw_pages:
        logger.warning("No pages extracted from %s — marking as error", filename)
        await _set_status(doc_id, "error")
        return doc_id

    # ── Summarise each page (Flash-Lite, sequential to respect RPM) ─────────
    from ..parsers._summarise import hierarchical_summarise
    summaries: list[str] = []
    for page in raw_pages:
        if page.content.strip():
            try:
                # Short pages get a direct 2-sentence summary
                summary = await hierarchical_summarise(page.content[:3000], subllm)
            except Exception:
                summary = page.content[:300]
        else:
            summary = ""
        summaries.append(summary)

    # ── Generate embeddings ──────────────────────────────────────────────────
    texts_to_embed = [
        " | ".join(p.heading_breadcrumb) + "\n\n" + p.content
        for p in raw_pages
    ]
    vectors = await embed_texts(texts_to_embed)

    # ── Bulk insert DocumentPage rows ────────────────────────────────────────
    page_ids: list[str] = []
    async with async_session_factory() as session:
        # Delete old pages if re-indexing
        from sqlalchemy import delete
        await session.execute(
            delete(DocumentPage).where(DocumentPage.document_id == doc_id)
        )

        for i, (page, summary, vector) in enumerate(zip(raw_pages, summaries, vectors)):
            page_id = str(uuid.uuid4())
            page_ids.append(page_id)

            # Serialise vector: stored as pgvector if available, else JSON text
            if HAS_PGVECTOR:
                embedding_val = vector
            else:
                embedding_val = json.dumps(vector)

            dp = DocumentPage(
                id=page_id,
                document_id=doc_id,
                user_id=user_id,
                scope=scope,
                page_number=page.page_number,
                section_path=page.section_path or str(page.page_number),
                heading_breadcrumb=page.heading_breadcrumb,
                content=page.content,
                content_summary=summary,
                tables_json=page.tables if page.tables else None,
                has_figures=page.has_figures,
                word_count=page.word_count,
                token_estimate=page.token_estimate,
                embedding=embedding_val,
                is_toc=page.is_toc,
                is_bibliography=page.is_bibliography,
                created_at=time.time(),
            )
            session.add(dp)

        await session.commit()

    # ── Link pages (prev/next chain) ─────────────────────────────────────────
    await _link_pages(doc_id, page_ids)

    # ── Detect cross-references ──────────────────────────────────────────────
    await _detect_cross_references(doc_id, raw_pages, page_ids)

    # ── Update document status ────────────────────────────────────────────────
    async with async_session_factory() as session:
        result = await session.execute(
            select(Document).where(Document.id == doc_id)
        )
        doc_row = result.scalar_one_or_none()
        if doc_row:
            doc_row.status      = "ready"
            doc_row.total_pages = len(raw_pages)
            doc_row.total_chars = sum(len(p.content) for p in raw_pages)
            doc_row.title       = doc_row.title or title
            doc_row.updated_at  = time.time()
        await session.commit()

    logger.info("PageIndex: indexed %d pages for document %s (%s)", len(raw_pages), doc_id, filename)
    return doc_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _set_status(doc_id: str, status: str) -> None:
    from backend.core.database import async_session_factory
    from backend.core.models import Document
    from sqlalchemy import select
    async with async_session_factory() as session:
        result = await session.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if doc:
            doc.status     = status
            doc.updated_at = time.time()
            await session.commit()


async def _link_pages(doc_id: str, page_ids: list[str]) -> None:
    """Set prev/next pointers on DocumentPage rows."""
    from backend.core.database import async_session_factory
    from backend.core.models import DocumentPage
    from sqlalchemy import select, update
    async with async_session_factory() as session:
        for i, pid in enumerate(page_ids):
            prev_id = page_ids[i - 1] if i > 0 else None
            next_id = page_ids[i + 1] if i < len(page_ids) - 1 else None
            await session.execute(
                update(DocumentPage)
                .where(DocumentPage.id == pid)
                .values(prev_page_id=prev_id, next_page_id=next_id)
            )
        await session.commit()


import re as _re
_SEE_PAGE_RE = _re.compile(r"\bsee\s+page\s+(\d+)\b", _re.I)


async def _detect_cross_references(doc_id: str, pages: list[Any], page_ids: list[str]) -> None:
    """Detect 'see page N' patterns and create PageLink rows."""
    from backend.core.database import async_session_factory
    from backend.core.models import PageLink
    import time as _time

    page_num_to_id = {p.page_number: page_ids[i] for i, p in enumerate(pages)}
    links_to_add: list[PageLink] = []

    for i, page in enumerate(pages):
        for m in _SEE_PAGE_RE.finditer(page.content):
            target_num = int(m.group(1))
            target_id  = page_num_to_id.get(target_num)
            if target_id and target_id != page_ids[i]:
                links_to_add.append(PageLink(
                    id=str(uuid.uuid4()),
                    source_page_id=page_ids[i],
                    target_page_id=target_id,
                    link_type="reference",
                    confidence=0.9,
                ))

    if links_to_add:
        async with async_session_factory() as session:
            session.add_all(links_to_add)
            await session.commit()


def _extract_title(filename: str, file_bytes: bytes, file_type: str) -> str:
    """Best-effort title extraction: try first-line of content, fall back to filename."""
    try:
        if file_type == "pdf":
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                if pdf.pages:
                    first_text = (pdf.pages[0].extract_text() or "").strip()
                    first_line = first_text.split("\n")[0][:120].strip()
                    if len(first_line) > 5:
                        return first_line
    except Exception:
        pass
    # Fallback: clean up the filename
    name = filename.rsplit(".", 1)[0]
    return name.replace("_", " ").replace("-", " ").strip()


import io  # noqa: E402 (import at bottom to keep top clean)
