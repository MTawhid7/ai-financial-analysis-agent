"""End-to-end PageIndex ingestion pipeline.

index_document(file_bytes, filename, user_id, subllm, scope='user')
  → parse pages → summarise → embed → store → return document_id
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import time
import uuid
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)

_UPLOAD_DIR = settings.upload_dir


def _split_into_chunks(text: str) -> list[str]:
    """Split text into overlapping chunks on paragraph/sentence boundaries.

    Returns an empty list when the text fits within settings.pageindex_chunk_max_chars
    (no splitting needed).  Each chunk overlaps the previous by
    settings.pageindex_chunk_overlap_chars to preserve inter-sentence context.
    """
    max_chars = settings.pageindex_chunk_max_chars
    overlap   = settings.pageindex_chunk_overlap_chars

    if len(text) <= max_chars:
        return []

    # Prefer paragraph (\n\n) splits; fall back to sentence splits for monolithic text.
    paragraph_splits = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraph_splits) <= 1:
        # No paragraph structure — split on sentence boundaries (period+space)
        sentence_splits = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        units: list[str] = sentence_splits if len(sentence_splits) > 1 else [text]
    else:
        units = paragraph_splits

    sep = "\n\n" if len(paragraph_splits) > 1 else " "

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        unit_len = len(unit)
        if current_len + unit_len + len(sep) > max_chars and current:
            chunk_text = sep.join(current)
            chunks.append(chunk_text)
            # Carry the last overlap_chars of the previous chunk forward
            overlap_text = chunk_text[-overlap:] if len(chunk_text) > overlap else chunk_text
            current = [overlap_text] if overlap_text else []
            current_len = len(overlap_text) if overlap_text else 0
        current.append(unit)
        current_len += unit_len + len(sep)

    if current:
        chunks.append(sep.join(current))

    return chunks if len(chunks) > 1 else []


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

    # ── Build embed texts: root pages get full text; chunks get chunk text ────
    current_model = settings.llm_embedding_model

    root_texts = [
        " | ".join(p.heading_breadcrumb) + "\n\n" + p.content
        for p in raw_pages
    ]

    # Compute sub-page chunks for long pages.  Stored as (page_idx, chunk_idx, text).
    chunk_records: list[tuple[int, int, str]] = []
    for i, page in enumerate(raw_pages):
        chunks = _split_into_chunks(page.content)
        for ci, chunk_text in enumerate(chunks):
            chunk_records.append((i, ci + 1, chunk_text))

    # Embed root pages and all chunks in one batched call.
    chunk_texts = [ct for _, _, ct in chunk_records]
    all_texts   = root_texts + chunk_texts
    all_vectors = await embed_texts(all_texts)
    root_vectors  = all_vectors[:len(root_texts)]
    chunk_vectors = all_vectors[len(root_texts):]

    # ── Bulk insert DocumentPage rows ────────────────────────────────────────
    page_ids: list[str] = []
    async with async_session_factory() as session:
        # Delete old pages if re-indexing
        from sqlalchemy import delete
        await session.execute(
            delete(DocumentPage).where(DocumentPage.document_id == doc_id)
        )

        for i, (page, summary, vector) in enumerate(zip(raw_pages, summaries, root_vectors)):
            page_id = str(uuid.uuid4())
            page_ids.append(page_id)

            embedding_val = vector if HAS_PGVECTOR else json.dumps(vector)

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
                embedding_model=current_model,
                chunk_index=0,
                is_toc=page.is_toc,
                is_bibliography=page.is_bibliography,
                created_at=time.time(),
            )
            session.add(dp)

        # Insert sub-page chunks (chunk_index >= 1) for vector-search precision.
        for (page_idx, chunk_idx, chunk_text), chunk_vector in zip(chunk_records, chunk_vectors):
            page = raw_pages[page_idx]
            chunk_embedding = chunk_vector if HAS_PGVECTOR else json.dumps(chunk_vector)
            session.add(DocumentPage(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                user_id=user_id,
                scope=scope,
                page_number=page.page_number,
                section_path=page.section_path or str(page.page_number),
                heading_breadcrumb=page.heading_breadcrumb,
                content=chunk_text,        # chunk text only
                content_summary=None,      # display not needed for chunks
                has_figures=False,
                word_count=len(chunk_text.split()),
                token_estimate=len(chunk_text) // 4,
                embedding=chunk_embedding,
                embedding_model=current_model,
                chunk_index=chunk_idx,
                is_toc=False,
                is_bibliography=False,
                created_at=time.time(),
            ))

        await session.commit()

    if chunk_records:
        logger.info(
            "PageIndex: inserted %d sub-page chunks for document %s",
            len(chunk_records), doc_id,
        )

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


_SEE_PAGE_RE = re.compile(r"\bsee\s+page\s+(\d+)\b", re.I)


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
