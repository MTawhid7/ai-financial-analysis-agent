"""Hybrid PageIndex retriever — vector similarity + FTS + Reciprocal Rank Fusion.

On PostgreSQL with pgvector: uses both vector ANN search and Postgres FTS.
On SQLite (dev): uses FTS-only (LIKE-based) since vector ops are unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_RRF_K = 60   # standard RRF constant


@dataclass
class PageResult:
    """A single page retrieved from the PageIndex."""
    page_id:            str
    document_id:        str
    document_title:     str
    filename:           str
    page_number:        int
    section_path:       str
    heading_breadcrumb: list[str]
    content:            str
    content_summary:    str
    tables:             list[dict]
    score:              float
    scope:              str   # 'user' | 'system'
    context_pages:      list["PageResult"] = field(default_factory=list)

    @property
    def citation(self) -> str:
        """Human-readable citation string."""
        heading = self.heading_breadcrumb[-1] if self.heading_breadcrumb else ""
        location = f"p. {self.page_number}"
        if heading:
            location += f" — {heading}"
        return f"{self.document_title}, {location}"

    def to_markdown(self) -> str:
        """Format page content as Markdown for the LLM response."""
        lines = [f"**{self.citation}**"]
        if self.content_summary:
            lines.append(f"*{self.content_summary}*")
        lines.append("")
        lines.append(self.content[:2000])
        if self.tables:
            for tbl in self.tables:
                lines.append(_table_to_markdown(tbl))
        return "\n".join(lines)


def _table_to_markdown(tbl: dict) -> str:
    headers = tbl.get("headers", [])
    rows    = tbl.get("rows", [])
    if not headers:
        return ""
    md = "| " + " | ".join(str(h) for h in headers) + " |\n"
    md += "| " + " | ".join("---" for _ in headers) + " |\n"
    for row in rows[:10]:
        md += "| " + " | ".join(str(c) for c in row) + " |\n"
    return md


# ---------------------------------------------------------------------------
# Main search function
# ---------------------------------------------------------------------------

async def search_documents(
    query: str,
    user_id: str,
    top_k: int = 8,
    document_ids: list[str] | None = None,
) -> list[PageResult]:
    """Hybrid search: vector similarity + FTS, merged via RRF.

    Always searches both the user's private documents (scope='user') and
    all system documents (scope='system') in a single query.
    """
    from backend.core.database import async_session_factory, engine
    from backend.core.models import HAS_PGVECTOR

    is_pg = str(engine.url).startswith("postgresql")

    async with async_session_factory() as session:
        if is_pg and HAS_PGVECTOR:
            vector_rows = await _vector_search(session, query, user_id, top_k * 2, document_ids)
            fts_rows    = await _fts_search(session, query, user_id, top_k * 2, document_ids)
            ranked      = _rrf_merge(vector_rows, fts_rows)[:top_k]
        else:
            # SQLite / no pgvector fallback
            ranked = await _fts_search_sqlite(session, query, user_id, top_k, document_ids)

        # Expand each result with prev/next pages for context
        results = []
        for row, score in ranked:
            pr = _row_to_page_result(row, score)
            pr.context_pages = await _get_context_pages(session, row)
            results.append(pr)

    return results


async def get_page(page_id: str, user_id: str) -> PageResult | None:
    """Retrieve a specific page by ID, enforcing access control."""
    from backend.core.database import async_session_factory
    from backend.core.models import DocumentPage, Document
    from sqlalchemy import select

    async with async_session_factory() as session:
        result = await session.execute(
            select(DocumentPage, Document)
            .join(Document, Document.id == DocumentPage.document_id)
            .where(
                DocumentPage.id == page_id,
                _access_filter(DocumentPage, user_id),
            )
        )
        row = result.first()
        if not row:
            return None
        dp, doc = row
        return PageResult(
            page_id=dp.id, document_id=dp.document_id,
            document_title=doc.title or doc.filename,
            filename=doc.filename,
            page_number=dp.page_number, section_path=dp.section_path or "",
            heading_breadcrumb=dp.heading_breadcrumb or [],
            content=dp.content, content_summary=dp.content_summary or "",
            tables=dp.tables_json or [], score=1.0, scope=dp.scope,
        )


async def get_document_page_by_number(
    document_id: str,
    page_number: int,
    user_id: str,
) -> PageResult | None:
    """Retrieve a specific page by document + page number."""
    from backend.core.database import async_session_factory
    from backend.core.models import DocumentPage, Document
    from sqlalchemy import select

    async with async_session_factory() as session:
        result = await session.execute(
            select(DocumentPage, Document)
            .join(Document, Document.id == DocumentPage.document_id)
            .where(
                DocumentPage.document_id == document_id,
                DocumentPage.page_number == page_number,
                _access_filter(DocumentPage, user_id),
            )
        )
        row = result.first()
        if not row:
            return None
        dp, doc = row
        pr = PageResult(
            page_id=dp.id, document_id=dp.document_id,
            document_title=doc.title or doc.filename,
            filename=doc.filename,
            page_number=dp.page_number, section_path=dp.section_path or "",
            heading_breadcrumb=dp.heading_breadcrumb or [],
            content=dp.content, content_summary=dp.content_summary or "",
            tables=dp.tables_json or [], score=1.0, scope=dp.scope,
        )
        pr.context_pages = await _get_context_pages(session, dp)
        return pr


async def list_user_documents(user_id: str) -> list[dict]:
    """Return metadata for all documents the user can access."""
    from backend.core.database import async_session_factory
    from backend.core.models import Document
    from sqlalchemy import select, or_

    async with async_session_factory() as session:
        result = await session.execute(
            select(Document)
            .where(
                or_(
                    (Document.user_id == user_id) & (Document.scope == "user"),
                    Document.scope == "system",
                )
            )
            .order_by(Document.updated_at.desc())
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
                "scope":       d.scope,
                "created_at":  d.created_at,
            }
            for d in docs
        ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _access_filter(PageModel: Any, user_id: str) -> Any:
    """SQLAlchemy WHERE clause: user's private docs OR all system docs."""
    from sqlalchemy import or_
    return or_(
        (PageModel.user_id == user_id) & (PageModel.scope == "user"),
        PageModel.scope == "system",
    )


async def _vector_search(session: Any, query: str, user_id: str, k: int,
                          doc_ids: list[str] | None) -> list[tuple[Any, float]]:
    """Vector ANN search using pgvector cosine similarity."""
    from .embedder import embed_query
    from backend.core.models import DocumentPage, Document
    from sqlalchemy import select, text

    query_vec = await embed_query(query)
    vec_literal = "[" + ",".join(f"{v:.6f}" for v in query_vec) + "]"

    stmt = (
        select(DocumentPage, Document,
               text(f"1 - (document_pages.embedding <=> '{vec_literal}'::vector) AS vscore"))
        .join(Document, Document.id == DocumentPage.document_id)
        .where(_access_filter(DocumentPage, user_id))
        .order_by(text(f"document_pages.embedding <=> '{vec_literal}'::vector"))
        .limit(k)
    )
    if doc_ids:
        stmt = stmt.where(DocumentPage.document_id.in_(doc_ids))

    result = await session.execute(stmt)
    return [(row.DocumentPage, float(getattr(row, "vscore", 0.0))) for row in result.fetchall()]


async def _fts_search(session: Any, query: str, user_id: str, k: int,
                       doc_ids: list[str] | None) -> list[tuple[Any, float]]:
    """Postgres full-text search with ts_rank_cd scoring."""
    from backend.core.models import DocumentPage, Document
    from sqlalchemy import select, text, func

    query_terms = " | ".join(
        w.strip() for w in query.split() if len(w.strip()) > 2
    ) or query

    tsquery_expr = func.to_tsquery("english", query_terms)
    tsvec_expr   = func.to_tsvector("english", DocumentPage.content)
    rank_expr    = func.ts_rank_cd(tsvec_expr, tsquery_expr)

    stmt = (
        select(DocumentPage, Document, rank_expr.label("fts_score"))
        .join(Document, Document.id == DocumentPage.document_id)
        .where(
            _access_filter(DocumentPage, user_id),
            tsvec_expr.op("@@")(tsquery_expr),
        )
        .order_by(rank_expr.desc())
        .limit(k)
    )
    if doc_ids:
        stmt = stmt.where(DocumentPage.document_id.in_(doc_ids))

    result = await session.execute(stmt)
    return [(row.DocumentPage, float(getattr(row, "fts_score", 0.0))) for row in result.fetchall()]


async def _fts_search_sqlite(session: Any, query: str, user_id: str, k: int,
                               doc_ids: list[str] | None) -> list[tuple[Any, float]]:
    """SQLite LIKE-based fallback (no vector ops available)."""
    from backend.core.models import DocumentPage, Document
    from sqlalchemy import select, or_

    words = [w.strip() for w in query.split() if len(w.strip()) > 2][:5]
    stmt = (
        select(DocumentPage, Document)
        .join(Document, Document.id == DocumentPage.document_id)
        .where(_access_filter(DocumentPage, user_id))
    )
    if words:
        stmt = stmt.where(
            or_(*[DocumentPage.content.ilike(f"%{w}%") for w in words])
        )
    if doc_ids:
        stmt = stmt.where(DocumentPage.document_id.in_(doc_ids))
    stmt = stmt.limit(k)

    result = await session.execute(stmt)
    return [(row.DocumentPage, 1.0) for row in result.fetchall()]


def _rrf_merge(
    vector_rows: list[tuple[Any, float]],
    fts_rows:    list[tuple[Any, float]],
) -> list[tuple[Any, float]]:
    """Merge two ranked lists with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    page_map: dict[str, Any] = {}

    for rank, (page, _) in enumerate(vector_rows):
        scores[page.id] = scores.get(page.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        page_map[page.id] = page

    for rank, (page, _) in enumerate(fts_rows):
        scores[page.id] = scores.get(page.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        page_map[page.id] = page

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(page_map[pid], score) for pid, score in ranked]


def _row_to_page_result(page: Any, score: float) -> PageResult:
    # We need the document title — fetched lazily in search_documents
    return PageResult(
        page_id=page.id,
        document_id=page.document_id,
        document_title=getattr(page, "_doc_title", page.document_id),
        filename=getattr(page, "_doc_filename", ""),
        page_number=page.page_number,
        section_path=page.section_path or "",
        heading_breadcrumb=page.heading_breadcrumb or [],
        content=page.content,
        content_summary=page.content_summary or "",
        tables=page.tables_json or [],
        score=score,
        scope=page.scope,
    )


async def _get_context_pages(session: Any, page: Any) -> list[PageResult]:
    """Fetch the immediately adjacent pages (prev and next) for context."""
    from backend.core.models import DocumentPage, Document
    from sqlalchemy import select

    neighbour_ids = [pid for pid in [page.prev_page_id, page.next_page_id] if pid]
    if not neighbour_ids:
        return []

    result = await session.execute(
        select(DocumentPage, Document)
        .join(Document, Document.id == DocumentPage.document_id)
        .where(DocumentPage.id.in_(neighbour_ids))
    )
    return [
        PageResult(
            page_id=row.DocumentPage.id,
            document_id=row.DocumentPage.document_id,
            document_title=row.Document.title or row.Document.filename,
            filename=row.Document.filename,
            page_number=row.DocumentPage.page_number,
            section_path=row.DocumentPage.section_path or "",
            heading_breadcrumb=row.DocumentPage.heading_breadcrumb or [],
            content=row.DocumentPage.content,
            content_summary=row.DocumentPage.content_summary or "",
            tables=row.DocumentPage.tables_json or [],
            score=0.0,
            scope=row.DocumentPage.scope,
        )
        for row in result.fetchall()
    ]
