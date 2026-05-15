"""SQLAlchemy models for the backend database."""

from __future__ import annotations

import time
import uuid

from sqlalchemy import Boolean, CheckConstraint, Column, Float, Integer, String, Text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def generate_uuid() -> str:
    return str(uuid.uuid4())


def current_time() -> float:
    return time.time()


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    email = Column(String, nullable=False, unique=True)
    display_name = Column(String, default="")
    picture_url = Column(String, default="")
    created_at = Column(Float, default=current_time, nullable=False)
    last_seen_at = Column(Float, default=current_time, nullable=False)


class Preference(Base):
    __tablename__ = "preferences"

    key = Column(String, primary_key=True)
    value = Column(String, nullable=False)
    updated_at = Column(Float, default=current_time, nullable=False)
    user_id = Column(String, primary_key=True, default="default", index=True)


class AnalysisSummary(Base):
    __tablename__ = "analysis_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    tickers = Column(String, nullable=False)
    summary_text = Column(Text, nullable=False)
    run_id = Column(String, default="")
    created_at = Column(Float, default=current_time, nullable=False)
    user_id = Column(String, default="default", index=True)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=generate_uuid)
    title = Column(String, nullable=False, default="New conversation")
    created_at = Column(Float, default=current_time, nullable=False)
    updated_at = Column(Float, default=current_time, nullable=False)
    user_id = Column(String, default="default", index=True)


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    conversation_id = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    intent = Column(String, default="")
    tickers = Column(String, default="")
    charts = Column(JSONB, default=list)
    report_id = Column(String, nullable=True)
    created_at = Column(Float, default=current_time, nullable=False)
    user_id = Column(String, default="default")


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(String, primary_key=True, default=generate_uuid)
    conversation_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    message_index = Column(Integer, nullable=False)
    rating = Column(Integer, nullable=False)
    created_at = Column(Float, default=current_time, nullable=False)


class Report(Base):
    __tablename__ = "reports"

    id = Column(String, primary_key=True, default=generate_uuid)
    conversation_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False)
    tickers = Column(String, nullable=False)
    report_markdown = Column(Text, nullable=False)
    raw_data_json = Column(Text, default="{}")
    analysis_json = Column(Text, default="{}")
    created_at = Column(Float, default=current_time, nullable=False)


Index("idx_reports_conv", Report.conversation_id, Report.created_at.desc())
Index("idx_conversations_user", Conversation.user_id, Conversation.updated_at.desc())
Index("idx_messages_conv", Message.conversation_id, Message.created_at)
Index("idx_summaries_user", AnalysisSummary.user_id, AnalysisSummary.created_at.desc())

# ---------------------------------------------------------------------------
# PageIndex — document storage and page-level retrieval
# ---------------------------------------------------------------------------
#
# scope = 'user'   → private to uploading user; user_id required
# scope = 'system' → visible to all authenticated users; user_id is NULL
# ---------------------------------------------------------------------------

try:
    from pgvector.sqlalchemy import Vector as _PGVector
    _EMBEDDING_COL = lambda: Column(_PGVector(768), nullable=True)  # noqa: E731
    HAS_PGVECTOR = True
except ImportError:
    # SQLite dev fallback — embeddings stored as JSON-serialised text
    _EMBEDDING_COL = lambda: Column(Text, nullable=True)  # noqa: E731
    HAS_PGVECTOR = False


class Document(Base):
    """Registry of all indexed documents (user-uploaded and system-level)."""
    __tablename__ = "documents"

    id           = Column(String, primary_key=True, default=generate_uuid)
    user_id      = Column(String, nullable=True,  index=True)   # NULL for system docs
    scope        = Column(String, nullable=False, default="user")  # 'user' | 'system'
    filename     = Column(String, nullable=False)
    file_type    = Column(String, nullable=False)               # pdf, docx, txt, md, html
    title        = Column(String, nullable=True)
    author       = Column(String, nullable=True)
    source_url   = Column(String, nullable=True)               # original URL (system docs)
    status       = Column(String, nullable=False, default="pending")  # pending|indexing|ready|error
    total_pages  = Column(Integer, nullable=True)
    total_chars  = Column(Integer, nullable=True)
    checksum     = Column(String, nullable=True, index=True)   # SHA256 for dedup
    version      = Column(Integer, nullable=False, default=1)
    raw_file_ref = Column(String, nullable=True)               # storage path
    doc_metadata = Column(JSONB, nullable=True)                # tags, fiscal_year, etc.
    created_at   = Column(Float, default=current_time, nullable=False)
    updated_at   = Column(Float, default=current_time, nullable=False)

    __table_args__ = (
        CheckConstraint("scope != 'user' OR user_id IS NOT NULL",  name="user_doc_has_owner"),
        CheckConstraint("scope != 'system' OR user_id IS NULL",    name="system_doc_no_owner"),
        Index("idx_documents_user_scope", "user_id", "scope"),
    )


class DocumentPage(Base):
    """Page-level index — atomic unit of retrieval in the PageIndex system."""
    __tablename__ = "document_pages"

    id                 = Column(String, primary_key=True, default=generate_uuid)
    document_id        = Column(String, ForeignKey("documents.id", ondelete="CASCADE"),
                                nullable=False, index=True)
    user_id            = Column(String, nullable=True, index=True)  # denorm from Document
    scope              = Column(String, nullable=False, default="user")

    page_number        = Column(Integer, nullable=False)
    section_path       = Column(String, nullable=True)     # dotted path e.g. "2.3.1"
    heading_breadcrumb = Column(JSONB,   nullable=True)    # ["Chapter 2", "Revenue"]
    content            = Column(Text,    nullable=False)
    content_summary    = Column(Text,    nullable=True)    # 2-3 sentence Flash-Lite summary
    tables_json        = Column(JSONB,   nullable=True)    # extracted tables
    has_figures        = Column(Boolean, nullable=False, default=False)
    word_count         = Column(Integer, nullable=True)
    token_estimate     = Column(Integer, nullable=True)
    embedding          = _EMBEDDING_COL()                  # Vector(768) or Text fallback

    # Document structure navigation (set after bulk insert)
    prev_page_id       = Column(String, ForeignKey("document_pages.id"), nullable=True)
    next_page_id       = Column(String, ForeignKey("document_pages.id"), nullable=True)
    parent_section_id  = Column(String, ForeignKey("document_pages.id"), nullable=True)
    is_toc             = Column(Boolean, nullable=False, default=False)
    is_bibliography    = Column(Boolean, nullable=False, default=False)
    created_at         = Column(Float, default=current_time, nullable=False)

    __table_args__ = (
        Index("idx_pages_doc_page",   "document_id", "page_number"),
        Index("idx_pages_user_scope", "user_id", "scope"),
    )


class PageLink(Base):
    """Semantic cross-references between pages."""
    __tablename__ = "page_links"

    id             = Column(String, primary_key=True, default=generate_uuid)
    source_page_id = Column(String, ForeignKey("document_pages.id", ondelete="CASCADE"),
                            nullable=False, index=True)
    target_page_id = Column(String, ForeignKey("document_pages.id", ondelete="CASCADE"),
                            nullable=False)
    link_type      = Column(String, nullable=False)  # reference|continuation|defines|cited_by
    confidence     = Column(Float, nullable=True)
