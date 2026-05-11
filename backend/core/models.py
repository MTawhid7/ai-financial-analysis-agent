"""SQLAlchemy models for the backend database."""

from __future__ import annotations

import time
import uuid

from sqlalchemy import Column, Float, Integer, String, Text, ForeignKey, Index
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
    charts = Column(JSONB, default=list)  # New field for charts
    report_id = Column(String, nullable=True)  # New field for exported reports
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
