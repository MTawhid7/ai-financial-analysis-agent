"""Base classes and shared error schema for all LangChain tools."""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    TOOL_ERROR = "TOOL_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    PARSING_ERROR = "PARSING_ERROR"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    STATE_VALIDATION_ERROR = "STATE_VALIDATION_ERROR"
    UNKNOWN = "UNKNOWN"


class ToolError(BaseModel):
    """Structured error returned by any tool on failure.

    Tools never raise — they return ToolError JSON so the agent can
    self-correct rather than crash.
    """

    model_config = ConfigDict(extra="forbid")

    error_type: ErrorType
    tool: str
    message: str
    input: dict[str, Any]

    def to_json(self) -> str:
        return self.model_dump_json()


class StrictToolInput(BaseModel):
    """Base class for all tool input schemas.

    extra='forbid' rejects any field the LLM hallucinates beyond the schema,
    which prevents silent data corruption from mis-formatted tool calls.
    """

    model_config = ConfigDict(extra="forbid")


def safe_tool_call(tool_name: str, fn, input_data: dict) -> str:
    """Wrap a tool execution in try/except, returning ToolError JSON on failure.

    Returns the JSON string the agent receives as a tool observation.
    """
    try:
        return fn()
    except Exception as exc:
        error_type = ErrorType.RATE_LIMIT if "429" in str(exc) else ErrorType.TOOL_ERROR
        err = ToolError(
            error_type=error_type,
            tool=tool_name,
            message=str(exc),
            input=input_data,
        )
        logger.error("Tool %s failed: %s", tool_name, exc)
        return err.to_json()
