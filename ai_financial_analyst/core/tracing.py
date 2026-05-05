"""Structured run_trace.json builder and LangSmith callback integration."""

from __future__ import annotations

import json
import logging
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    TOOL_ERROR = "TOOL_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    PARSING_ERROR = "PARSING_ERROR"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    STATE_VALIDATION_ERROR = "STATE_VALIDATION_ERROR"
    UNKNOWN = "UNKNOWN"


class RunStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    RATE_LIMITED = "RATE_LIMITED"
    FAILED = "FAILED"


class RunTracer:
    """Builds run_trace.json incrementally and exports it on completion.

    Also writes adversarial injection attempts to errors.jsonl.
    """

    def __init__(self, model: str = "gemini-3-flash-preview") -> None:
        self.run_id = str(uuid.uuid4())
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.model = model
        self._tool_calls: list[dict[str, Any]] = []
        self._errors: list[dict[str, Any]] = []
        self._status = RunStatus.COMPLETE
        self._step = 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record_tool_call(
        self,
        agent: str,
        tool: str,
        input_data: dict,
        output_data: Any,
        output_tokens: int = 0,
        cache_hit: bool = False,
    ) -> None:
        self._step += 1
        self._tool_calls.append(
            {
                "step": self._step,
                "agent": agent,
                "tool": tool,
                "input": input_data,
                "output": output_data,
                "output_tokens": output_tokens,
                "cache_hit": cache_hit,
            }
        )

    def record_error(
        self,
        error_type: ErrorType,
        agent: str,
        tool: str,
        message: str,
        input_data: dict | None = None,
    ) -> None:
        entry = {
            "error_type": error_type.value,
            "agent": agent,
            "tool": tool,
            "message": message,
            "input": input_data or {},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._errors.append(entry)
        logger.error("[%s] %s — %s: %s", agent, error_type.value, tool, message)

        # Append injection attempts to errors.jsonl for post-hoc review.
        if error_type == ErrorType.TOOL_ERROR and "injection" in message.lower():
            _append_jsonl("errors.jsonl", entry)

    def set_status(self, status: RunStatus) -> None:
        self._status = status

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def build(self, budget_stats: dict | None = None) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "model": self.model,
            "tool_calls": self._tool_calls,
            "errors": self._errors,
            "status": self._status.value,
            "budget_stats": budget_stats or {},
        }

    def export(self, output_dir: str = ".") -> Path:
        trace = self.build()
        path = Path(output_dir) / f"run_trace_{self.run_id[:8]}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trace, indent=2, default=str))
        logger.info("Run trace exported to %s", path)
        return path


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _append_jsonl(path: str, entry: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
