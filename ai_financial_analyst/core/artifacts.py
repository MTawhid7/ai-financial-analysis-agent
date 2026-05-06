"""RunArtifacts — full (untruncated) API and LLM response storage.

Written to run_artifacts_<run_id>.json after every pipeline run so that
individual tool responses and LLM exchanges can be inspected in detail.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunArtifacts:
    """Stores complete, untruncated API responses and LLM exchanges for one run."""

    def __init__(self, run_id: str, tickers: list[str]) -> None:
        self.run_id = run_id
        self.tickers = tickers
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._tool_responses: list[dict[str, Any]] = []
        self._llm_exchanges: list[dict[str, Any]] = []
        self._report_markdown: str = ""

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_tool_response(
        self,
        agent: str,
        tool: str,
        input_data: dict,
        full_output: str,
        step: int | None = None,
        cache_hit: bool = False,
    ) -> None:
        """Store a complete (untruncated) tool response string."""
        try:
            parsed = json.loads(full_output)
        except (json.JSONDecodeError, TypeError):
            parsed = None

        self._tool_responses.append({
            "step": step,
            "agent": agent,
            "tool": tool,
            "input": input_data,
            "full_output_str": full_output,
            "output_parsed": parsed,
            "cache_hit": cache_hit,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    def record_llm_exchange(
        self,
        agent: str,
        purpose: str,
        prompt_messages: list[dict[str, Any]],
        raw_response: str,
        ticker: str | None = None,
    ) -> None:
        """Store a complete LLM prompt + raw response pair."""
        self._llm_exchanges.append({
            "agent": agent,
            "purpose": purpose,
            "ticker": ticker,
            "prompt_messages": prompt_messages,
            "raw_response": raw_response,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    def set_report(self, markdown: str) -> None:
        self._report_markdown = markdown

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "tickers": self.tickers,
            "tool_responses": self._tool_responses,
            "llm_exchanges": self._llm_exchanges,
            "report_markdown": self._report_markdown,
        }

    def save(self, output_dir: str = ".") -> Path:
        path = Path(output_dir) / f"run_artifacts_{self.run_id[:8]}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path
