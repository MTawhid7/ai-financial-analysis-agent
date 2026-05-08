"""Refinement handler — modifies a previous analysis result per user instructions.

Instead of re-running the full pipeline, this handler:
1. Retrieves the stored report and analysis data from the reports table.
2. Calls Flash primary LLM with the original data + the user's modification request.
3. Returns the modified/updated section.

This covers both structural modifications ("add a risks section") and
conceptual modifications ("make the bear case more pessimistic").
"""

from __future__ import annotations

import json
import logging
from typing import Any

import aiosqlite
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.llm import content_to_str

logger = logging.getLogger(__name__)

_REFINEMENT_SYSTEM = (
    "You are a senior financial analyst editor. The user wants to refine or update "
    "a previously generated financial analysis. Apply the requested change precisely. "
    "Use ONLY figures from the provided analysis data — never invent numbers. "
    "Return the modified section or updated analysis in professional Markdown."
)

_REFINEMENT_PROMPT = """\
The user wants to modify a previous financial analysis.

User's modification request: {request}

Original report (excerpt):
{report_excerpt}

Original quantitative analysis (for reference):
{analysis_json}

Instructions:
- Apply the user's requested change to the relevant section(s).
- If the change is structural (add/remove/rewrite a section), return the updated section.
- If the change is conceptual ("more pessimistic", "assume X% growth"), revise the
  bull/bear case or conclusion accordingly, citing specific metrics where possible.
- Keep the professional financial analyst tone.
- Do NOT invent new numbers — only use figures from the analysis data above.

Provide the modified output now:
"""


async def refine_analysis(
    message: str,
    conversation_id: str,
    user_id: str,
    primary_llm: Any,
    db_path: str,
) -> str:
    """Apply user's refinement request to the latest stored report."""
    report_data = await _load_latest_report(conversation_id, user_id, db_path)

    if not report_data:
        return (
            "I don't have a stored analysis to refine for this conversation. "
            "Please run a financial analysis first — e.g. *\"Analyse AAPL\"* — "
            "and then ask me to refine it."
        )

    report_excerpt = report_data["report_markdown"][:3500]
    analysis_summary = {
        ticker: {
            k: v for k, v in ta.items()
            if k in ("price_cagr_5y_pct", "company_pe", "sector_pe_avg",
                     "pe_vs_sector_premium_pct", "sector", "bull_case",
                     "bear_case", "closest_peer")
        }
        for ticker, ta in report_data["analysis"].items()
    }

    try:
        response = await primary_llm.ainvoke([
            SystemMessage(content=_REFINEMENT_SYSTEM),
            HumanMessage(content=_REFINEMENT_PROMPT.format(
                request=message,
                report_excerpt=report_excerpt,
                analysis_json=json.dumps(analysis_summary, indent=2, default=str)[:2000],
            )),
        ])
        result = content_to_str(
            response.content if hasattr(response, "content") else response
        ).strip()
        tickers = report_data.get("tickers", "")
        return f"Here is the updated analysis for **{tickers}**:\n\n{result}"

    except Exception as exc:
        logger.error("Refinement LLM call failed: %s", exc)
        return f"I couldn't apply the refinement: `{exc}`. Please try again."


async def _load_latest_report(
    conversation_id: str, user_id: str, db_path: str
) -> dict | None:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT tickers, report_markdown, raw_data_json, analysis_json"
            " FROM reports WHERE conversation_id = ? AND user_id = ?"
            " ORDER BY created_at DESC LIMIT 1",
            (conversation_id, user_id),
        ) as cursor:
            row = await cursor.fetchone()

    if not row:
        return None

    return {
        "tickers": row[0],
        "report_markdown": row[1],
        "raw_data": json.loads(row[2] or "{}"),
        "analysis": json.loads(row[3] or "{}"),
    }
