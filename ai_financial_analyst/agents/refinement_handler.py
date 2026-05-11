"""Refinement handler — surgical str_replace document editing.

Mirrors the same approach used by Claude Code's file editor:
  1. LLM receives the FULL report (no truncation)
  2. LLM outputs exactly two fields: old_string and new_string
  3. We do a literal str.replace(old_string, new_string, 1) in the document
  4. If old_string is not found (LLM hallucinated it) → retry once with an error hint
  5. The persisted report is updated so future refinements and exports use the latest version

This is deterministic, preserves all unchanged sections character-perfect,
and is faster than whole-document regeneration for small edits.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import aiosqlite
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.llm import content_to_str

logger = logging.getLogger(__name__)

_EDITOR_SYSTEM = (
    "You are a precise financial report editor. "
    "Apply the user's requested change surgically — modify only what is asked. "
    "Preserve all Markdown formatting. "
    "Never invent numbers — use only figures that already appear in the report."
)

_STR_REPLACE_PROMPT = """\
Apply the following edit to the financial analysis report below.

Edit request: {instruction}

Full report:
```markdown
{full_report}
```

Output ONLY a JSON object with exactly two keys:
- "old_string": the exact verbatim text from the report that should be replaced.
  It must be unique within the document. Include a line of context above and below
  the section you are changing to ensure uniqueness.
- "new_string": the replacement text, preserving Markdown structure.

Rules:
- old_string must exist verbatim in the report (copy-paste it carefully)
- Do NOT invent numerical figures not already in the report
- For additions (new section), set old_string to the adjacent heading line and
  include it unchanged in new_string, followed by the new content
- Output only valid JSON — no explanation, no markdown fences

JSON:"""

_RETRY_PROMPT = """\
Your previous edit attempt failed because the old_string you provided was not found
in the report verbatim. Here is the error:

old_string you provided:
```
{bad_old_string}
```

This exact text does not appear in the report. Please try again with the correct
verbatim text from the report.

Edit request: {instruction}

Full report:
```markdown
{full_report}
```

Output ONLY the corrected JSON object (old_string + new_string):"""


def _extract_json(text: str) -> str:
    """Strip markdown code fences and extract the first JSON object."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rstrip("`").strip()
    # Find the first complete JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


async def refine_analysis(
    message: str,
    conversation_id: str,
    user_id: str,
    primary_llm: Any,
    db_path: str,
) -> str:
    """Apply a surgical str_replace edit to the latest stored report."""
    report_data = await _load_latest_report(conversation_id, user_id, db_path)

    if not report_data:
        return (
            "I don't have a stored analysis to refine for this conversation. "
            "Please run a financial analysis first — e.g. *\"Analyse AAPL\"*."
        )

    full_report: str = report_data["report_markdown"]  # no truncation
    tickers: str = report_data.get("tickers", "")

    # --- First attempt ---
    result = await _attempt_str_replace(message, full_report, primary_llm)

    if isinstance(result, str):
        # Success — result is the updated document
        updated_report = result
    else:
        # result is the bad old_string — retry once with corrective prompt
        bad_old_string = result["bad_old_string"]
        logger.warning("str_replace: old_string not found, retrying. Bad string: %s…", bad_old_string[:80])
        retry_result = await _retry_str_replace(message, full_report, bad_old_string, primary_llm)

        if isinstance(retry_result, str):
            updated_report = retry_result
        else:
            # Both attempts failed — fall back to a descriptive error
            return (
                "I wasn't able to locate the exact passage to edit. "
                "Could you be more specific about which section you'd like me to change?"
            )

    # Persist the updated report so future refinements and exports use the latest version
    try:
        await _save_updated_report(updated_report, conversation_id, user_id, db_path)
    except Exception as exc:
        logger.warning("Could not persist updated report: %s", exc)

    return f"Here is the updated analysis for **{tickers}**:\n\n{updated_report}"


async def _attempt_str_replace(
    instruction: str,
    full_report: str,
    primary_llm: Any,
) -> str | dict:
    """
    Returns the updated document string on success.
    Returns {"bad_old_string": ...} if old_string was not found.
    """
    try:
        response = await primary_llm.ainvoke([
            SystemMessage(content=_EDITOR_SYSTEM),
            HumanMessage(content=_STR_REPLACE_PROMPT.format(
                instruction=instruction,
                full_report=full_report,
            )),
        ])
        raw = content_to_str(response.content if hasattr(response, "content") else response)
        parsed = json.loads(_extract_json(raw))
        old_string: str = parsed["old_string"]
        new_string: str = parsed["new_string"]
    except Exception as exc:
        logger.error("str_replace: LLM or parse error: %s", exc)
        raise

    if old_string not in full_report:
        return {"bad_old_string": old_string}

    return full_report.replace(old_string, new_string, 1)


async def _retry_str_replace(
    instruction: str,
    full_report: str,
    bad_old_string: str,
    primary_llm: Any,
) -> str | dict:
    """Second attempt after showing the LLM its mistake."""
    try:
        response = await primary_llm.ainvoke([
            SystemMessage(content=_EDITOR_SYSTEM),
            HumanMessage(content=_RETRY_PROMPT.format(
                bad_old_string=bad_old_string,
                instruction=instruction,
                full_report=full_report,
            )),
        ])
        raw = content_to_str(response.content if hasattr(response, "content") else response)
        parsed = json.loads(_extract_json(raw))
        old_string: str = parsed["old_string"]
        new_string: str = parsed["new_string"]
    except Exception as exc:
        logger.error("str_replace retry: LLM or parse error: %s", exc)
        return {"bad_old_string": bad_old_string}

    if old_string not in full_report:
        return {"bad_old_string": old_string}

    return full_report.replace(old_string, new_string, 1)


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


async def _save_updated_report(
    updated_markdown: str,
    conversation_id: str,
    user_id: str,
    db_path: str,
) -> None:
    """Update the most recent report row with the edited markdown."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE reports SET report_markdown = ?"
            " WHERE id = (SELECT id FROM reports"
            " WHERE conversation_id = ? AND user_id = ?"
            " ORDER BY created_at DESC LIMIT 1)",
            (updated_markdown, conversation_id, user_id),
        )
        await db.commit()
