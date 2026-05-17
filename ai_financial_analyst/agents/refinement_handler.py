"""Refinement handler — surgical str_replace document editing.

Mirrors the same approach used by Claude Code's file editor:
  1. LLM receives the FULL report (no truncation)
  2. LLM outputs exactly two fields: old_string and new_string
  3. We try _flexible_str_replace (exact first, then line-strip fallback)
  4. If still not found → retry once with an error hint
  5. The updated report is persisted as a NEW row (INSERT, not UPDATE) so
     every edit is versioned; _load_latest_report picks it up via ORDER BY created_at DESC.

This is deterministic, preserves all unchanged sections character-perfect,
and is faster than whole-document regeneration for small edits.
Versioning via INSERT allows natural rollback to any prior version.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

import aiosqlite
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.llm import content_to_str

logger = logging.getLogger(__name__)


class EditConflictError(Exception):
    """Raised when a concurrent request modified the report since it was last read."""

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

_STR_REPLACE_SECTION_PROMPT = """\
Apply the following edit to the **{section_name}** section of the financial report.

Edit request: {instruction}

Section to edit:
```markdown
{section_text}
```

Output ONLY a JSON object with exactly two keys:
- "old_string": the exact verbatim text from the section above that should be replaced.
  It must be unique within the section. Include a line of context above and below.
- "new_string": the replacement text, preserving Markdown structure.

Rules:
- old_string must exist verbatim in the section text above (copy-paste carefully)
- Do NOT invent numerical figures not already in the report
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


# ── Section-aware helpers ────────────────────────────────────────────────────

_SECTION_KEYWORDS: dict[str, str] = {
    "executive summary": "Executive Summary",
    "bull case":         "Bull Case",
    "bear case":         "Bear Case",
    "quantitative":      "Quantitative Analysis",
    "conclusion":        "Conclusion",
    "financial overview": "Financial Overview",
    "data coverage":     "Data Coverage",
}


def _infer_target_section(message: str) -> str | None:
    """Detect which report section the user intends to edit from their message."""
    lower = message.lower()
    for keyword, section_name in _SECTION_KEYWORDS.items():
        if keyword in lower:
            return section_name
    return None


def _extract_section_context(report: str, section_name: str) -> tuple[str, str, str] | None:
    """Extract (before, section_text, after) for a named section.

    Returns None if the section heading cannot be found in the report.
    The section spans from its heading line to the start of the next heading.
    """
    # Use {{1,3}} to produce the literal regex quantifier {1,3} inside an f-string
    pattern = rf"(?m)(^#{{1,3}}\s+.*{re.escape(section_name)}.*$)"
    match = re.search(pattern, report, re.IGNORECASE)
    if not match:
        return None
    start = match.start()
    heading_end = match.end()  # position just after the heading line
    # Search for the next heading starting AFTER the current heading line
    next_hdr = re.search(r"(?m)^#{1,3}\s", report[heading_end:])
    end = heading_end + next_hdr.start() if next_hdr else len(report)
    return report[:start], report[start:end], report[end:]


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


def _strip_lines(s: str) -> str:
    """Strip trailing whitespace from each line (preserves leading whitespace)."""
    return "\n".join(line.rstrip() for line in s.splitlines())


def _flexible_str_replace(full_report: str, old_string: str, new_string: str) -> str | None:
    """Apply a str_replace with graceful whitespace tolerance.

    Try 1 — exact match: standard `str.replace(old, new, 1)`.
    Try 2 — line-strip match: if trailing whitespace differs on any line,
             strip all trailing whitespace from both sides and re-attempt.

    Returns the updated document string, or None if old_string cannot be found.
    """
    # Try 1: exact match
    if old_string in full_report:
        return full_report.replace(old_string, new_string, 1)

    # Try 2: line-strip normalisation (covers trailing-space differences)
    stripped_report = _strip_lines(full_report)
    stripped_old = _strip_lines(old_string)
    if stripped_old and stripped_old in stripped_report:
        idx = stripped_report.index(stripped_old)
        newlines_before = stripped_report[:idx].count("\n")
        n_lines = stripped_old.count("\n") + 1
        orig_lines = full_report.splitlines()
        original_block = "\n".join(orig_lines[newlines_before:newlines_before + n_lines])
        return full_report.replace(original_block, new_string, 1)

    return None


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

    full_report: str    = report_data["report_markdown"]  # no truncation
    tickers: str        = report_data.get("tickers", "")
    base_created_at     = report_data.get("created_at")   # optimistic lock token

    # --- Section-aware scoping: narrow context for the LLM when possible ---
    target_section = _infer_target_section(message)
    section_parts = _extract_section_context(full_report, target_section) if target_section else None
    if section_parts:
        logger.info("Refinement: scoping LLM to section '%s'", target_section)
    else:
        target_section = None  # fallback to full-document edit

    # --- First attempt ---
    result = await _attempt_str_replace(
        message, full_report, primary_llm,
        section_name=target_section,
        section_text=section_parts[1] if section_parts else None,
    )

    if isinstance(result, str):
        # Success — result is the updated document
        updated_report = result
    else:
        # result is the bad old_string — retry once with corrective prompt
        bad_old_string = result["bad_old_string"]
        logger.warning("str_replace: old_string not found, retrying. Bad string: %s…", bad_old_string[:80])
        retry_result = await _retry_str_replace(
            message, full_report, bad_old_string, primary_llm)

        if isinstance(retry_result, str):
            updated_report = retry_result
        else:
            # Both attempts failed — fall back to a descriptive error
            return (
                "I wasn't able to locate the exact passage to edit. "
                "Could you be more specific about which section you'd like me to change?"
            )

    # Persist the updated report so future refinements and exports use the latest version.
    try:
        await _save_updated_report(
            updated_report, conversation_id, user_id, db_path,
            base_created_at=base_created_at,
        )
    except EditConflictError as exc:
        logger.warning("Concurrent edit conflict for %s: %s", conversation_id[:8], exc)
        return (
            "Your edit couldn't be saved because the report was modified by another "
            "request. Please reload the report and try again."
        )
    except Exception as exc:
        logger.warning("Could not persist updated report: %s", exc)

    return f"Here is the updated analysis for **{tickers}**:\n\n{updated_report}"


async def _attempt_str_replace(
    instruction: str,
    full_report: str,
    primary_llm: Any,
    *,
    section_name: str | None = None,
    section_text: str | None = None,
) -> str | dict:
    """
    Returns the updated document string on success.
    Returns {"bad_old_string": ...} if old_string was not found.

    When section_name and section_text are provided, the LLM receives only that
    section — reducing token usage and improving edit accuracy.  The str_replace
    is still applied against full_report so surrounding content is preserved.
    """
    try:
        if section_name and section_text:
            prompt_content = _STR_REPLACE_SECTION_PROMPT.format(
                section_name=section_name,
                instruction=instruction,
                section_text=section_text,
            )
        else:
            prompt_content = _STR_REPLACE_PROMPT.format(
                instruction=instruction,
                full_report=full_report,
            )
        response = await primary_llm.ainvoke([
            SystemMessage(content=_EDITOR_SYSTEM),
            HumanMessage(content=prompt_content),
        ])
        raw = content_to_str(response.content if hasattr(response, "content") else response)
        parsed = json.loads(_extract_json(raw))
        old_string: str = parsed["old_string"]
        new_string: str = parsed["new_string"]
    except Exception as exc:
        logger.error("str_replace: LLM or parse error: %s", exc)
        raise

    updated = _flexible_str_replace(full_report, old_string, new_string)
    if updated is None:
        return {"bad_old_string": old_string}
    return updated


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

    updated = _flexible_str_replace(full_report, old_string, new_string)
    if updated is None:
        return {"bad_old_string": old_string}
    return updated


async def _load_latest_report(
    conversation_id: str, user_id: str, db_path: str
) -> dict | None:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT tickers, report_markdown, raw_data_json, analysis_json, created_at"
            " FROM reports WHERE conversation_id = ? AND user_id = ?"
            " ORDER BY created_at DESC LIMIT 1",
            (conversation_id, user_id),
        ) as cursor:
            row = await cursor.fetchone()

    if not row:
        return None
    return {
        "tickers":         row[0],
        "report_markdown": row[1],
        "raw_data":        json.loads(row[2] or "{}"),
        "analysis":        json.loads(row[3] or "{}"),
        "created_at":      row[4],   # optimistic lock token
    }


async def _save_updated_report(
    updated_markdown: str,
    conversation_id: str,
    user_id: str,
    db_path: str,
    base_created_at: float | None = None,
) -> None:
    """Insert the edited report as a new versioned row (INSERT, not UPDATE).

    If base_created_at is provided (optimistic lock token from _load_latest_report),
    verifies that no concurrent request has written a newer version before inserting.
    Raises EditConflictError when the check fails — the caller should surface a
    user-friendly message asking them to reload and reapply.
    """
    async with aiosqlite.connect(db_path) as db:
        # Optimistic lock check: verify no concurrent write occurred since load.
        if base_created_at is not None:
            async with db.execute(
                "SELECT MAX(created_at) FROM reports"
                " WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, user_id),
            ) as cur:
                row = await cur.fetchone()
            current_latest = row[0] if row else None
            if current_latest != base_created_at:
                raise EditConflictError(
                    f"Report was modified by a concurrent request "
                    f"(expected version {base_created_at}, found {current_latest}). "
                    "Please reload and reapply your edit."
                )

        # Copy metadata from the current latest row
        async with db.execute(
            "SELECT tickers, raw_data_json, analysis_json FROM reports"
            " WHERE conversation_id = ? AND user_id = ?"
            " ORDER BY created_at DESC LIMIT 1",
            (conversation_id, user_id),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return

        tickers, raw_data_json, analysis_json = row
        await db.execute(
            "INSERT INTO reports (id, conversation_id, user_id, tickers,"
            " report_markdown, raw_data_json, analysis_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()), conversation_id, user_id, tickers,
                updated_markdown, raw_data_json, analysis_json, time.time(),
            ),
        )
        await db.commit()
