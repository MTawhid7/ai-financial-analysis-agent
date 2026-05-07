# Roadmap: AI Financial Analyst → Conversational AI Agent

**Last updated:** 2026-05-07  
**Status:** Phase 2 complete — Phase 2.5 + Phase 4 (auth) next

---

## Overview

This document is the authoritative implementation roadmap for transforming the AI Financial Analyst Agent from a single-turn, stateless pipeline into a full conversational AI agent comparable in experience to ChatGPT or Claude.

**Guiding constraints:**
- All existing infrastructure (circuit breaker, injection filter, budget tracker, caching, grounding check, test suite) is preserved and extended — never replaced.
- The inner Researcher → Quant → Editor pipeline remains a black box to the chat layer.
- Free-tier API limits (Gemini Flash: 15 RPM, 1,500 RPD) govern all sequencing decisions.
- Every phase must leave the test suite fully green before the next begins.

---

## Architecture Decision: Streamlit → FastAPI + React

**Strategy: Streamlit for Phases 1–3, migrate to FastAPI + React in Phase 4.**

Streamlit 1.45.1 supports `st.chat_message` / `st.chat_input` — sufficient for validating the conversational AI behaviour. It cannot, however, deliver the production-quality UX the roadmap requires: true SSE streaming, a file workspace sidebar, bidirectional UI updates, or full typography control. The migration happens once the AI logic is proven, not before.

---

## Recommended Additions (Beyond Original Scope)

Capabilities not in the original request that significantly strengthen the product:

| # | Feature | Value |
|---|---------|-------|
| 1 | **Portfolio mode** | Analyse a weighted set of tickers as one portfolio (combined P/E, CAGR, diversification score) |
| 2 | **Comparison mode** | Side-by-side table for "AAPL vs MSFT" — one compact report instead of two full runs |
| 3 | **FRED macro context** | Federal Reserve Economic Data (free API) — inject interest rates, CPI into analysis |
| 4 | **Excel export with live formulas** | CAGR formula in cells, not just pre-computed values |
| 5 | **Provenance UI** | "Show Source" on every number → opens the exact tool call that produced it (already in `run_artifacts.json`) |
| 6 | **Session authentication** | `DEMO_PASSWORD` env-var HTTP Basic Auth before Phase 4 deploys publicly |
| 7 | **Watchlist + alerts** | Save tickers; background scheduler re-runs weekly and reports changes |

---

## Phase Summary

| Phase | Name | Status | Complexity | New Packages |
|-------|------|--------|-----------|--------------|
| **1** | Conversational Core | ✅ **Complete** | Medium | None |
| **2** | Memory System | ✅ **Complete** | Medium | None |
| **2.5** | Memory Bug Fix + Conversation Persistence | Planned | Low | None |
| **3** | Streaming + Intervention | Planned | High | None |
| **4A** | FastAPI Backend + Google Auth | Planned | High | fastapi, uvicorn, google-auth, python-jose |
| **4B** | React Frontend + Conversation UI | Planned | High | (frontend npm deps) |
| **5** | Multimodal | Planned | Medium-High | plotly, pandas, weasyprint, pdfplumber, python-docx, openpyxl |
| **6** | Refinement + Comparison | Planned | Medium | None |
| **7** | Polish + Provenance + Vector Memory | Planned | Medium | sentence-transformers |

See [MEMORY_ARCHITECTURE.md](MEMORY_ARCHITECTURE.md) for the full design of Phases 2.5, 4A, 4B, and the vector memory strategy.

---

## Phase 1 — Conversational Core ✅

**Delivered:** 2026-05-06 · Commit `14d426f`

**Goal:** Accept free-form natural language. Classify intent. Route to appropriate handler. Reject off-topic requests politely.

### What was built

A `ConversationalAgent` that sits above the existing pipeline and handles all user messages before deciding whether to invoke the analysis pipeline at all.

**Intent taxonomy (classified by Flash-Lite):**

| Intent | Behaviour |
|--------|-----------|
| `financial_analysis` | Invokes Researcher → Quant → Editor pipeline; returns full report |
| `financial_question` | Answered directly by primary LLM with conversation history context |
| `off_topic` | Politely declined with no pipeline or LLM cost |
| `clarification_needed` | Prompts the user for more information |

### New files

| File | Description |
|------|-------------|
| `ai_financial_analyst/core/conversation_state.py` | `ConversationState` TypedDict (session_id, messages, intent, tickers). Kept entirely separate from `AgentState`. |
| `ai_financial_analyst/agents/intent_classifier.py` | Flash-Lite JSON classifier with regex fallback for ticker extraction. Fails safely to `financial_question` on any error. |
| `ai_financial_analyst/agents/conversational_agent.py` | `ConversationalAgent` — session-scoped, async `process_message()`. Passes `step_callback` through to the pipeline for live UI updates. |
| `ui/chat_app.py` | Streamlit chat UI: `st.chat_input` + `st.chat_message`, live TAO stream, debug downloads in sidebar. |
| `tests/unit/test_intent_classifier.py` | 10 unit tests covering intent routing, fallbacks, and JSON fence stripping. |
| `tests/integration/test_conversational_agent.py` | 11 integration tests covering routing, state management, history injection, and error handling. |

### Modified files

| File | Change |
|------|--------|
| `ai_financial_analyst/agents/orchestrator.py` | Added `run_pipeline_from_tool()` sync wrapper for future `@tool` use in Phase 4. |

### Verification

```bash
streamlit run ui/chat_app.py
# "Analyse AAPL"           → pipeline runs, full report rendered
# "What is a P/E ratio?"   → direct LLM answer, no pipeline
# "What's the weather?"    → polite rejection
pytest tests/unit/ tests/integration/ tests/adversarial/ -v
# → 84/84 passing
```

---

## Phase 2 — Memory System ✅

**Delivered:** 2026-05-07

**Goal:** The agent remembers within a session and across sessions.

### What was built

**Short-term memory** (`memory/short_term.py`) — stateless utility that selects the most recent messages fitting a 3,000-token budget (estimated at `len // 4`). Used in `_handle_financial_question` to manage the LLM context window.

**Long-term memory** (`memory/long_term.py`) — SQLite at `.memory/memory.db` (gitignored) using `aiosqlite` (already a dependency). Two tables:
- `preferences(key, value, updated_at)` — user-stated preferences, upserted per key
- `analysis_summaries(session_id, tickers, summary_text, run_id, created_at)` — one paragraph per completed pipeline run, retrieved by LIKE search

**MemoryManager** (`memory/memory_manager.py`) — facade providing:
- `build_memory_context(messages, query) → str` — ≤500-token string injected into the system prompt, combining known preferences and relevant past analyses
- `maybe_extract_preferences(message)` — regex pre-filter (avoids LLM call on every message) + Flash-Lite extraction; only fires when explicit preference signals are detected
- `maybe_save_analysis_summary(...)` — Flash-Lite one-paragraph summary saved after each pipeline run

### New files

| File | Description |
|------|-------------|
| `ai_financial_analyst/memory/__init__.py` | Module exports |
| `ai_financial_analyst/memory/short_term.py` | `ShortTermMemory` — token-budget context window |
| `ai_financial_analyst/memory/long_term.py` | `LongTermMemory` — SQLite preferences + summaries |
| `ai_financial_analyst/memory/memory_manager.py` | `MemoryManager` facade + UI accessors |
| `tests/unit/test_short_term_memory.py` | 7 unit tests |
| `tests/unit/test_long_term_memory.py` | 14 unit tests |
| `tests/unit/test_memory_manager.py` | 14 unit tests |

### Modified files

| File | Change |
|------|--------|
| `agents/conversational_agent.py` | Init `MemoryManager`; inject memory context into system prompt; save analysis summary after each pipeline run; use `ShortTermMemory` for context window |
| `ui/chat_app.py` | Memory sidebar panel: preferences list, past analysis count, "Clear memory" button |
| `.gitignore` | Added `.memory/` |

### Verification

```bash
streamlit run ui/chat_app.py
# "I prefer conservative analysis"             → sidebar shows investment_style: conservative
# "Analyse AAPL"                               → analysis runs; summary saved to .memory/memory.db
# "What did you find about AAPL last time?"    → agent surfaces stored summary in response
# Click "Clear memory"                         → sidebar shows 0 analyses, no preferences
pytest tests/unit/ tests/integration/ tests/adversarial/ -v
# → 133/133 passing
```

---

## Phase 3 — Real-Time Streaming and User Intervention

**Goal:** Show tool calls inside chat bubbles as they happen. Allow the user to stop or redirect the pipeline mid-execution.

### Architecture

**`StreamBus`** — an `asyncio.Queue` the pipeline writes typed events to. Event types: `tool_start`, `tool_result`, `agent_transition`, `thinking`, `complete`, `error`. The UI consumes these via the `step_callback` hook that already exists in every agent.

**`InterruptSignal`** — a mutable dataclass stored in `st.session_state`. Fields: `should_stop: bool`, `modification: str | None`. The pipeline reads it at every `step_callback` invocation. If `should_stop` is set, it raises `UserInterruptError`, which `_safe_node` catches and converts to a partial report.

### New files

| File | Description |
|------|-------------|
| `ai_financial_analyst/core/stream_bus.py` | `StreamBus` with typed event dataclasses |
| `ai_financial_analyst/core/interrupt_signal.py` | `InterruptSignal` dataclass |

### Modified files

| File | Change |
|------|--------|
| `agents/orchestrator.py` | `_safe_node` checks `InterruptSignal.should_stop` after each node |
| `agents/researcher.py`, `quant_analyst.py`, `editor.py` | Replace bare `step_callback` calls with `stream_bus.emit(ToolStartEvent / ToolResultEvent)` |
| `ui/chat_app.py` | Render `StreamBus` events inline in chat bubbles via `st.empty()` placeholders; "Stop Analysis" button; LLM reasoning in collapsible `st.expander` (hidden by default) |

### Key technical constraint

Streamlit's full-rerun model requires careful use of `st.empty()` with `with` blocks persisted via `session_state`. The `step_callback` is called synchronously from within `asyncio.run()`, so `st.empty()` updates render in real time without a WebSocket.

### Verification

```
"Analyse AAPL" → tool calls appear step-by-step inside the chat bubble
Click "Stop Analysis" mid-run → partial report within 3 seconds, no crash
```

---

## Phase 4 — Task Planning, Clarification + FastAPI/React Migration

### Part A — Task Planning and Self-Correction

**Goal:** For complex or ambiguous queries, show a plan and ask for confirmation. Ask one clarifying question at a time. Retry partial failures before halting.

**`PlannerAgent`** (`agents/planner.py`) — produces a `TaskPlan` Pydantic model:

```python
class TaskPlan(BaseModel):
    steps: list[PlanStep]
    estimated_api_calls: int
    clarifications_needed: list[str]
```

Uses Flash-Lite (budget-preserving). Pre-checks: if `estimated_api_calls > remaining_budget * 0.5`, warns the user before starting.

**Planning gate** in `ConversationalAgent`:
- Simple query (single ticker, clear intent) → skip to execution
- Complex query (3+ tickers, ambiguous, multi-intent) → generate plan → display → await confirmation

**`ClarificationHandler`** (`agents/clarification_handler.py`) — asks one question at a time, stores the pending plan in `ConversationState`.

**`_safe_node` improvement** — on `PartialStateError`, attempts one retry with reduced `max_iterations=3` before halting.

---

### Part B — FastAPI + React Migration

**Backend structure:**

```
backend/
  main.py
  routers/
    chat.py          POST /chat, SSE /stream/{session_id}
    files.py         GET/POST/DELETE /files + version history
    memory.py        GET/DELETE /memory, PATCH /memory/preferences
    feedback.py      POST /feedback
  core/
    session_manager.py   LRU cache of ConversationalAgent instances (30-min TTL)
```

**Frontend structure:**

```
frontend/
  src/components/
    ChatInterface.tsx
    ChatBubble.tsx           user / assistant / tool-call variants
    ToolCallBubble.tsx       streaming tool events
    PlanConfirmation.tsx     numbered checklist + Proceed/Modify buttons
    WorkspaceSidebar.tsx     file browser with rename/delete/download/versions
    MemoryPanel.tsx
  src/styles/
    design_tokens.css        Inter font, tabular-nums, dark/light mode
```

**Key decisions:**
- **SSE, not WebSocket** — unidirectional server-to-client stream; simpler and sufficient for tool event streaming
- **Legacy entrypoints preserved** — `ui/app.py` and `ui/chat_app.py` continue to work; zero test regression
- **Single-process sessions** — `session_manager.py` LRU cache; fine for personal/demo use
- **`DEMO_PASSWORD` auth** — env-var HTTP Basic Auth added before any public deploy

**New packages:** `fastapi>=0.115`, `uvicorn[standard]>=0.30`, `python-multipart>=0.0.9`

### Verification

```bash
uvicorn backend.main:app --reload
npm run dev   # frontend
# Chat input → FastAPI → SSE → tool events stream to browser in real time
# Task plan displayed for multi-ticker queries
# Workspace file list shows created reports
```

---

## Phase 5 — Multimodal: File Upload, Charts, Export

**Goal:** Accept user-provided files (CSV, PDF). Generate interactive Plotly charts. Export reports to PDF, Word, and Excel.

### New tools

| Tool | Description |
|------|-------------|
| `tools/file_parser.py` | CSV → fixed-schema pandas summary JSON; PDF → pdfplumber text → Flash-Lite summary. Formula injection check on CSV cell values (`=`, `+HYPERLINK`). |
| `tools/chart_generator.py` | Returns serialised Plotly JSON (client renders interactively). Types: `price_history`, `pe_comparison`, `metric_bar`. No image generation — preserves the no-REPL security model. |
| `tools/pdf_exporter.py` | Markdown → HTML (markdown-it-py) → PDF (weasyprint). Saves to workspace. |
| `tools/docx_exporter.py` | Markdown → Word (python-docx) with proper heading styles. |
| `tools/xlsx_exporter.py` | `raw_data` + `analysis` → structured Excel workbook. CAGR formula written as a live cell formula, not a pre-computed value. |

**New packages:** `pdfplumber>=0.11`, `weasyprint>=62.0`, `plotly>=5.24`, `pandas>=2.2`, `markdown-it-py>=3.0`, `python-docx>=1.1`, `openpyxl>=3.1`

**Security note:** CSV data is never exposed to arbitrary pandas operations. The file parser produces only a fixed-schema JSON summary — the "no Python REPL" invariant is preserved.

### Verification

```
Upload a CSV                            → agent acknowledges columns and shape
"Show AAPL price history as a chart"   → Plotly chart rendered inline in chat
"Export as PDF"                         → PDF downloads from workspace
```

---

## Phase 6 — Iterative Refinement, Comparison Mode, and Feedback

**Goal:** Allow users to refine results. Add comparison mode. Collect thumbs-up/down ratings.

### Refinement handler (`agents/refinement_handler.py`)

Detects "modify the previous result" intent and routes to the minimum necessary re-execution:
- Structural modification (*"add a risks section"*) → re-run Editor node only
- Numerical modification (*"assume 15% revenue growth"*) → re-run calculator + SOP chain only

Saves the previous version as `report_v{n}.md` in the workspace before overwriting.

### Comparison agent (`agents/comparison_agent.py`)

*"AAPL vs MSFT"* → runs the pipeline per ticker, then generates a compact side-by-side comparison table (P/E, CAGR, sector premium, bull/bear). Returns one comparison report instead of two full reports.

### Feedback store (`memory/feedback_store.py`)

SQLite table: `feedback(message_id, session_id, rating, flag_reason, timestamp)`. The `MemoryManager` surfaces recent ratings to calibrate response verbosity.

**UI change:** 👍/👎 and "Flag" buttons added to each `ChatBubble`.

### Key technical challenge

Partial pipeline re-execution — calling only the Editor node or only the SOP LLM chain — requires exposing those sub-steps as independently callable units without breaking the full pipeline flow.

### Verification

```
"Make the bear case more pessimistic"   → editor re-runs, report_v2.md saved in workspace
"Compare AAPL vs MSFT"                  → side-by-side comparison table rendered
👍 on a response                         → rating written to feedback.db
```

---

## Phase 7 — UI Polish, Provenance, and Hardening

**Goal:** Final visual pass. Surface the existing provenance data (already captured in `run_artifacts.json`) in the UI. Error boundaries. Keyboard shortcuts.

### Design system

- **Fonts:** Inter (body text) + JetBrains Mono (all numbers/code)
- **Tabular figures:** CSS `font-variant-numeric: tabular-nums` on every number column so financial tables align correctly
- **Themes:** `prefers-color-scheme` dark/light mode via CSS custom properties in `design_tokens.css`

### Provenance panel

Every number in the report gets a "Show Source" button. Clicking it opens a side panel that shows the exact tool call from `run_artifacts.json` that produced that figure — making the AI's reasoning auditable without leaving the chat. This data is already collected; Phase 7 surfaces it.

### Memory management UI (`MemoryPanel.tsx`)

- Past analyses table with per-row delete and export
- Preferences editor: displays stored preferences in natural language ("You told me: I prefer conservative picks"), not raw JSON
- "Clear All Memory" button with a confirmation modal

### Error boundaries

- React `ErrorBoundary` components catch rendering errors and show a fallback UI instead of a blank screen
- FastAPI returns structured errors: `{"error_type": ..., "detail": ..., "suggestion": ...}` — the `suggestion` field gives the user actionable guidance (e.g., *"Gemini rate limit hit — wait 60 seconds and retry"*)

### Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+K` | Focus chat input |
| `Ctrl+/` | Toggle memory panel |
| `Esc` | Stop analysis |

### Export menu (`ExportMenu.tsx`)

Download report as: Markdown (existing) · PDF · Word · Excel (with live formulas)

### Verification

```
Every number in the report → "Show Source" button present and functional
Memory panel → preferences shown as natural language sentences
Dark mode → app renders correctly via prefers-color-scheme
All keyboard shortcuts functional
Error boundary → renders fallback on component crash, no blank screen
```

---

## Files That Must Never Break

These are the invariant files that every phase must leave intact and passing:

| File | Why |
|------|-----|
| `ai_financial_analyst/core/sanitizer.py` | Injection filter + canary token — security baseline |
| `ai_financial_analyst/core/llm.py` | Circuit breaker + budget callback — rate limit resilience |
| `ai_financial_analyst/core/state.py` | `AgentState` contract — changing this breaks all three pipeline agents |
| `tests/` | All tests must remain green after every phase |

---

## Test Count by Phase

| After Phase | Tests |
|-------------|-------|
| Baseline | 57 |
| Phase 1 ✅ | **98** |
| Phase 2 ✅ | **138** |
| Phase 2.5 (target) | ~155 |
| Phase 3 (target) | ~170 |
| Phase 4 (target) | ~200+ |
