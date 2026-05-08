# Roadmap: AI Financial Analyst Agent

**Last updated:** 2026-05-08  
**Status:** Phases 1–7 complete

---

## Architecture

```
Browser (React 19 + Vite, port 5173)
  ↕ @react-oauth/google — Google Sign-In popup
  ↕ fetch() credentials:include — REST (JSON)
  ↕ EventSource credentials:include — SSE streaming
FastAPI 0.115 (port 8000, uvicorn)
  ↕ session_manager: user_id → ConversationalAgent (LRU, 30-min TTL)
ConversationalAgent  ← Flash-Lite intent classifier (5 intents)
  ↓ financial_analysis      ↓ memory_query      ↓ financial_question   ↓ off_topic
run_pipeline()        search summaries     primary LLM answer    rejection
  ↓
Researcher → Quant Analyst → Editor → Markdown report
```

**Guiding constraints:**
- All existing Python infrastructure (circuit breaker, injection filter, budget tracker, caching, grounding check) is preserved and extended — never replaced.
- The inner Researcher → Quant → Editor pipeline is a black box to the chat layer.
- Free-tier Gemini limits (15 RPM, 1,500 RPD) govern all sequencing decisions.
- Every phase must leave the test suite and `npm run build` fully green before the next begins.

---

## Phase Summary

| Phase | Name | Status | Key Deliverables |
|-------|------|--------|-----------------|
| **1** | Conversational Core | ✅ Complete | Chat UI, 5-intent routing, pipeline-as-tool |
| **2** | Memory System | ✅ Complete | SQLite preferences + analysis summaries, memory-aware system prompt |
| **2.5** | Memory Bug Fix + Conversation Persistence | ✅ Complete | `memory_query` intent, conversation history in SQLite, sidebar |
| **3** | Streamlit Streaming | Dropped | Streaming implemented properly in Phase 4B |
| **4A** | FastAPI Backend + Google Auth | ✅ Complete | Google OAuth, JWT httpOnly cookie, DB migration, SSE endpoint |
| **4B** | React + Vite Frontend | ✅ Complete | Login page, chat interface, markdown rendering, SSE streaming, memory panel |
| **5** | Multimodal | ✅ Complete | CSV/PDF upload, Plotly charts, PDF/Word/Excel export |
| **6** | Refinement + Comparison | ✅ Complete | Comparison intent, result refinement, 👍/👎 feedback |
| **7** | Polish + Provenance | ✅ Complete | Typography system, provenance panel, memory management UI |

---

## Completed Phases

### Phase 1 — Conversational Core
**Commit:** `14d426f`

Five-intent routing (Flash-Lite classifier):

| Intent | Handler |
|--------|---------|
| `financial_analysis` | Full Researcher → Quant → Editor pipeline |
| `financial_question` | Primary LLM with conversation history context |
| `memory_query` | Returns stored analysis summaries from SQLite |
| `off_topic` | Polite rejection template (zero API cost) |
| `clarification_needed` | Asks for more context |

**New:** `core/conversation_state.py`, `agents/intent_classifier.py`, `agents/conversational_agent.py`, `ui/chat_app.py` (archived)

---

### Phase 2 — Memory System
**Commit:** `726475c`

- `LongTermMemory`: SQLite via aiosqlite — preferences + analysis summaries
- `ShortTermMemory`: token-budget context window (3,000-token cap, no DB)
- `MemoryManager`: `build_memory_context()` injects ≤500-token string into system prompt; `maybe_extract_preferences()` (regex-gated Flash-Lite); `maybe_save_analysis_summary()` (Flash-Lite) after each pipeline run

---

### Phase 2.5 — Memory Bug Fix + Conversation Persistence
**Commit:** `0d900f0`

**Bug fixed:** "What did we find about AAPL earlier?" was classified as `financial_analysis` (because AAPL is present) and re-ran the pipeline. Root cause: no `memory_query` intent existed. Fixed with a fifth intent and a dedicated handler that searches stored summaries.

**Conversation persistence:** `conversations` + `messages` tables in SQLite. Every turn saved in real time. Auto-resumes the most recent conversation on app restart.

---

### Phase 4A — FastAPI Backend + Google Auth
**Commit:** `d547316` (partial) · Bugs fixed in `8bcf2af` area

- Google ID token validation (`google-auth`) → JWT signed with `FASTAPI_JWT_SECRET` → httpOnly `fin_session` cookie (30-day expiry)
- DB migration on startup: adds `user_id` column to all tables backward-compatibly
- `session_manager`: user-scoped `ConversationalAgent` LRU cache (30-min TTL)
- SSE pattern: `POST /chat/{conv_id}` starts pipeline → returns `event_id` → `GET /stream/{event_id}` delivers tool-step events + final response via `EventSource`

---

### Phase 4B — React + Vite Frontend
**Commit:** `d547316` (partial) · Post-deploy fixes in subsequent commits

**Stack:** React 19 · Vite 6 · TypeScript · Tailwind CSS · TanStack Router/Query · `@react-oauth/google` · `react-markdown` + `remark-gfm`

**Components:**
- `LoginPage` — Google Sign-In button → POST /auth/google
- `ChatInterface` — loads history on mount, manages streaming state
- `ChatBubble` — user (violet) / assistant (with inline tool steps) / markdown-rendered reports
- `ConversationList` — TanStack Query, delete-on-hover, time labels
- `MemoryPanel` — collapsible; shows preferences + past analyses + clear button
- `useStreamingChat` — POST → `EventSource` → `onStep`/`onComplete`/`onError`

---

## Running Locally

```bash
# 1. Fill in .env (see .env.example for all required keys)
conda activate fin-agent
pip install -e ".[server]"

# Terminal 1 — backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
cp .env.local.example .env.local   # add VITE_GOOGLE_CLIENT_ID
npm run dev
# → http://localhost:5173
```

**Required env vars:**

| Variable | Purpose |
|---|---|
| `GOOGLE_API_KEY` | Gemini AI (aistudio.google.com) |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret |
| `FASTAPI_JWT_SECRET` | JWT signing — generate with `python -c "import secrets; print(secrets.token_hex(32))"` |
| `TAVILY_API_KEY` | Web search (app.tavily.com) |
| `LANGSMITH_API_KEY` | Tracing (smith.langchain.com) |

---

## Test Count

| After Phase | Python Tests | Frontend |
|-------------|-------------|---------|
| Baseline | 57 | — |
| Phase 1 ✅ | 98 | — |
| Phase 2 ✅ | 138 | — |
| Phase 2.5 ✅ | 156 | — |
| Phase 4A + 4B ✅ | **156** | `npm run build` ✅ (zero TS errors) |
| Phase 5 (target) | ~180 | — |

---

## Planned Phases

### Phase 5 — Multimodal
CSV/PDF upload → structured summary · Plotly charts from analysis data · PDF/Word/Excel export with live formulas. Key constraint: CSV is parsed to a fixed-schema JSON summary only (no arbitrary pandas operations — preserves the no-REPL invariant).

### Phase 6 — Refinement + Comparison
Partial pipeline re-execution (Editor-only for structural edits, SOP-chain-only for numerical edits). Side-by-side "AAPL vs MSFT" comparison table. Thumbs-up/down feedback stored in SQLite.

### Phase 7 — Polish + Vector Memory
Typography system (Inter body, JetBrains Mono numbers). "Show Source" provenance button on every number in the report (data already in `run_artifacts.json`). Upgrade memory retrieval from LIKE search to `sentence-transformers` vector similarity when `count_summaries() > 200`.
