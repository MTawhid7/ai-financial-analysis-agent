# Roadmap: AI Financial Analyst Agent

**Last updated:** 2026-05-08  
**Status:** All phases complete (1 → 7)

---

## Architecture

```
Browser (React 19 + Vite, port 5173)
  ↕ @react-oauth/google — Google Sign-In popup
  ↕ fetch() credentials:include — REST (JSON)
  ↕ EventSource credentials:include — SSE streaming
FastAPI 0.115 (port 8000, uvicorn)
  ↕ session_manager: user_id → ConversationalAgent (LRU, 30-min TTL)
ConversationalAgent  ← Flash-Lite intent classifier (7 intents)
  ↓ financial_analysis  ↓ comparison  ↓ refinement  ↓ memory_query  ↓ financial_question  ↓ off_topic
run_pipeline()       comparison_   refinement_   search           primary LLM       rejection
                     agent         handler       summaries        + history
  ↓
Researcher → Quant Analyst → Editor → Report + Charts + run_artifacts.json
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
| **5** | Multimodal | ✅ Complete | CSV/PDF upload, Plotly charts, PDF/Word/Excel export with live formulas |
| **6** | Refinement + Comparison | ✅ Complete | `comparison` + `refinement` intents, 👍/👎 feedback, 7-intent taxonomy |
| **7** | Polish + Provenance | ✅ Complete | Inter/JetBrains Mono typography, provenance panel, memory management UI |

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

**New:** `core/conversation_state.py`, `agents/intent_classifier.py`, `agents/conversational_agent.py`

---

### Phase 2 — Memory System
**Commit:** `726475c`

- `LongTermMemory`: SQLite via aiosqlite — preferences + analysis summaries
- `ShortTermMemory`: token-budget context window (3,000-token cap, no DB)
- `MemoryManager`: `build_memory_context()` injects ≤500-token string into system prompt; `maybe_extract_preferences()` (regex-gated Flash-Lite); `maybe_save_analysis_summary()` (Flash-Lite)

---

### Phase 2.5 — Memory Bug Fix + Conversation Persistence
**Commit:** `0d900f0`

**Bug fixed:** "What did we find about AAPL earlier?" was classified as `financial_analysis` (AAPL present) and re-ran the pipeline. Fixed with a dedicated `memory_query` intent and handler.

**Conversation persistence:** `conversations` + `messages` SQLite tables. Every turn saved in real time. Auto-resumes the most recent conversation on app restart.

---

### Phase 4A — FastAPI Backend + Google Auth
**Commit:** `d547316`

- Google ID token validation (`google-auth`) → JWT → httpOnly `fin_session` cookie (30-day expiry)
- Idempotent DB migration on startup: adds `user_id` column to all tables
- `session_manager`: user-scoped `ConversationalAgent` LRU cache (30-min TTL)
- SSE pattern: `POST /chat/{conv_id}` → `event_id` → `GET /stream/{event_id}` via `EventSource`

---

### Phase 4B — React + Vite Frontend
**Commit:** `d547316` · post-deploy fixes in `712bfdb`

**Stack:** React 19 · Vite 6 · TypeScript · Tailwind CSS · TanStack Router/Query · `@react-oauth/google` · `react-markdown`

**Components:** `LoginPage`, `ChatInterface`, `ChatBubble` (inline tool steps + markdown), `ConversationList`, `MemoryPanel`, `useStreamingChat`

---

### Phase 5 — Multimodal
**Commit:** `232892a`

**Python tools:**
- `chart_generator.py` — Plotly JSON: 1-year price line with 52w bands, P/E bar, key financials bar
- `file_parser.py` — CSV (pandas fixed-schema + formula injection scrub); PDF (pdfplumber + Flash-Lite summary)
- `pdf_exporter.py` — weasyprint PDF; `docx_exporter.py` — python-docx; `xlsx_exporter.py` — openpyxl with live CAGR formula cells

**FastAPI:** `POST /files/upload`, `POST /export/{pdf,docx,xlsx}/{report_id}`, `GET /export/available`

**Frontend:** `PlotlyChart` (lazy-loaded, code-split), `FileUploadZone`, `ExportMenu`

---

### Phase 6 — Refinement + Comparison
**Commit:** `52b2214`

**Two new intents added (7 total):**
- `comparison` — triggered by "vs", "versus", "compare X and Y"; runs multi-ticker pipeline + Flash generates side-by-side Markdown table
- `refinement` — triggered by "make it more", "redo with X%", "add section"; retrieves stored report from `reports` table, modifies with Flash LLM — no full pipeline re-run

**Feedback:** SQLite `feedback` table; `POST /feedback`; 👍/👎 buttons on every assistant message; ratings persist and reload across sessions

---

### Phase 7 — Polish + Provenance
**Commit:** `52b2214`

**Typography system:**
- Inter (body) + JetBrains Mono (numbers/code) via Google Fonts
- `font-variant-numeric: tabular-nums` on all financial tables — numeric columns right-aligned in monospace
- Dark-theme prose overrides: styled headings, tables, code blocks, blockquotes

**Provenance panel:**
- "View Sources" button on every analysis report (hover to reveal)
- Shows: metric → formatted value → source tool (Calculator / Sector Benchmarks / Yahoo Finance) → observation step
- Data from `citations` dict stored in `reports` table — no extra computation

**Memory management UI:**
- Preferences shown as natural-language sentences ("You prefer conservative investment style")
- Past analysis cards: ticker tag, date, excerpt
- Confirmation modal before clearing memory

---

## Running Locally

```bash
# 1. Fill in .env (see .env.example)
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
| `GOOGLE_API_KEY` | Gemini AI |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret |
| `FASTAPI_JWT_SECRET` | JWT signing (`python -c "import secrets; print(secrets.token_hex(32))"`) |
| `TAVILY_API_KEY` | Web search |
| `LANGSMITH_API_KEY` | Tracing |

---

## Test Count

| After Phase | Python Tests | Frontend |
|-------------|-------------|---------|
| Baseline | 57 | — |
| Phase 1 ✅ | 98 | — |
| Phase 2 ✅ | 138 | — |
| Phase 2.5 ✅ | 156 | — |
| Phase 4A + 4B ✅ | 156 | `npm run build` ✅ |
| Phase 5 ✅ | 156 | ✅ |
| Phase 6 + 7 ✅ | **156** | ✅ |
