# AI Financial Analyst Agent — CLAUDE.md

## Project Overview

Conversational AI Financial Analyst Agent. Uses a ReAct + Multi-Agent architecture with LangGraph. Natural language input, Google OAuth authentication, per-user persistent memory, real-time SSE streaming, interactive Plotly charts, multi-format export, and result comparison/refinement — delivered through a FastAPI + React stack.

**This is not a production system.** Portfolio showcase of agentic AI engineering patterns.

---

## Environment Setup

```bash
conda activate fin-agent
pip install -e ".[server]"   # AI + FastAPI server deps
cp .env.example .env         # fill in all required keys (see below)
```

### Required API Keys

| Variable | Service | Where to get |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini AI | aistudio.google.com/apikey |
| `GOOGLE_CLIENT_ID` | Google OAuth | console.cloud.google.com/apis/credentials |
| `GOOGLE_CLIENT_SECRET` | Google OAuth | same as above |
| `FASTAPI_JWT_SECRET` | JWT signing | `python -c "import secrets; print(secrets.token_hex(32))"` |
| `TAVILY_API_KEY` | Web search | app.tavily.com |
| `LANGSMITH_API_KEY` | Tracing | smith.langchain.com |

Note: LangSmith vars changed in v0.8 — use `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true`.

---

## Running the Project

```bash
# Terminal 1 — FastAPI backend
conda activate fin-agent
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — React frontend
cd frontend
cp .env.local.example .env.local   # add VITE_GOOGLE_CLIENT_ID
npm run dev
# Open http://localhost:5173
```

Google OAuth setup: create an OAuth 2.0 Client (Web application) in Google Cloud Console with Authorised JavaScript origin `http://localhost:5173`.

---

## Running Tests

```bash
pytest tests/unit/          # fast, no API calls — run first
pytest tests/integration/   # agent logic with mocked LLM/tools
pytest tests/adversarial/   # security: prompt injection payload detection

# Full Python suite with coverage
pytest --cov=ai_financial_analyst --cov-report=term-missing

# Frontend build check (must pass with zero TS errors)
cd frontend && npm run build
```

Current status: **152/152** Python tests passing + frontend build clean.

---

## Architecture

```
React 19 + Vite (port 5173)
  ↕ Google OAuth popup (@react-oauth/google)
  ↕ fetch credentials:include — REST
  ↕ EventSource credentials:include — SSE
FastAPI 0.115 (port 8000)
  ↕ session_manager: user_id → ConversationalAgent (LRU, 30-min TTL)
ConversationalAgent
  ↕ Manager LLM (Flash + tool-use / function-calling)
      tools: run_financial_analysis, compare_stocks, recall_past_analysis,
             edit_report_section, answer_finance_question, generate_chart,
             search_documents, get_document_page,
             reject_request, ask_clarification
  ↓ run_financial_analysis / compare_stocks
Researcher → Quant Analyst → Editor → Report + Charts
```

| Component | File | Responsibilities |
|---|---|---|
| FastAPI app | `backend/main.py` | CORS, lifespan DB migration, router registration |
| Auth | `backend/routers/auth.py` | Google ID token → JWT httpOnly cookie |
| Chat + SSE | `backend/routers/chat.py` | POST /chat → event_id; GET /stream → EventSource; charts + report save |
| Files + Export | `backend/routers/files.py` | POST /files/upload (8 formats); POST /export/{pdf,docx,xlsx}; background PageIndex indexing |
| Admin | `backend/routers/admin.py` | POST/GET/PATCH/DELETE /admin/documents — system document management (ADMIN_USER_IDS) |
| Feedback | `backend/routers/feedback.py` | POST /feedback; GET /feedback/stats (stored, not yet surfaced in UI) |
| Session manager | `backend/core/session_manager.py` | user_id → ConversationalAgent LRU cache |
| ConversationalAgent | `agents/conversational_agent.py` | Delegates to Manager; preference extraction; memory summary saving |
| Manager LLM | `agents/manager.py` | Tool-use orchestrator replacing hardcoded intent classifier |
| ComparisonAgent | `agents/comparison_agent.py` | Multi-ticker pipeline + Flash comparison table |
| RefinementHandler | `agents/refinement_handler.py` | str_replace surgical report editing |
| Researcher | `agents/researcher.py` | yfinance + Tavily; max 5 iterations/ticker |
| Quant Analyst | `agents/quant_analyst.py` | CAGR, P/E vs benchmark, bull/bear cases |
| Editor | `agents/editor.py` | SOP rubric, grounding check, disclaimer |
| Orchestrator | `agents/orchestrator.py` | LangGraph StateGraph + SQLite checkpointing |

---

## Key Files

| Path | Purpose |
|---|---|
| `backend/main.py` | FastAPI entry point |
| `backend/core/database.py` | Idempotent schema migration (runs on startup) |
| `backend/core/auth.py` | JWT + Google ID token validation |
| `backend/core/event_store.py` | event_id → asyncio.Queue registry for SSE |
| `frontend/src/hooks/useStreamingChat.ts` | POST /chat → EventSource /stream |
| `frontend/src/lib/api.ts` | Typed fetch wrappers for all FastAPI endpoints |
| `frontend/src/components/chat/CitationRenderer.tsx` | (Source: xxx) → numbered [N] badges + popovers + References section |
| `frontend/src/components/PlotlyChart.tsx` | Lazy-loaded Plotly chart renderer |
| `ai_financial_analyst/agents/manager.py` | LangChain bind_tools orchestrator |
| `ai_financial_analyst/core/state.py` | `AgentState` TypedDict — inner pipeline contract |
| `ai_financial_analyst/core/conversation_state.py` | `ConversationState` TypedDict — chat layer |
| `ai_financial_analyst/core/llm.py` | Gemini client: retry + circuit breaker + Flash-Lite fallback |
| `ai_financial_analyst/core/sanitizer.py` | Injection filter (full-content rejection) + canary token |
| `ai_financial_analyst/memory/long_term.py` | SQLAlchemy/Postgres: preferences, summaries, conversations, messages, reports, feedback (user-scoped) |
| `ai_financial_analyst/memory/memory_manager.py` | Memory facade: context injection, preference extraction, summary saving |
| `ai_financial_analyst/tools/calculator.py` | AST-validated numexpr evaluator (no REPL) |
| `ai_financial_analyst/tools/chart_generator.py` | Shim → `ai_financial_analyst/charts/` (13 chart types) |
| `ai_financial_analyst/tools/file_parser.py` | Shim → `ai_financial_analyst/parsers/` (7 format parsers) |
| `ai_financial_analyst/tools/xlsx_exporter.py` | Excel workbook with live CAGR formula cells |
| `ai_financial_analyst/pageindex/__init__.py` | PageIndex public API: index_document, search_documents, get_page |
| `ai_financial_analyst/pageindex/pipeline.py` | Ingest pipeline: extract → summarise → embed → pgvector |
| `ai_financial_analyst/pageindex/retriever.py` | Hybrid search: pgvector ANN + Postgres FTS + RRF; SQLite fallback |
| `ai_financial_analyst/pageindex/embedder.py` | Gemini text-embedding-004 (768-dim) with ResultCache |
| `ai_financial_analyst/pageindex/ocr.py` | Scanned PDF detection + pytesseract OCR |
| `ai_financial_analyst/parsers/_page_extractor.py` | RawPage dataclass + per-format structured extraction |
| `backend/core/models.py` | ORM: adds Document, DocumentPage, PageLink (pgvector-aware) |
| `backend/routers/admin.py` | Admin endpoints for system documents (ADMIN_USER_IDS env var) |

---

## Critical Design Decisions (Do Not Change Without Review)

### No Python REPL
`CalculatorTool` uses `numexpr` with a three-level AST guard. CSV/XLSX files are parsed to a fixed-schema JSON summary only — no arbitrary pandas operations on user data.

### Full-Content Injection Rejection
`ContentSanitizer._regex_filter()` rejects the **entire content block** on any injection pattern match. CSV cell values starting with `=`, `+`, `-`, `@` are also scrubbed.

### Sequential Agent Execution
Agents run one at a time. Concurrent execution saturates the free-tier 15 RPM limit and triggers the circuit breaker.

### `AgentState` Return Pattern
All agent nodes return `AgentState(**{**state, "key": value})` — never `AgentState(**state, key=value)`. The latter causes `TypeError: got multiple values for keyword argument`.

### Manager LLM — Tool-Use Orchestrator
The Manager uses LangChain `bind_tools` (function-calling) rather than a hardcoded intent classifier. Adding new capabilities requires only registering a new `@tool` function — no classifier changes. The tool-use loop has a hard cap of 5 rounds to prevent infinite loops.

### str_replace Document Editing
`refinement_handler.py` sends the full report to Flash primary and asks for `old_string` + `new_string`. A literal `str.replace(old, new, 1)` is applied. If `old_string` is not found (LLM hallucinated it), the handler retries once with a corrective prompt. This preserves all unchanged sections character-perfect.

### user_id Scoping
All `LongTermMemory` queries include `WHERE user_id = ?`. The FastAPI DB migration adds `user_id TEXT DEFAULT 'default'` to all tables. Existing tests use `user_id="default"` implicitly.

### Hierarchical Document Summarisation
Large documents (PDF, DOCX, TXT) are split into overlapping 3,000-char chunks, each summarised by Flash-Lite, then combined into a final summary. No truncation — all content is covered.

### PageIndex — Two-Tier Document Access Model
Documents are either `scope='user'` (private, `user_id` required) or `scope='system'` (visible to all authenticated users, `user_id=NULL`). DB-level `CHECK` constraints enforce this. Every retrieval query always returns both tiers via `WHERE (user_id=$uid AND scope='user') OR scope='system'`. User documents are deleted on account deletion (ON DELETE CASCADE). System documents are managed exclusively through `POST /admin/documents/upload` protected by the `ADMIN_USER_IDS` env var.

### PageIndex — Embedding + Hybrid Search
`text-embedding-004` (768-dim) is used for both document and query embeddings (different `task_type` for each). The retriever runs two parallel queries — pgvector IVFFlat ANN and Postgres `tsvector` FTS — then merges results with Reciprocal Rank Fusion (k=60). On SQLite (dev), falls back to LIKE-based FTS since pgvector is unavailable. Embeddings are cached in `ResultCache` via SHA256 key to avoid re-embedding identical text.

### Required env vars for PageIndex
`ADMIN_USER_IDS` — comma-separated list of user IDs or emails that can call `/admin/*` endpoints. Leave empty to disable admin access. `UPLOAD_DIR` — directory for raw file storage (default `.uploads`).

---

## Package Versions (Pinned)

| Package | Version |
|---|---|
| `langchain` | 1.2.17 |
| `langgraph` | 1.1.10 |
| `langgraph-checkpoint-sqlite` | 3.0.3 |
| `langchain-google-genai` | 4.2.2 |
| `langchain-tavily` | 0.2.18 |
| `yfinance` | 1.3.0 |
| `langsmith` | 0.8.0 |
| `fastapi` | ≥0.115 |
| `google-auth` | ≥2.29 |
| `python-jose[cryptography]` | ≥3.3 |
| `plotly` | ≥5.24 |
| `pandas` | ≥2.2 |
| `pdfplumber` | ≥0.11 |
| `weasyprint` | ≥62.0 |
| `python-docx` | ≥1.1 |
| `openpyxl` | ≥3.1 |

---

## Free-Tier Limits

| Service | Limit | Mitigation |
|---|---|---|
| Gemini Flash | ~1,500 RPD, 15 RPM | Circuit breaker (3×429 in 30s) + Flash-Lite fallback |
| Gemini Flash-Lite | ~1,500 RPD, 30 RPM | Sub-tasks: summaries, PDF parsing, preference extraction |
| Tavily | 1,000 credits/month | 4-hour diskcache |
| yfinance | No hard limit | 4-hour diskcache |

---

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `CircuitBreakerError` | 3× 429 within 30s | Wait ~1 min; system auto-falls back to Flash-Lite |
| `PartialStateError` | Missing required state at agent boundary | Check `run_trace.json` iteration_log |
| `SanitizationAlert` | Canary token in agent output | Inspect `run_artifacts.json` |
| `401 Unauthorized` on `/auth/me` at startup | Expected — no session cookie yet | Not a bug; handled by `useAuth` catch returning null |
| `button?type=standard 403` | Google button iframe with undefined params | Cosmetic only; sign-in still works |
| PDF export `501 Not Implemented` | weasyprint not installed | `pip install weasyprint`; macOS may need `brew install pango` |
| `GOOGLE_API_KEY not set` | `.env` not loaded | Run from project root |
