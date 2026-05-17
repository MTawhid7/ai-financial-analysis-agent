# AI Financial Analyst Agent

A **conversational AI financial analyst** with Google authentication, persistent memory, real-time streaming, interactive charts, multi-format export, surgical report editing, and an intelligent tool-use orchestrator. Built on a ReAct + Multi-Agent architecture using LangGraph, Gemini free tier, yfinance, and Tavily.

> **Portfolio project** — demonstrates production-grade agentic AI engineering. Not for real investment decisions.

---

## What it can do

| Say this | What happens |
|---|---|
| *"Analyse AAPL"* | Full pipeline → report + 4 Plotly charts + PDF/Word/Excel export |
| *"Compare AAPL vs MSFT"* | Both tickers analysed → side-by-side comparison table |
| *"Make the bear case more pessimistic"* | Surgical str_replace edit — only that section changes |
| *"Show me a chart of AAPL's financial profile"* | On-demand radar chart generated |
| *"Show NVDA with Bollinger Bands over 10 years"* | Candlestick + BB overlays, period="10y" |
| *"Compare Nasdaq vs S&P 500 year to date"* | Normalised return chart, QQQ vs SPY |
| *"What did we find about AAPL last time?"* | Returns stored summary, no API calls |
| *"I prefer conservative analysis"* | Preference saved, injected into future responses |
| *"Analyse AAPL, then compare with MSFT"* | Manager chains both tools in one message |
| Upload XLSX, DOCX, PDF, CSV, TXT, MD, JSON | Immediate summary + background PageIndex deep-indexing |
| *"What does page 47 of the uploaded report say?"* | PageIndex returns that exact page with citation |
| *"Find the revenue table in my annual report"* | Hybrid vector+FTS search → page-level result |
| *"Analyse AAPL"* (enhanced) | 7 yfinance data types: price + fundamentals + cash flow + earnings + risk metrics + quarterly trend + dividend history |

---

## Architecture

```
React 19 + Vite  →  FastAPI 0.115  →  ConversationalAgent  →  Manager LLM (tool-use)  →  LangGraph Pipeline
       │                   │                    │                         │                        │
  Google OAuth         JWT cookie          LLMRegistry               11 tools                 Researcher (concurrent)
  SSE streaming        DB migration        MemoryBackend              auto-routing             Quant Analyst (DCF+scenarios)
  Citation badges      user_id scope       (Protocol)                 memory context           Editor

Layered package structure:
  config/    — Settings (pydantic-settings, all config from env vars)
  core/llm/  — LLMClient Protocol, CircuitBreaker, LLMRegistry (no module-level singletons)
  core/utils — safe_float, estimate_tokens, extract_domain (shared utilities)
  data/      — yahoo/* (7 modules), market/*, benchmark/*, search/* (injectable, concurrent)
  tools/     — thin @tool wrappers delegating to data/
  agents/    — ConversationalAgent (DI), ManagerAgent (cached tools), researcher (concurrent)
  memory/    — MemoryBackend Protocol, SQLite + InMemory implementations
  pipeline/  — LangGraph DAG: researcher → quant_analyst → editor (conditional routing)
```

```mermaid
flowchart LR
    User -->|natural language| React["⚛️ React + Vite\n(port 5173)"]
    React -->|Google OAuth| Google[Google]
    React -->|REST + SSE| API["⚡ FastAPI\n(port 8000)"]
    API --> MGR["🧠 Manager LLM\n(tool-use orchestrator)"]

    MGR -->|run_financial_analysis| Pipeline
    MGR -->|compare_stocks| Pipeline
    MGR -->|edit_report_section| DB2[("📋 reports\ntable")]
    MGR -->|recall_past_analysis| DB[("🐘 Postgres\nSupabase")]
    MGR -->|answer_finance_question| LLM["Gemini Flash\n(direct answer)"]
    MGR -->|generate_chart| Charts["📊 16 chart types\nplotly JSON"]
    MGR -->|search_documents| PageIdx[("📄 PageIndex\npgvector+FTS")]
    MGR -->|get_document_page| PageIdx
    MGR -->|reject_request| Reject["Polite rejection"]

    subgraph Pipeline ["LangGraph Pipeline"]
        R["🔍 Researcher"] --> Q["📐 Quant"] --> E["✏️ Editor"]
    end

    Pipeline --> Report["Report + Citations\n+ 4 Charts\n+ run_artifacts.json"]
```

---

## Key Engineering Decisions

### Manager LLM Orchestrator (replaces hardcoded intent classifier)
The Manager uses LangChain `bind_tools` (function-calling) to autonomously decide which tool(s) to call, in what order. Handles compound requests ("Analyse AAPL then compare with MSFT") and adds new capabilities by simply registering a new `@tool` — no routing changes needed.

### str_replace Surgical Document Editing
When refining a report, the LLM receives the **full** document and outputs `old_string` + `new_string`. A `_flexible_str_replace()` function tries exact match first, then falls back to line-strip normalization (tolerates LLM trailing-space differences). Each successful edit is persisted as a new INSERT row — natural version history, rollback is always available.

### Hierarchical Document Summarisation
Large documents (PDF, DOCX, TXT) are split into overlapping 3,000-char chunks, each summarised by Flash-Lite, then combined. No truncation — important context is preserved regardless of document length.

### Citation System
`(Source: fundamentals)` inline citations are parsed into numbered `[N]` superscript badges with click/hover popovers. Each source shows a always-visible link (Yahoo Finance, Reuters, Bloomberg, etc. — parsed from actual URLs). A References section at the bottom lists all citations.

### No Python REPL
`CalculatorTool` uses `numexpr` with an AST whitelist. File parsers produce fixed-schema JSON summaries only — no raw user data reaches the LLM, no arbitrary code execution.

### Rate Limit Resilience
`tenacity` retry + half-open circuit breaker (5×429 in 60s, 60s probe delay). Automatic fallback from Gemini Flash to Flash-Lite — analysis continues at reduced quality rather than failing. Once a probe request succeeds, the breaker resets and full quality is restored.

### Conditional Pipeline Routing
The LangGraph pipeline uses conditional edges after `researcher` and `quant_analyst`. When no usable data is retrieved (all errors, empty raw_data, or rate-limited), the pipeline routes to an `early_exit` node instead of burning API quota on downstream agents.

### Multi-Ticker Parallelism (Researcher)
The researcher fetches data for all tickers concurrently via `asyncio.gather`. yfinance and Tavily calls consume zero Gemini RPM, so parallelism is safe. One ticker failing does not abort the others. Concurrency is capped at 3 (configurable via `RESEARCHER_TICKER_CONCURRENCY`).

### Section-Aware Report Editing
When refining a report, the system infers the target section from the user's message ("make the bull case more pessimistic" → "Bull Case"). The LLM receives only that section instead of the full document, reducing token usage and improving edit accuracy. Falls back to full-document editing when no section is detected.

### Short-Term Memory Hierarchical Summarisation
When conversation history exceeds the token budget, dropped turn-pairs are condensed into a synthetic summary message by the subllm rather than being silently discarded. Available via `MemoryManager.get_windowed_context()`.

### Report Quality Scoring
After every `write_report()` call, `_check_quality()` validates word counts per section against minimum thresholds and logs warnings for under-length sections (e.g., Quantitative Analysis < 80 words). Non-blocking — the report is returned regardless.

### PageIndex Sub-Page Chunking
Pages longer than 1,500 chars are split into overlapping sentence/paragraph chunks. Each chunk gets its own embedding vector, stored as a `DocumentPage` row with `chunk_index >= 1`. Search results are deduplicated back to root pages for display. Every embedding row records its source model for staleness detection.

### HyDE Query Expansion (PageIndex)
When searching uploaded documents, the retriever uses Hypothetical Document Embedding: Flash-Lite generates a synthetic passage that would answer the query, that passage is embedded (not the raw question), and the embedding is used for vector search. Questions and passages inhabit different embedding spaces — this improves retrieval for short or ambiguous queries.

### Semantic Memory Retrieval
Analysis summaries are embedded with Gemini `text-embedding-004` at save time and stored as JSON vectors in SQLite. At query time, cosine similarity ranks past analyses by semantic relevance — "profit margins" correctly matches "earnings quality" across sessions.

---

## Free-Tier Setup

### Prerequisites
- Python 3.11+ · Node.js 18+
- Google AI Studio account (free `GOOGLE_API_KEY`)
- Google Cloud Console project with OAuth 2.0 Client ID
- Tavily account (free `TAVILY_API_KEY` — 1,000 searches/month)
- LangSmith account (free `LANGSMITH_API_KEY`)

### Installation

```bash
git clone <this-repo>
cd ai-financial-analyst
conda activate fin-agent
pip install -e ".[server]"
cp .env.example .env              # fill in all 6 required variables
cd frontend
npm install
cp .env.local.example .env.local  # add VITE_GOOGLE_CLIENT_ID
```

### Google OAuth setup
1. [console.cloud.google.com/apis/credentials](https://console.cloud.google.com/apis/credentials) → Create OAuth 2.0 Client ID → Web application
2. Authorised JavaScript origins: `http://localhost:5173`
3. Copy Client ID → `.env` (`GOOGLE_CLIENT_ID`) and `frontend/.env.local` (`VITE_GOOGLE_CLIENT_ID`)
4. Copy Client Secret → `.env` (`GOOGLE_CLIENT_SECRET`)
5. Generate JWT secret: `python -c "import secrets; print(secrets.token_hex(32))"` → `FASTAPI_JWT_SECRET`

### Run

```bash
# Terminal 1 — backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend && npm run dev
# Open http://localhost:5173
```

---

## Running Tests

```bash
pytest tests/unit/ tests/integration/ tests/adversarial/ -v
# 622 tests passing
cd frontend && npm run build      # zero TypeScript errors required
```

---

## Project Structure

```
ai_financial_analyst/
  config/
    settings.py              Single-source Settings class (pydantic-settings, all env-overridable)
  core/
    llm/                     LLM abstraction layer
      protocols.py           LLMClient Protocol (@runtime_checkable)
      circuit_breaker.py     CircuitBreaker (per-instance, no module-level singletons)
      gemini.py              Gemini Flash + Flash-Lite wrappers with retry
      registry.py            LLMRegistry: creates and caches LLM instances per session
    utils.py                 Shared utilities: safe_float, estimate_tokens, extract_domain
    sanitizer.py             Injection filter + dynamic canary token + NFKC normalization
    budget_tracker.py / cache.py / tracing.py / artifacts.py
    state.py / conversation_state.py
  data/                      Data access layer (all fetch logic lives here)
    yahoo/                   7 focused modules: price, fundamentals, balance_sheet,
                             cash_flow, earnings, metrics, trends + concurrent coordinator
    market/                  risk_free.py (^TNX), sp500.py (^GSPC)
    benchmark/               damodaran.py (live), static.py (lazy fallback), normalizer.py
    search/                  tavily.py (injectable client), credibility.py (source tiers)
  tools/                     Thin LangChain @tool wrappers — delegate to data/
    yahoo_finance.py / benchmark_lookup.py / web_search.py / market_data.py
    calculator.py            numexpr AST guard + format hints + context variables
    financial_formulas.py    NPV, IRR, PV, FV, CAGR, WACC, payback, ROI
  agents/
    conversational_agent.py  DI constructor + create() factory; uses LLMRegistry
    manager.py               Tool-use orchestrator (cached tools, _StepCallbackProxy)
    comparison_agent.py      Multi-ticker pipeline + comparison table
    refinement_handler.py    str_replace surgical editing; INSERT versioning
    researcher.py            Concurrent two-phase yfinance fetch + TavilySearchClient
    quant_analyst.py         DCF, scenario analysis, structured output
    editor.py                SOP rubric, financial number grounding check
    orchestrator.py          LangGraph DAG with conditional routing + early_exit
  memory/
    protocol.py              MemoryBackend Protocol (@runtime_checkable)
    in_memory.py             InMemoryBackend (zero I/O, for tests)
    long_term.py             SQLite implementation via aiosqlite
    memory_manager.py        Facade: context injection, preference extraction, summary saving
    short_term.py            Turn-pair aware token-budget context window
  pageindex/                 PageIndex document retrieval system
    __init__.py              Public API: index_document, search_documents, get_page
    embedder.py              Gemini text-embedding-004 (768-dim) + ResultCache
    pipeline.py              Ingestion: extract → summarise → embed → store → link
    retriever.py             Hybrid search: pgvector ANN + Postgres FTS + RRF + HyDE
    ocr.py                   Scanned PDF detection + pytesseract OCR fallback
  charts/                    16 Plotly chart types (modular)
    _theme.py / _data.py     Palette, fetch utils, ticker aliases, earnings annotations
    price_action.py          Candlestick (BB/EMA overlays, earnings markers, date ranges)
    technical.py             RSI, MACD
    combined.py              Price+RSI, Price+MACD multi-panel
    fundamentals.py          Revenue trend, margins, cashflow, debt profile
    comparison.py            Normalised return comparison
    risk.py                  Drawdown
    pipeline.py / dispatcher.py
  parsers/                   File parsers (modular)
    _summarise.py / _page_extractor.py
    csv_parser.py / pdf_parser.py / excel_parser.py / word_parser.py
    text_parser.py / json_parser.py
  tools/
    yahoo_finance.py / web_search.py / calculator.py / benchmark_lookup.py / report_writer.py
    pdf_exporter.py / docx_exporter.py / xlsx_exporter.py (with live CAGR formulas)

backend/
  main.py                    FastAPI app, CORS, lifespan DB migration
  routers/
    auth.py / conversations.py / chat.py / files.py / memory.py / feedback.py
    admin.py                 Admin: system document management (ADMIN_USER_IDS protected)
  core/
    auth.py / database.py / session_manager.py / event_store.py / deps.py
    models.py                ORM models: User, Conversation, Message, Report, Document,
                             DocumentPage, PageLink (pgvector-aware)

frontend/
  src/
    pages/          LoginPage, ChatPage (collapsible sidebar rail)
    components/
      chat/         ChatInterface, ChatBubble, MessageInput (paperclip attachment),
                    CitationRenderer (numbered badges), ExportMenu, ProvenancePanel
      sidebar/      ConversationList (inline rename), MemoryPanel
      PlotlyChart.tsx (interactive: hover/zoom/pan toolbar)
    hooks/          useAuth, useStreamingChat
    lib/            api.ts (typed fetch wrappers), constants.ts

tests/              unit / integration / adversarial / e2e
docs/
  MANUAL_TESTING.md  Complete manual testing guide (18 sections)
```

---

## Known Limitations

| Limitation | Notes |
|---|---|
| Gemini free tier: ~1,500 RPD, 15 RPM | Auto-fallback to Flash-Lite on rate limit |
| yfinance data lag (~15 min) | `data_timestamp` field makes this explicit |
| Tavily: 1,000 credits/month | 4-hour diskcache reduces consumption |
| Static sector benchmarks | Approximate 2024 P/E averages — relative comparison only |
| Sequential pipeline (~60–120s / 2–3 tickers) | Required to stay within free-tier RPM cap |
| PDF export requires weasyprint | `pip install weasyprint`; macOS may need `brew install pango` |
| Single-process FastAPI sessions | Fine for local/demo; Redis needed for horizontal scaling |

---

## Security

- No secrets committed — all credentials in `.env` (gitignored)
- No Python REPL — constrained `numexpr` only; file parsers produce fixed-schema summaries
- Prompt injection filter on all web search content; CSV/XLSX formula injection scrubbed
- Canary token detection in agent output
- All tool inputs validated with Pydantic v2 `extra='forbid'`
- JWT in httpOnly cookie (not accessible to JavaScript)
- Per-user data isolation via `user_id` scoping on all Postgres queries

---

*DISCLAIMER: Portfolio and educational purposes only. Generated reports should not be used for real investment decisions. This is not financial advice.*
