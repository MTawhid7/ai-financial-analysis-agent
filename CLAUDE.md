# AI Financial Analyst Agent — CLAUDE.md

## Project Overview

Conversational AI Financial Analyst Agent. Uses a ReAct + Multi-Agent architecture with LangGraph. Natural language input, Google OAuth authentication, per-user persistent memory, and real-time SSE streaming through a FastAPI + React stack.

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

Legacy Streamlit UI (dry-run demos only):
```bash
streamlit run ui/app.py
```

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

Current status: **156/156** Python tests passing + frontend build clean.

---

## Architecture

```
React 19 + Vite (port 5173)
  ↕ Google OAuth popup (@react-oauth/google)
  ↕ fetch credentials:include — REST
  ↕ EventSource credentials:include — SSE
FastAPI 0.115 (port 8000)
  ↕ session_manager: user_id → ConversationalAgent (LRU, 30-min TTL)
ConversationalAgent  ← Flash-Lite intent classifier (5 intents)
  ↓ financial_analysis    ↓ memory_query    ↓ financial_question    ↓ off_topic
run_pipeline()       search summaries    primary LLM + history   rejection
  ↓
Researcher → Quant Analyst → Editor → Markdown report
```

| Component | File | Responsibilities |
|---|---|---|
| FastAPI app | `backend/main.py` | CORS, lifespan DB migration, router registration |
| Auth | `backend/routers/auth.py` | Google ID token → JWT httpOnly cookie |
| Chat + SSE | `backend/routers/chat.py` | POST /chat → event_id; GET /stream → EventSource |
| Session manager | `backend/core/session_manager.py` | user_id → ConversationalAgent LRU cache |
| ConversationalAgent | `agents/conversational_agent.py` | Routing, memory injection, pipeline calls |
| IntentClassifier | `agents/intent_classifier.py` | Flash-Lite JSON classifier; 5 intents |
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
| `ai_financial_analyst/core/state.py` | `AgentState` TypedDict — inner pipeline contract |
| `ai_financial_analyst/core/conversation_state.py` | `ConversationState` TypedDict — chat layer |
| `ai_financial_analyst/core/llm.py` | Gemini client: retry + circuit breaker + Flash-Lite fallback |
| `ai_financial_analyst/core/sanitizer.py` | Injection filter (full-content rejection) + canary token |
| `ai_financial_analyst/memory/long_term.py` | SQLite: preferences, summaries, conversations, messages (user-scoped) |
| `ai_financial_analyst/memory/memory_manager.py` | Memory facade: context injection, preference extraction, summary saving |
| `ai_financial_analyst/tools/calculator.py` | AST-validated numexpr evaluator (no REPL) |

---

## Critical Design Decisions (Do Not Change Without Review)

### No Python REPL
`CalculatorTool` uses `numexpr` with a three-level AST guard (node whitelist, name allowlist, function allowlist). Changing to a general REPL is a security regression.

### Full-Content Injection Rejection
`ContentSanitizer._regex_filter()` rejects the **entire content block** on any injection pattern match. Sentence-level redaction is insufficient — surrounding context still guides adversarial behaviour.

### Sequential Agent Execution
Agents run one at a time. Concurrent execution saturates the free-tier 15 RPM limit and triggers the circuit breaker.

### `AgentState` Return Pattern
All agent nodes return `AgentState(**{**state, "key": value})` — never `AgentState(**state, key=value)`. The latter causes `TypeError: got multiple values for keyword argument`.

### `ConversationState` vs `AgentState` — Two Separate TypedDicts
`ConversationState` is the chat layer (session ID, messages, intent). `AgentState` is the pipeline (raw_data, analysis, report). Never merged — the agent calls `run_pipeline()` and receives the final report; it never touches `AgentState` directly.

### Five-Intent Taxonomy
`memory_query` must remain a distinct intent. Without it, "What did we find about AAPL earlier?" gets classified as `financial_analysis` (AAPL is present) and re-runs the full pipeline instead of returning the stored summary.

### user_id Scoping
All `LongTermMemory` queries include `WHERE user_id = ?`. The FastAPI DB migration adds `user_id TEXT DEFAULT 'default'` to all tables — safe to run on existing databases. Existing tests use `user_id="default"` implicitly.

### Flash-Lite for Classification
`IntentClassifier` and `MemoryManager` preference extraction use `get_subllm()` (Flash-Lite). Primary LLM (Flash) is reserved for analysis reasoning.

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

---

## Free-Tier Limits

| Service | Limit | Mitigation |
|---|---|---|
| Gemini Flash | ~1,500 RPD, 15 RPM | Circuit breaker (3×429 in 30s) + Flash-Lite fallback |
| Gemini Flash-Lite | ~1,500 RPD, 30 RPM | Sub-tasks only (classification, summaries) |
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
| `GOOGLE_API_KEY not set` | `.env` not loaded | Run from project root |
