# AI Financial Analyst Agent — CLAUDE.md

## Project Overview

Autonomous AI Financial Analyst Agent built as a portfolio/prototype project. Uses a ReAct + Multi-Agent architecture with LangGraph. Produces structured, multi-page financial research reports from a stock ticker query with zero ongoing API cost.

**This is not a production system.** It is a hire-worthy portfolio showcase of agentic AI engineering patterns.

---

## Environment Setup

```bash
conda activate fin-agent        # always activate before working
pip install -e ".[dev]"         # installs project + dev tools editable
cp .env.example .env            # then fill in the 3 API keys
```

### Required API Keys (all free tier)
- `GOOGLE_API_KEY` — [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- `TAVILY_API_KEY` — [app.tavily.com/sign-in](https://app.tavily.com/sign-in) (1,000 free searches/month)
- `LANGSMITH_API_KEY` — [smith.langchain.com](https://smith.langchain.com)

Note: LangSmith env vars changed in v0.8 — use `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true`, not the old `LANGCHAIN_*` names.

---

## Running the Project

```bash
# Start the Streamlit UI
streamlit run ui/app.py

# Demo replay without any API calls (for interviews)
# Check "Dry-run mode" in the sidebar and upload a run_trace.json
```

---

## Running Tests

```bash
pytest tests/unit/          # fast, no API calls — always run these first
pytest tests/integration/   # agent logic with mocked LLM/tools
pytest tests/adversarial/   # security: prompt injection payload detection
pytest tests/e2e/           # full pipeline with pre-recorded mocked responses

# Full suite with coverage
pytest --cov=ai_financial_analyst --cov-report=term-missing
```

All four layers must pass before any feature is complete. Current status: **57/57** (unit + integration + adversarial).

---

## Architecture

Three-agent LangGraph pipeline, sequential execution:

```
User Query → Orchestrator → Researcher → Quant Analyst → Editor → Markdown Report
```

| Agent | File | Responsibilities |
|---|---|---|
| Researcher | `agents/researcher.py` | yfinance + Tavily fetch; max 5 iterations per ticker |
| Quant Analyst | `agents/quant_analyst.py` | CAGR, P/E vs benchmark, SOP analysis, bull/bear cases |
| Editor | `agents/editor.py` | SOP rubric check, grounding check, disclaimer enforced |
| Orchestrator | `agents/orchestrator.py` | LangGraph StateGraph, SQLite checkpointing, safe wrappers |

---

## Key Files

| Path | Purpose |
|---|---|
| `ai_financial_analyst/core/state.py` | `AgentState` TypedDict — shared contract between all agents |
| `ai_financial_analyst/core/llm.py` | Gemini client with `tenacity` retry + circuit breaker |
| `ai_financial_analyst/core/sanitizer.py` | Prompt injection filter (full-content rejection) + canary token |
| `ai_financial_analyst/core/budget_tracker.py` | Free-tier API call counter; warns at 80% |
| `ai_financial_analyst/core/cache.py` | `diskcache` 4-hour TTL for yfinance + Tavily results |
| `ai_financial_analyst/core/tracing.py` | `run_trace.json` builder + LangSmith hooks |
| `ai_financial_analyst/tools/calculator.py` | AST-validated numexpr evaluator (no REPL) |
| `ai_financial_analyst/data/benchmarks.json` | Static GICS sector P/E averages (no API call) |
| `ui/app.py` | Streamlit UI with TAO stream, transparency panel, dry-run replay |
| `docs/AI_Financial_Analyst_Agent_PRD.docx` | Full product requirements document |

---

## Models

| Model | Use |
|---|---|
| `gemini-3-flash-preview` | Primary ReAct reasoning loop |
| `gemini-3.1-flash-lite-preview` | Sanitizer extraction sub-tasks (optional) |

`max_retries=1` is set on both LLM instances. In `langchain-google-genai` 4.x, `max_retries=0` means "use SDK default (5 retries)" — setting `1` disables the SDK layer so only our `tenacity` circuit-breaker retries.

---

## Critical Design Decisions (Do Not Change Without Review)

### No Python REPL
`CalculatorTool` uses `numexpr` with a three-level AST guard:
1. Node-type whitelist — only arithmetic AST node types
2. Name allowlist (`_SAFE_NAMES`) — blocks `__import__`, `os`, etc.
3. Function allowlist (`_SAFE_FUNCTIONS`) — blocks all calls except `sqrt`, `log`, etc.
4. Constant type check — string/bytes literals rejected

Changing to a general REPL is a security regression.

### Full-Content Injection Rejection
`ContentSanitizer._regex_filter()` rejects the **entire content block** when any injection pattern matches. Sentence-level `[REDACTED]` replacement was insufficient — surrounding context still guided adversarial behaviour. Do not revert to partial stripping.

### Sequential Agent Execution
Agents run one at a time. Do **not** make them concurrent — this saturates the free-tier 15 RPM limit and triggers the circuit breaker.

### `AgentState` Return Pattern
All agent nodes return via `AgentState(**{**state, "key": value, ...})` — **not** `AgentState(**state, key=value)`. TypedDict spreads all keys as kwargs; duplicate keys cause `TypeError: got multiple values for keyword argument`.

### max_iterations = 5
The Researcher agent hard-caps at 5 tool calls per ticker. Increasing this risks exhausting the free-tier RPD limit on multi-ticker runs.

### Tavily over DuckDuckGo
Tavily is LangChain's default search tool, purpose-built for AI agents. `DuckDuckGoSearchRun` is a legacy tool. Google Custom Search API is closed to new signups in 2026. Do not revert.

---

## Package Versions (Pinned — Do Not Upgrade Without Testing)

| Package | Version | Notes |
|---|---|---|
| `langchain` | 1.2.17 | Major version jump from 0.3.x |
| `langgraph` | 1.1.10 | `langgraph.prebuilt` deprecated in 1.x |
| `langgraph-checkpoint-sqlite` | 3.0.3 | Separate package required for `SqliteSaver` |
| `langchain-google-genai` | 4.2.2 | Now uses `google-genai` SDK; `max_retries` quirk |
| `langchain-tavily` | 0.2.18 | Official Tavily + LangChain integration |
| `yfinance` | 1.3.0 | |
| `langsmith` | 0.8.0 | Env var names changed: `LANGSMITH_*` |

---

## Free-Tier Limits

| Service | Limit | Mitigation |
|---|---|---|
| Gemini Flash | ~1,500 RPD, 15 RPM | Circuit breaker (3 × 429 in 30s → halt) + budget tracker |
| Gemini Flash-Lite | ~1,500 RPD, 30 RPM | Used only for sanitizer sub-tasks |
| Tavily | 1,000 credits/month | 4-hour diskcache for repeated queries |
| yfinance | Unofficial API, no hard limit | 4-hour diskcache |

---

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `CircuitBreakerError` | 3× 429 within 30s | Wait ~1 min; use `--dry-run` for demo |
| `PartialStateError` | Agent boundary missing required field | Check `iteration_log` in trace for upstream tool failure |
| `SanitizationAlert` | Canary token in agent output | Potential injection — inspect `run_trace.json` |
| `TypeError: got multiple values` | `AgentState(**state, key=val)` pattern | Use `AgentState(**{**state, "key": val})` instead |
| `GOOGLE_API_KEY not set` | `.env` not loaded | Run from project root; ensure `python-dotenv` loaded |
