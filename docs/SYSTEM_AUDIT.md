# System Audit Report: Implementation Status

**Original audit date:** 2026-05-15  
**Last updated:** 2026-05-16  
**Scope:** Full pipeline — data layer, LLM infrastructure, memory, orchestration, security, charts, PageIndex  

Status legend: ✅ Resolved · ⚠️ Partial · ❌ Remaining

---

## 1. Financial Data Layer

### 1.1 Yahoo Finance Tool (`tools/yahoo_finance.py`)

| Issue | Status | Notes |
|---|---|---|
| No adjusted close prices | ✅ | `auto_adjust=True` throughout; split+dividend adjusted OHLCV |
| Stale `info` dict | ✅ | Uses `fast_info.last_price` for price; `TTL_PRICE=15m`, `TTL_FUNDAMENTALS=6h` per data type |
| Missing cash flow data | ✅ | `cash_flow` data type: OCF, FCF, capex, D&A, FCF yield, cash-conversion ratio, dividend history |
| No forward estimates | ✅ | `analyst_target_mean/high/low`, forward P/E in `fundamentals`; earnings EPS estimates in `earnings` |
| Silent data degradation | ✅ | Every response now carries `data_quality: "FULL"\|"PARTIAL"\|"UNAVAILABLE"` with a `degradation_note` explaining the gap; `period_requested` vs `period_received` surfaced in `price_history` |
| No dividend data | ✅ | Dividend history (payments, annual totals, 3Y CAGR) in `cash_flow` data type |
| 52-week window is approximate | ✅ | Uses `timedelta(365)` for true 52-week window, not `prices[-252:]` |
| No corporate event timeline | ⚠️ | Stock splits now in `corporate_events` list in `price_history` (date, ratio, description). Buyback announcements not available on yfinance free tier. |

---

### 1.2 Benchmark Lookup (`tools/benchmark_lookup.py`)

| Issue | Status | Notes |
|---|---|---|
| Static data, zero refresh | ✅ | Live-fetched from Damodaran NYU Stern HTML (6 URLs); 30-day cache; static JSON fallback |
| Only P/E | ✅ | Returns 7 multiples: trailing PE, forward PE, EV/EBITDA, P/B, P/S, operating margin %, beta |
| No fuzzy sector matching | ✅ | `_normalise_sector()` now: (1) exact GICS match, (2) alias dict for all common yfinance sector names (Technology→IT, Consumer Cyclical→Consumer Discretionary, etc.), (3) `difflib.get_close_matches(cutoff=0.6)` fuzzy fallback. `sector_matched_from` + `sector_match_method` fields surfaced in result. |
| No geographic segmentation | ✅ | `BenchmarkLookupInput` accepts optional `country`. Non-US companies get a `geographic_context` block: EM companies flagged with ~30% typical PE discount note; developed ex-US flagged with ~12% note; unclassified countries warned. `quant_analyst.py` passes `country` from fundamentals. |

---

## 2. Quantitative Analysis

### 2.1 Quant Analyst (`agents/quant_analyst.py`)

| Issue | Status | Notes |
|---|---|---|
| Only price CAGR | ✅ | Total-return CAGR (adjusted for splits+dividends) read from `price_metrics.total_return_cagr_pct`; raw-price fallback via calculator |
| No risk-adjusted metrics | ✅ | Sharpe, Sortino, max drawdown, beta vs S&P 500, volatility — all read from `price_metrics` |
| No DCF framework | ✅ | `_compute_dcf()` in quant_analyst.py: 5-year FCF projection, WACC via CAPM, terminal value, per-share intrinsic value, margin of safety. Option A: negative FCF returns `{dcf_not_applicable, reason}` — no meaningless negative value. |
| Bull/bear are LLM narrative | ✅ | `_compute_scenarios()` computes all price targets deterministically (P/E×EPS, analyst consensus, DCF). LLM only writes 1-sentence qualitative narrative per bullet citing computed numbers — cannot invent price targets. |
| LLM response parsing is brittle | ✅ | `with_structured_output(_SOPOutput)` added to `RateLimitFallbackLLM`; quant_analyst uses it for the narration step. Gemini JSON mode enforces schema; no markdown fence stripping. |
| Hardcoded sector taxonomy | ✅ | `_YFINANCE_TO_GICS` dict removed; raw yfinance sector string passed directly to `benchmark_lookup_tool` which resolves via alias dict + `difflib` fuzzy match (implemented in prior session). |
| No scenario analysis | ✅ | `_compute_scenarios()`: three independent methods — P/E × forward EPS (bear/base/bull at 0.8×/1.0×/1.2× sector P/E), analyst consensus range (low/mean/high), DCF intrinsic value — each with upside_pct vs current price. |

---

### 2.2 Calculator Tool (`tools/calculator.py`)

| Issue | Status | Notes |
|---|---|---|
| No financial formula library | ❌ | Still string-expression numexpr only; no `pv()`, `fv()`, `npv()`, `irr()` |
| Rounding inconsistency | ❌ | Hardcoded 6 decimal places; no per-type formatting |
| No unit tracking | ❌ | No dimension checking |
| No result range validation | ❌ | Float overflow / NaN returned silently |

**Severity: MEDIUM — Narrow scope but safe; financial functions would add value.**

---

## 3. Research & Retrieval

### 3.1 Researcher Agent (`agents/researcher.py`)

| Issue | Status | Notes |
|---|---|---|
| Hardcoded year in news query | ✅ | `_current_year()` → `datetime.utcnow().year` |
| No adaptive iteration | ⚠️ | `MAX_ITERATIONS=10` (raised from 5); still no early-exit when coverage threshold met |
| No multi-source reconciliation | ❌ | First yfinance result used for each data type; no cross-source validation |
| Token estimation | ⚠️ | `len(string) // 4` still used in researcher's iteration log; short_term memory now uses content-type-aware estimate |
| No data freshness labeling | ⚠️ | `data_timestamp` field present in price_history; other types do not carry `as_of` timestamps |

---

### 3.2 Web Search Tool (`tools/web_search.py`)

| Issue | Status | Notes |
|---|---|---|
| 4-hour cache on financial news | ✅ | `TTL_WEB_SEARCH = 1h` (from 4h); `get_or_fetch(..., ttl=TTL_WEB_SEARCH)` wired correctly |
| No source credibility scoring | ❌ | All results ranked equally by Tavily score; no publication-tier weighting |
| No date filtering on results | ❌ | No `published_after` constraint; stale articles can still be returned |
| Sanitizer fallback leaks raw text | ✅ | Fallback stub returns `key_facts=[]` — no raw content reaches the agent |
| `search_depth="basic"` is silent downgrade | ❌ | Still hardcoded to `"basic"`; no retry with query reformulation |

---

## 4. LLM Infrastructure

### 4.1 LLM Client (`core/llm.py`)

| Issue | Status | Notes |
|---|---|---|
| Circuit breaker window too short / no recovery | ✅ | 5×429/60s threshold; half-open state with 60s probe delay; `probe_succeeded()` / `probe_failed()` |
| No recovery from Flash-Lite | ✅ | Probe success resets breaker to CLOSED; full-quality Flash restored automatically |
| Streaming/non-streaming mismatch | ⚠️ | Primary `streaming=True`, fallback `streaming=False` — intentionally different; downstream streaming SSE handles both via `content_to_str()`; no stream adapter |
| Hardcoded model names | ❌ | `_PRIMARY_MODEL = "gemini-3-flash-preview"`, `_SUB_MODEL = "gemini-3.1-flash-lite-preview"` — hardcoded constants; no model registry |
| No per-call timeout | ✅ | `asyncio.wait_for(coro, timeout=120s)` in `ainvoke()`; sync `invoke()` lacks a timeout |
| No token counting | ❌ | All rate-limit logic is request-based; no token accumulation or token-weighted quota management |

---

### 4.2 Budget Tracker (`core/budget_tracker.py`)

| Issue | Status | Notes |
|---|---|---|
| Counts requests, not cost | ❌ | Still call-counting; no token-weighted cost model |
| Daily budget, not per-minute | ✅ | `_RpmBucket` tracks rolling 60s window; warns when `current_rpm > 15/30`; `primary_rpm_current` / `sub_rpm_current` in `get_stats()` |
| 80% alert too late / only one threshold | ❌ | Single 80% threshold; no 60% soft warning or 95% deferral mode |
| No per-tool budgeting | ❌ | All calls share one pool |
| Cache hit recording is no-op | ❌ | `record_cache_hit()` increments counter but doesn't adjust budget |

---

## 5. Memory System

### 5.1 Long-Term Memory (`memory/long_term.py`)

| Issue | Status | Notes |
|---|---|---|
| Keyword-only retrieval | ✅ | Semantic search: Gemini `text-embedding-004` embeddings stored as `embedding_json`; cosine similarity in Python; LIKE fallback for un-embedded rows |
| No preference versioning | ❌ | Preferences upserted in place; no `(key, value, created_at, superseded_at)` history |
| No preference conflict resolution | ❌ | Contradictory preferences coexist silently; latest wins |
| Summaries never expire or decay | ❌ | Uniform weight regardless of age; no time-decay scoring |
| Only 2 summaries for context | ❌ | `limit=2` hardcoded in `memory_manager.py:build_memory_context()`; not size-adaptive |

---

### 5.2 Short-Term Memory (`memory/short_term.py`)

| Issue | Status | Notes |
|---|---|---|
| No real tokenizer | ⚠️ | Improved heuristic: JSON-heavy content → 2 chars/token; prose → 4 chars/token; not a true tokenizer |
| No message priority | ❌ | No priority ordering (system > recent > old) |
| Drops messages wholesale | ❌ | Turn pairs are kept together but still dropped entirely; no hierarchical summarization |
| Breaks mid-turn pairs | ✅ | Turn-pair aware: user+assistant pair is always included together or dropped together |

---

### 5.3 Memory Manager (`memory/memory_manager.py`)

| Issue | Status | Notes |
|---|---|---|
| Regex pre-filter is brittle | ❌ | `_PREFERENCE_SIGNALS` static regex still in place; novel phrasings missed |
| No structured output enforcement | ❌ | Preference extraction still uses free-form JSON parsing with markdown-fence stripping |
| Preference semantics ignored | ❌ | "conservative" and "risk-averse" stored as different entries; no synonym normalization |
| Summary truncation at 3000 chars | ❌ | Report passed to summarizer still truncated at `report_markdown[:3000]` |
| No context relevance ranking | ✅ | `search_summaries(embedder=self._embedder)` ranks by cosine similarity when embedder provided |

---

## 6. Orchestration & Pipeline

### 6.1 Orchestrator (`agents/orchestrator.py`)

| Issue | Status | Notes |
|---|---|---|
| Hardcoded linear pipeline | ✅ | Conditional routing via `_route_after_researcher` and `_route_after_quant`; `early_exit` node skips downstream agents on empty data / rate-limited state |
| No agent-level retry | ❌ | Node failure still aborts the downstream pipeline; no re-run of individual failed nodes |
| No parallelism for multi-ticker | ❌ | Tickers still processed sequentially within researcher, quant, editor |
| Error types collapse to same output | ⚠️ | `PartialStateError` → `PARTIAL`, `CircuitBreakerError` → `RATE_LIMITED`, generic → `FAILED`; distinct status codes exist but all produce the same user-visible error format |
| SQLite checkpoint in production | ❌ | `AsyncSqliteSaver` still used; Postgres checkpointer not implemented |

---

### 6.2 Comparison Agent (`agents/comparison_agent.py`)

| Issue | Status | Notes |
|---|---|---|
| Full pipeline per ticker for comparison | ❌ | All tickers passed to a single `run_pipeline()` call which processes them sequentially |
| Hard-truncates data at 4000/2000 chars | ❌ | `analysis_json[:4000]` and `fundamentals_json[:2000]` still present |
| No validation of output table | ❌ | Comparison result is unvalidated LLM output; missing tickers in table not detected |
| Comparison dimensions fixed | ❌ | Always the same columns; no user-specified dimensions |

---

### 6.3 Refinement Handler (`agents/refinement_handler.py`)

| Issue | Status | Notes |
|---|---|---|
| No fuzzy matching fallback | ✅ | `_flexible_str_replace()`: exact match first, then line-strip normalization (rstrips trailing whitespace per line) |
| No version history | ✅ | Each edit inserts a new row; `_load_latest_report()` uses `ORDER BY created_at DESC`; original is never overwritten |
| No section-aware editing | ❌ | LLM still receives the full document; no section extraction before editing LLM call |
| Concurrent edit race condition | ❌ | No optimistic locking; simultaneous edits from two tabs still race |

---

## 7. Report Generation

### 7.1 Report Writer (`tools/report_writer.py`)

| Issue | Status | Notes |
|---|---|---|
| No structured output enforcement | ⚠️ | Two-call generation (Flash-Lite for structural, Flash for analytical); `_validate_sections()` appends stubs for missing headings; no Pydantic schema enforcement via `.with_structured_output()` |
| 9000-char truncation of analysis JSON | ✅ | Truncation removed; full analysis JSON passed to both LLM calls |
| Disclaimer checked twice, differently | ⚠️ | `_enforce_disclaimer()` in `report_writer.py` handles it programmatically; `editor.py` still checks for the exact string as a secondary guard |
| No quality scoring of output | ❌ | No word count or section length threshold validation |
| Module-level `_primary_llm` state | ❌ | `_primary_llm = None` and `_sub_llm = None` are module-level globals; not thread-safe |

---

## 8. Content Security

### 8.1 Sanitizer (`core/sanitizer.py`)

| Issue | Status | Notes |
|---|---|---|
| Regex injection filter evadable by homoglyphs | ✅ | Unicode NFKC normalization applied before all pattern matching |
| High false-positive rate | ❌ | Patterns still broad; context-aware filtering not implemented |
| Static canary token | ✅ | `secrets.token_urlsafe(16).upper()` per process; different value every restart |
| Canary only checked post-generation | ❌ | `check_canary()` called only on final `report_markdown` in `editor.py`; not checked after intermediate LLM calls |
| Extraction fallback leaks raw text | ✅ | Fallback returns `key_facts=[]`; no raw content passes through |

---

## 9. Editor Agent (`agents/editor.py`)

| Issue | Status | Notes |
|---|---|---|
| Grounding check false negatives | ✅ | `_NUMERIC_PATTERN` handles `$1,234`, `(4,200)`, `$4.17T`, SI suffixes; `_clean_numeric()` strips commas/parens/dollar signs/suffixes |
| Tolerance is flat 1.5% | ✅ | Tiered: 2% for percentages, 5% for values > 1M, 1.5% otherwise |
| SOP failure is binary | ❌ | Each key present/absent with no weighted scoring; one missing optional field fails the entire SOP |
| Disclaimer is hardcoded string | ⚠️ | `_enforce_disclaimer()` in report_writer appends programmatically; editor still checks exact string "This is not financial advice" as a secondary guard |
| No validation of report completeness | ⚠️ | `_validate_sections()` in report_writer.py catches missing headings; no word-count thresholds |

---

## 10. Charts System

### 10.1 Charts Module (`charts/`)

| Issue | Status | Notes |
|---|---|---|
| No intraday data | ❌ | All charts use daily/weekly closes; 1m/5m/15m intervals not supported |
| No analyst price targets on charts | ✅ | Candlestick: mean target (dashdot line) + high/low shaded band from `fundamentals.analyst_target_*` |
| No volume profile | ❌ | Volume bars show total volume; horizontal volume-at-price histogram not implemented |
| Hardcoded P/E coloring thresholds | ❌ | >20% premium → red, >10% discount → green, otherwise blue; percentile-based coloring not implemented |

---

## 11. PageIndex System

### 11.1 PageIndex (`pageindex/`)

| Issue | Status | Notes |
|---|---|---|
| No re-ranking step | ❌ | RRF is the final ranking; no cross-encoder pass (BGE-Reranker, Cohere Rerank) |
| No query expansion | ✅ | HyDE: Flash-Lite generates a hypothetical passage; that embedding is used for vector search; FTS uses original query |
| No chunking strategy for long pages | ❌ | A 10,000-char PDF page is embedded as one 768-dim vector; sentence-level chunking not implemented |
| Embeddings not refreshed on model change | ❌ | No `model_version` column; incompatible embeddings not detected on model upgrade |

---

## Remaining Work — Priority Matrix

| # | Component | Issue | Severity | Effort |
|---|---|---|---|---|
| 1 | `core/llm.py` | Hardcoded model names — no model registry | HIGH | Low |
| 2 | `core/sanitizer.py` | Canary checked only at final output | HIGH | Low |
| 3 | `agents/quant_analyst.py` | Brittle markdown-fence stripping still present as fallback; no true `.with_structured_output()` | MEDIUM | Low |
| 4 | `tools/benchmark_lookup.py` | No fuzzy sector matching | MEDIUM | Low |
| 5 | `memory/memory_manager.py` | Summary truncation at 3000 chars before LLM call | MEDIUM | Low |
| 6 | `memory/memory_manager.py` | Hardcoded `limit=2` summaries in context | MEDIUM | Low |
| 7 | `agents/editor.py` | SOP failure is binary (no weighted scoring) | MEDIUM | Low |
| 8 | `agents/comparison_agent.py` | 4000/2000-char truncation of comparison data | MEDIUM | Low |
| 9 | `core/budget_tracker.py` | Single 80% daily threshold; no per-RPM-minute warning granularity | MEDIUM | Low |
| 10 | `tools/report_writer.py` | Module-level LLM globals; no Pydantic output schema | MEDIUM | Medium |
| 11 | `memory/long_term.py` | No preference versioning; no time-decay for summaries | MEDIUM | Medium |
| 12 | `agents/orchestrator.py` | No agent-level retry; no per-ticker parallelism | MEDIUM | Medium |
| 13 | `pageindex/retriever.py` | No cross-encoder re-ranking | LOW-MEDIUM | Medium |
| 14 | `pageindex/pipeline.py` | No sentence-level chunking for long pages | LOW-MEDIUM | Medium |
| 15 | `agents/quant_analyst.py` | No DCF; no scenario analysis | LOW-MEDIUM | High |
| 16 | `tools/calculator.py` | No financial formula library (pv, npv, irr) | LOW | Medium |
| 17 | `charts/` | No intraday data; no volume profile; hardcoded P/E thresholds | LOW | Low-Medium |
| 18 | `agents/refinement_handler.py` | No concurrent edit protection; no section-aware editing | LOW | Medium |

---

## What Was Closed (25 of 34 sub-issues from original audit)

| Original # | Component | What was fixed |
|---|---|---|
| 1 | `web_search.py` | TTL reduced from 4h to 1h |
| 2a | `core/llm.py` | Half-open circuit breaker with probe recovery |
| 2b | `core/llm.py` | `asyncio.wait_for(120s)` timeout on async calls |
| 3a | `core/sanitizer.py` | Dynamic per-process canary token |
| 3b | `core/sanitizer.py` | NFKC Unicode normalization before regex |
| 3c | `core/sanitizer.py` | Safe fallback — empty `key_facts`, no raw content leak |
| 4 | `quant_analyst.py` | Risk metrics: Sharpe, Sortino, max drawdown, beta, volatility |
| 5 | `benchmark_lookup.py` | Live Damodaran; 7 multiples (PE, forward PE, EV/EBITDA, P/B, P/S, margin, beta) |
| 6 | `yahoo_finance.py` | 7 data types; adjusted close; cash flow; dividend history; analyst targets |
| 7 | `editor.py` | Financial number parser; tiered tolerance |
| 8 | `editor.py` / `report_writer.py` | 9000-char truncation removed; two-call generation; section validation |
| 9 | `memory/long_term.py` | Semantic search via cosine similarity on stored embeddings |
| 10 | `orchestrator.py` | Conditional early-exit routing on empty data / rate-limited state |
| 12 | `memory/short_term.py` | Turn-pair windowing; JSON-density token estimate |
| 13 | `refinement_handler.py` | Fuzzy match (line-strip fallback); INSERT versioning |
| 14 | `pageindex/retriever.py` | HyDE query expansion |
| 17 | `yahoo_finance.py` | Silent data degradation — `data_quality` + `degradation_note` in all 7 data types |
| 18 | `yahoo_finance.py` | Corporate events — stock splits in `price_history.corporate_events` |
| 19 | `benchmark_lookup.py` | Fuzzy sector matching — alias dict + `difflib.get_close_matches` fallback |
| 20 | `benchmark_lookup.py` | Geographic segmentation — `geographic_context` block for non-US companies |
| 21 | `quant_analyst.py` | DCF valuation — `_compute_dcf()`: 5-yr FCF projection, WACC via CAPM, intrinsic value per share |
| 22 | `quant_analyst.py` | Scenario analysis — `_compute_scenarios()`: P/E×EPS, analyst consensus, DCF (bear/base/bull) |
| 23 | `quant_analyst.py` | LLM parsing brittle — `with_structured_output(_SOPOutput)` via Gemini JSON mode |
| 24 | `quant_analyst.py` | Hardcoded sector taxonomy — `_YFINANCE_TO_GICS` removed; raw sector passed to benchmark_lookup |
| 25 | `core/llm.py` | `with_structured_output()` added to `RateLimitFallbackLLM` (same pattern as `bind_tools`) |

*9 sub-issues remain across the priority matrix above.*
