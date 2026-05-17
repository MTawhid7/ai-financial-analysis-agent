# System Audit Report: Implementation Status

**Original audit date:** 2026-05-15  
**Last updated:** 2026-05-17 (updated post-architectural-refactoring to reflect Phases 0–4 changes)  
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
| No financial formula library | ✅ | New `financial_formulas` tool: NPV, IRR, PV, FV, CAGR, WACC, payback_period, ROI — pure-Python function registry; domain warnings (IRR>500%, CAGR>100%); registered in Manager LLM |
| Rounding inconsistency | ✅ | Optional `format` hint on calculator: "percent"→"14.22%", "currency"→"$4.17T", "ratio"→"28.5×", "integer"→"15,726,000,000"; auto-detection by magnitude; structured JSON return when format specified |
| No unit tracking | ✅ | Optional `context` dict on calculator for named variable binding (`{"market_cap_usd": 3e12, "revenue_usd": 4e11}`); unit suffixes document intent; context keys validated (no leading `_`) |
| No result range validation | ✅ | `_validate_result()` catches `inf`/`nan` and returns ToolError; warns on magnitude > 1e15 or < 1e-10; financial functions validate domain (IRR convergence, CAGR start > 0, etc.) |

**Severity: MEDIUM — Narrow scope but safe; financial functions would add value.**

---

## 3. Research & Retrieval

### 3.1 Researcher Agent (`agents/researcher.py`)

| Issue | Status | Notes |
|---|---|---|
| Hardcoded year in news query | ✅ | `_current_year()` → `datetime.utcnow().year` |
| No adaptive iteration | ✅ | Phase 3: two-phase concurrent fetch in `data/yahoo/__init__.py` — Phase 1 fetches 3 core types, Phase 2 fetches 4 extended only if ≥1 core succeeded. Early exit is structural, not a counter. |
| No multi-source reconciliation | ⚠️ | No free second data source exists; `data_quality` + `degradation_note` + `data_timestamp` on every DataResult surfaces quality transparency. True reconciliation requires paid feeds. |
| Token estimation | ✅ | Phase 3: `researcher.py` now calls `estimate_tokens()` from `core/utils.py` (JSON-density-aware 2–4 chars/token). |
| No data freshness labeling | ✅ | Phase 2: all 7 DataResult types carry `data_timestamp` (UTC ISO 8601) and `data_quality`/`degradation_note`; researcher surfaces `PARTIAL` types as `researcher_gaps`. |

**Section 3.1 is fully resolved.** The only honest limitation (no multi-source reconciliation) is a free-tier constraint, not an implementation gap.

---

### 3.2 Web Search Tool (`tools/web_search.py`)

| Issue | Status | Notes |
|---|---|---|
| 4-hour cache on financial news | ✅ | `TTL_WEB_SEARCH = 1h`; wired correctly |
| No source credibility scoring | ✅ | Phase 2: `data/search/credibility.py` — 30+ domains mapped to tiers 1–3; `TavilySearchClient` sorts results by `(source_tier ASC, score DESC)` before returning |
| No date filtering on results | ✅ | Phase 2: `TavilySearchClient(days=settings.search_days_window)` — default 90 days; passed to Tavily API `days` param |
| Sanitizer fallback leaks raw text | ✅ | Fallback returns `key_facts=[]`; no raw content passes through |
| `search_depth="basic"` is silent downgrade | ✅ | Phase 2: `TavilySearchClient.search()` retries with `_reformulate_query()` (removes year + narrow phrases) on zero results; logs WARNING before retry |

**Section 3.2 is fully resolved.**

---

## 4. LLM Infrastructure

### 4.1 LLM Client (`core/llm/`)

| Issue | Status | Notes |
|---|---|---|
| Circuit breaker window too short / no recovery | ✅ | 5×429/60s threshold; half-open state; `probe_succeeded()` / `probe_failed()` |
| No recovery from Flash-Lite | ✅ | Probe success resets breaker to CLOSED |
| Streaming/non-streaming mismatch | ⚠️ | Primary `streaming=True`, fallback `streaming=False` — intentional; `content_to_str()` normalises both. Acceptable as-is. |
| Hardcoded model names | ✅ | Phase 0: `settings.llm_primary_model` / `settings.llm_fallback_model` — fully env-overridable via `LLM_PRIMARY_MODEL` / `LLM_FALLBACK_MODEL` |
| No per-call timeout | ✅ | `asyncio.wait_for(coro, timeout=settings.llm_call_timeout_s)` in `ainvoke()` |
| No token counting | ❌ | Request-based budgeting only; Gemini API doesn't expose token counts in real-time without an extra call. Low priority for free tier. |

**4.1 is substantially resolved.** Only token counting remains and is genuinely hard on free tier.

---

### 4.2 Budget Tracker (`core/budget_tracker.py`)

| Issue | Status | Notes |
|---|---|---|
| Counts requests, not cost | ❌ | Request-based; token-weighted cost model would require token counts from API responses (not easily available free tier) |
| Daily budget, not per-minute | ✅ | Phase 1: `_RpmBucket` tracks rolling 60s window; warns when limit exceeded; `primary_rpm_current`/`sub_rpm_current` in `get_stats()` |
| 80% alert too late / only one threshold | ✅ | Phase 0: three-tier via `settings`: `llm_budget_soft_warn_pct=0.60`, `llm_budget_warn_pct=0.80`, `llm_budget_defer_pct=0.95`; `budget_tracker.py` reads all three |
| No per-tool budgeting | ❌ | All calls share one pool; tracking per-tool would require instrumenting each call site |
| Cache hit recording is no-op | ❌ | `record_cache_hit()` increments counter but doesn't reduce quota estimate |

**4.2 is mostly resolved.** Remaining items (token counting, per-tool budgeting, cache adjustment) are low-priority operational refinements.

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

## Remaining Work — Recommended Implementation Order

### What sections 3, 4 look like after the architectural refactoring

**Sections 3.1 and 3.2 are fully resolved** by the Phase 2–3 data-layer refactoring.
Tackling them now would only confirm they are done, not produce improvements.

**Section 4 is substantially resolved** by Phase 0 (settings.py) and Phase 1 (LLM package).
Only token counting (requires real-time API token metadata) and minor budget refinements remain.

**The recommended order differs from section-by-section reading because severity does not follow section order.**

---

### Recommended sequence

| Priority | Section | Specific Issue | Severity | Why Now |
|---|---|---|---|---|
| **1** | **8.1** Sanitizer | Canary only checked at final output | **HIGH** | 30-min fix; highest-severity open issue; adds defence-in-depth to every pipeline run |
| **2** | **6.2** Comparison Agent | 4000/2000-char truncation; no table validation | MEDIUM | Low effort; structured extraction follows the same pattern used in quant_analyst.py |
| **3** | **9** Editor Agent | SOP failure is binary | MEDIUM | Low effort; weighted scoring uses the same SOP dict that already exists |
| **4** | **5.3** Memory Manager | Summary truncation (3000 chars); hardcoded `limit=2` | MEDIUM | Quick config-driven fix; immediately improves memory context quality |
| **5** | **5.1** Long-Term Memory | Preference versioning; time-decay for summaries | MEDIUM | Medium effort; builds naturally on the MemoryBackend Protocol from Phase 4 |
| **6** | **7.1** Report Writer | Module-level LLM globals | MEDIUM | Phase 4 DI pattern makes this straightforward now (inject via NodeConfig) |
| **7** | **5.2** Short-Term Memory | Message priority; hierarchical summarisation | MEDIUM | Medium effort; improves long-conversation coherence |
| **8** | **6.1** Orchestrator | No agent-level retry; SQLite checkpoint | MEDIUM | Medium effort; Postgres checkpointer uses `settings.database_url` already wired |
| **9** | **6.3** Refinement Handler | Section-aware editing; concurrent edit lock | LOW | Concurrent edit is a race condition only under multi-tab use |
| **10** | **11** PageIndex | Cross-encoder re-ranking; sentence chunking | LOW-MEDIUM | Meaningful precision gain but requires additional model or chunking pass |
| **11** | **10** Charts | Intraday data; volume profile; P/E thresholds | LOW | Incremental; intraday requires yfinance intraday quotas |
| **12** | **4.2** Budget Tracker | Token counting; per-tool budgeting | LOW | Hard on free tier (no real-time token API); low practical impact |

### What to skip (already done, just not marked)

The following issues appear open in the section tables but are already resolved
by the architectural refactoring — **do not spend time on them**:

| Apparent open issue | Actually resolved by |
|---|---|
| 3.1 Adaptive iteration | Phase 3: two-phase concurrent fetch with structural early exit |
| 3.1 Token estimation | Phase 3: `estimate_tokens()` from `core/utils` in researcher |
| 3.1 Data freshness labeling | Phase 2: `data_timestamp` + `data_quality` on all 7 DataResult types |
| 3.2 Source credibility scoring | Phase 2: `data/search/credibility.py` + `TavilySearchClient` sort |
| 3.2 Date filtering | Phase 2: `TavilySearchClient(days=settings.search_days_window)` |
| 3.2 Silent downgrade | Phase 2: `TavilySearchClient._invoke()` retry + `_reformulate_query()` |
| 4.1 Hardcoded model names | Phase 0: `settings.llm_primary_model` / `LLM_PRIMARY_MODEL` env var |
| 4.2 80% single threshold | Phase 0: three-tier 60%/80%/95% in `settings`; `budget_tracker.py` reads all three |

---

## What Was Closed (34 of 34 sub-issues from original audit + architectural refactoring)

### Original 34 audit sub-issues

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
| 26 | `tools/financial_formulas.py` | New tool: NPV, IRR, PV, FV, CAGR, WACC, payback_period, ROI — function registry pattern |
| 27 | `tools/calculator.py` | `format` hint + structured JSON return (percent/currency/ratio/integer/auto) |
| 28 | `tools/calculator.py` | `context` dict for named variable binding (unit-context documentation) |
| 29 | `tools/calculator.py` | `_validate_result()`: inf/nan → ToolError; large-magnitude soft warning |

### Architectural refactoring (Phases 0–3)

| # | Component | What was refactored |
|---|---|---|
| A | `config/settings.py` (new) | Single-source Settings class (pydantic-settings); all 40+ constants moved from 16 files; env-overridable |
| B | `core/llm/` (split) | Package with protocols.py, circuit_breaker.py, gemini.py, registry.py; no module-level singletons |
| C | `core/utils.py` (new) | safe_float, null_result, assess_data_quality, estimate_tokens, extract_domain — extracted from duplicated locations |
| D | `data/` (new package) | data/yahoo/* (7 modules), data/market/*, data/benchmark/*, data/search/* |
| E | `agents/researcher.py` | Concurrent two-phase fetch via asyncio.gather; adaptive early-exit; TavilySearchClient |
| F | `tools/` (thinned) | All tools now delegate to data/ layer; no business logic in tool wrappers |
| G | `memory/protocol.py` (new) | MemoryBackend Protocol + InMemoryBackend for test isolation |
| H | `tests/conftest.py` (new) | Shared fixtures: test_settings, mock_llm, mock_cache, in_memory_backend |

*475 tests passing (was 356 at start of refactoring session).*
