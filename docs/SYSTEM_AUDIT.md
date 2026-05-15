# System Audit Report: Naive & Underdeveloped Implementations

**Date:** 2026-05-15  
**Scope:** Full pipeline — data layer, LLM infrastructure, memory, orchestration, security, charts, PageIndex  
**Purpose:** Identify components where current implementations are simplistic relative to industry standards, as a basis for prioritised redesign.

---

## 1. Financial Data Layer

### 1.1 Yahoo Finance Tool (`tools/yahoo_finance.py`)

**Current:** Single-source yfinance wrapper returning a static snapshot of fundamentals and a 5-year weekly close series.

| Issue | Detail | Industry Standard |
|---|---|---|
| **No adjusted close prices** | Uses raw `Close` throughout. Stock splits and dividends distort historical analysis — a 4:1 split makes the price look like it fell 75%. | Bloomberg, FactSet always distinguish `Close` vs `Adj Close`; adjusted is the default for return calculations |
| **Stale `info` dict** | `stock.info` is cached by Yahoo's CDN and can be 15–24 hours old. No timestamp validation or freshness check | Real-time feeds (IEX, Polygon.io) carry explicit `lastUpdated` timestamps and warn on stale data |
| **Missing cash flow data** | Operating CF, free cash flow, capex are absent despite being core to DCF valuation and quality-of-earnings analysis | Every professional terminal exposes cash flow as a first-class data type alongside income statement |
| **No forward estimates** | Only trailing P/E; no forward P/E, consensus EPS estimates, revenue estimates, or analyst target prices | FactSet Estimates, Bloomberg Consensus pull from 30+ sell-side firms for a forward view |
| **Silent data degradation** | Falls back from 5y → 2y → 1y without notifying the agent or the user | Production data pipelines surface data quality grades ("FULL", "PARTIAL", "ESTIMATED") to consumers |
| **No dividend data** | Yield, ex-dates, payout ratio absent | Critical for income-focused analysis; yield-on-cost changes with price but payout ratio doesn't |
| **52-week window is approximate** | `prices[-252:]` assumes exactly 252 trading days per calendar year — off by 2–5 days | Use `.loc[date - timedelta(365):]` for a true 52-week window |
| **No corporate event timeline** | No earnings dates, stock split history, share buyback announcements | Bloomberg Events timeline overlays these on price charts and warns analysts before earnings |

**Severity: HIGH** — Stale data and missing cash flow materially affect report quality.

---

### 1.2 Benchmark Lookup (`tools/benchmark_lookup.py`)

**Current:** A static JSON file loaded once at import time containing approximate 2024 sector P/E averages.

| Issue | Detail | Industry Standard |
|---|---|---|
| **Static data, zero refresh** | Values are frozen at one point in time. P/E multiples shift dramatically in rate cycles (S&P P/E: 38x in 2021, 18x in late 2022) | Damodaran's dataset is updated annually; Bloomberg recomputes sector multiples daily |
| **Only P/E** | Lacks EV/EBITDA, Price/Book, Price/Sales, EV/Revenue, FCF yield by sector | A proper relative valuation uses 4–6 multiples, not one |
| **No fuzzy sector matching** | Exact case-insensitive match only; "Consumer Discretionary" vs "Consumer" silently returns an error | Production systems use fuzzy string matching (edit distance) with a fallback hierarchy |
| **No geographic segmentation** | Global sector benchmarks are very different from US-only ones. A Chinese tech company benchmarked against US tech is misleading | Bloomberg allows filtering by region, country, and GICS sub-sector |

**Severity: HIGH** — Single stale multiple renders relative valuation unreliable.

---

## 2. Quantitative Analysis

### 2.1 Quant Analyst (`agents/quant_analyst.py`)

**Current:** Calculates CAGR via string-interpolated numexpr expression and P/E premium against one static benchmark. Sends analysis to Flash for bull/bear case generation.

| Issue | Detail | Industry Standard |
|---|---|---|
| **Only price CAGR** | 5-year price return without dividends is Total Return Shareholder Value (TRSV) only if dividends are zero | Analysts use Total Return (price + dividends reinvested), not price-only CAGR |
| **No risk-adjusted metrics** | No Sharpe ratio, Sortino ratio, max drawdown, or beta. "Strong 15% CAGR" looks different at Sharpe 0.3 vs 1.8 | Risk-adjusted return is the primary performance metric in institutional analysis |
| **No DCF framework** | Valuation is entirely multiple-based. No intrinsic value estimate | A complete buy-side note includes at least a back-of-envelope DCF with explicit assumptions |
| **Bull/bear cases are LLM-generated narrative** | The LLM invents bull/bear scenarios without checking them against historical volatility, analyst estimates, or stress scenarios | Buy-side bull/bear cases are anchored to specific catalysts with quantified price targets |
| **LLM response parsing is brittle** | Manual markdown fence stripping: `.split("```", 2)[1]`. If the LLM changes formatting, parsing silently fails and cases are empty | Use structured outputs (JSON mode / tool-use) to enforce schema; never parse free-form LLM text for structured data |
| **Hardcoded sector taxonomy** | Static `_YFINANCE_TO_GICS` dict; if yfinance renames a sector, falls back to the original name silently | GICS is the standard — map to it definitively, not aspirationally |
| **No scenario analysis** | Single-point estimate only. No sensitivity analysis on key assumptions | Scenario analysis (base/bull/bear P/E × earnings range) is standard in equity research |

**Severity: HIGH** — Missing risk adjustment means the agent consistently presents returns without context.

---

### 2.2 Calculator Tool (`tools/calculator.py`)

**Current:** AST-validated numexpr evaluator. Safe but minimal.

| Issue | Detail | Industry Standard |
|---|---|---|
| **No financial formula library** | Must construct every formula as a string expression. No `pv()`, `fv()`, `npv()`, `irr()` | Excel's financial functions are the baseline expectation for any financial calculator |
| **Rounding inconsistency** | Hardcoded 6 decimal places; analysts expect 2 for percentages, 0 for share counts, 2 for currency | Professional calculation engines carry full precision internally and format on output |
| **No unit tracking** | `200 * 1e6 / 100` is correct only if inputs are in the same unit. No dimension checking | QuantLib and Bloomberg carry units through calculations |
| **No result range validation** | Returns `1e308` (float overflow) silently; division-by-zero produces `nan` without warning | Any value outside plausible financial range should raise with explanation |

**Severity: MEDIUM** — Current scope is narrow but safe; adding financial functions would be high value.

---

## 3. Research & Retrieval

### 3.1 Researcher Agent (`agents/researcher.py`)

**Current:** LangGraph ReAct loop calling yfinance (3×), web search (1×), benchmark lookup. Max 5 iterations.

| Issue | Detail | Industry Standard |
|---|---|---|
| **Hardcoded year in news query** | `"2024"` is baked into the search query; will produce zero results in 2026 | Dynamic `datetime.now().year` or a recency parameter |
| **No adaptive iteration** | Hard stop at 5 regardless of data completeness; also never terminates early if data is already complete | ReAct agents should terminate when a coverage threshold is met, not at a fixed iteration count |
| **No multi-source reconciliation** | Takes the first tool result for each data type. If yfinance returns stale data and web search has fresher info, only yfinance is used | Bloomberg reconciles from 50+ sources and surfaces discrepancies |
| **Token estimation** | `len(string) // 4` for all text. Off by 2–3× for JSON, markdown, non-ASCII | Use `tiktoken` or the model provider's tokenization endpoint |
| **No data freshness labeling** | No `data_as_of` timestamp propagated to the report | Every Bloomberg terminal quote shows "Last Updated" |

**Severity: MEDIUM** — The hardcoded year is an imminent bug; adaptive termination would improve quality.

---

### 3.2 Web Search Tool (`tools/web_search.py`)

**Current:** Tavily primary, DuckDuckGo fallback. Results sanitized by regex filter and cached 4 hours.

| Issue | Detail | Industry Standard |
|---|---|---|
| **4-hour cache on financial news** | A 4-hour cache means post-earnings moves, analyst upgrades, and M&A announcements are missed entirely | Financial news should have ≤15-minute TTL; general reference content can be 24h |
| **No source credibility scoring** | Reuters and a stock-forum post are treated identically | Refinitiv and Bloomberg weight sources by publication tier (Tier 1: Reuters/AP; Tier 2: regional; Tier 3: unverified) |
| **No date filtering on results** | Tavily can return 2019 articles as equally relevant | Query should include a `published_after` constraint or post-filter by `date` field |
| **Sanitizer fallback leaks raw text** | When `subllm is None`, extraction returns the first 200 chars of raw HTML | Security bypass: raw injected content reaches the agent if LLM extraction is unavailable |
| **`search_depth="basic"` is silent downgrade** | Code comment says "advanced" returns empty arrays, so always uses basic. No alerting, no retry with different query | Should log a warning and attempt query reformulation before accepting lower-quality results |

**Severity: HIGH** — 4-hour cache for financial news is a correctness issue, not just a performance tradeoff.

---

## 4. LLM Infrastructure

### 4.1 LLM Client (`core/llm.py`)

**Current:** Gemini Flash primary, Flash-Lite fallback. Tenacity retry with exponential jitter. 3×429 circuit breaker within 30 seconds.

| Issue | Detail | Industry Standard |
|---|---|---|
| **Circuit breaker window too short** | 3 errors within 30 seconds trips the breaker. A single 5-second cluster of transient 429s permanently degrades the session to Flash-Lite | Standard circuit breaker windows are 60–120 seconds; recovery probe attempts before permanent downgrade |
| **No recovery from Flash-Lite** | Once tripped, the session never tries Flash again, even if the rate limit has cleared | Proper half-open state: after 60s, probe primary once; if succeeds, close breaker |
| **Streaming/non-streaming mismatch** | Primary uses `streaming=True`, fallback uses `streaming=False`. If primary fails mid-stream, downstream code may break on non-streamed format | Both should use the same output protocol, or a stream adapter wraps the fallback |
| **Hardcoded model names** | `"gemini-3-flash-preview"` — if Google renames or retires the model, the entire system breaks silently | Model registry with version pinning + fallback list: `[primary, fallback_v1, fallback_v2]` |
| **No per-call timeout** | A streaming call that hangs blocks indefinitely | `asyncio.wait_for(call, timeout=120)` with a graceful abort |
| **No token counting** | All rate-limit logic is request-based, but Gemini's actual quota is token-based | Track approximate token usage to prevent quota exhaustion before the API rejects requests |

**Severity: HIGH** — The circuit breaker is too sensitive and has no recovery mechanism.

---

### 4.2 Budget Tracker (`core/budget_tracker.py`)

**Current:** Counts API calls. Warns at 80%. Tracks primary vs sub-LLM calls separately.

| Issue | Detail | Industry Standard |
|---|---|---|
| **Counts requests, not cost** | A 50-token Flash call and a 4000-token Flash call count the same | Cost tracking should be token-weighted: `cost += input_tokens × price_in + output_tokens × price_out` |
| **Daily budget, not per-minute** | Gemini's critical limit is RPM (15/minute free tier), not daily total | Alert at 80% of per-minute RPM, not per-day RPC |
| **80% alert too late** | By the time the warning fires, the next request may already be rate-limited | Alert at 60% (soft warning), 80% (hard warning), 95% (activate caching/deferral mode) |
| **No per-tool budgeting** | Web searches and main reasoning calls share the same pool | Professional ML systems assign per-tool budgets: "max 3 Tavily calls per query" |
| **Cache hit recording is no-op** | `record_cache_hit()` logs but doesn't subtract from the budget pool | A cache hit should free budget for another real call |

**Severity: MEDIUM** — Not a correctness issue currently (free tier), but blocks any real cost management.

---

## 5. Memory System

### 5.1 Long-Term Memory (`memory/long_term.py`)

**Current:** Postgres tables for preferences, analysis summaries, conversations, messages. Retrieval via SQL `LIKE %query%`.

| Issue | Detail | Industry Standard |
|---|---|---|
| **Keyword-only retrieval** | `search_summaries` uses `LIKE %query%` on ticker names and summary text. "profit margins" won't match "earnings quality" | Hybrid retrieval: BM25 for sparse + semantic embeddings for dense; merge with RRF |
| **No preference versioning** | A preference is upserted — old value is gone forever | Track `(key, value, created_at, superseded_at)` to show preference evolution |
| **No preference conflict resolution** | "I prefer aggressive analysis" + "I prefer conservative picks" can coexist. Latest wins, silently | A conflict resolution step should surface the contradiction and ask the user |
| **Summaries never expire or decay** | A 2-year-old analysis summary about a company that has since been acquired is still returned with equal weight | Time-decay scoring: recent memories weighted higher; very old ones flagged as potentially stale |
| **Only 2 summaries returned for context** | Hardcoded `limit=2` in `build_memory_context` | Should be relevance-ranked and size-adaptive (fit as many summaries as the context window allows) |

**Severity: MEDIUM** — Keyword retrieval limits the value of the memory system for complex queries.

---

### 5.2 Short-Term Memory / Context Window (`memory/short_term.py`)

**Current:** Fixed 3000-token budget. Walks backward through messages, truncates older ones.

| Issue | Detail | Industry Standard |
|---|---|---|
| **No real tokenizer** | `len(content) // 4` char-to-token estimate. Actual: JSON is ~2 chars/token; code is ~3; prose is ~4; non-ASCII is ~1–2 | Use `langchain_google_genai`'s `count_tokens()` method or Gemini's native token counter |
| **No message priority** | A critical error message from turn 1 and a filler "OK." from turn 10 are equally truncated | Priority-based windowing: system messages → recent assistant responses → recent user messages → older messages |
| **Drops messages wholesale** | Either a message is fully included or fully excluded | Hierarchical summarization: condense old messages to a summary paragraph instead of dropping them |
| **Breaks mid-turn pairs** | Might include an assistant message but exclude the user message that prompted it | Always keep user+assistant turn pairs together; split at turn boundaries only |

**Severity: MEDIUM** — Inaccurate token estimation causes either context overruns or unnecessary truncation.

---

### 5.3 Memory Manager (`memory/memory_manager.py`)

**Current:** Regex pre-filter + LLM extraction for preferences. Single Flash-Lite call for analysis summary. Hardcoded 2 summaries for context injection.

| Issue | Detail | Industry Standard |
|---|---|---|
| **Regex pre-filter is brittle** | `_PREFERENCE_SIGNALS` is a static pattern list. Novel phrasings like "my investment thesis is quality over growth" are missed | LLM-based intent detection for preference signals, not regex |
| **No structured output enforcement** | Preference extraction uses free-form JSON parsing. Malformed JSON is silently swallowed | Use Gemini's JSON response mode or LangChain structured output — guarantees schema conformance |
| **Preference semantics ignored** | "conservative" and "risk-averse" stored as different entries | Preference normalization: map synonyms to canonical values; detect contradictions |
| **Summary truncation at 3000 chars** | Only the first 3000 chars of a report are summarised. For a multi-ticker analysis, that might be only the first ticker | Use the full report with hierarchical summarization already built in `parsers/_summarise.py` |
| **No context relevance ranking** | 2 most-recent summaries injected without checking relevance to the current query | Rank summaries by semantic similarity to current query before injecting |

**Severity: MEDIUM** — Regex preference extraction and fixed summary count limit personalisation quality.

---

## 6. Orchestration & Pipeline

### 6.1 Orchestrator (`agents/orchestrator.py`)

**Current:** Fixed LangGraph DAG: `Researcher → Quant → Editor`. No branching. SQLite checkpointing.

| Issue | Detail | Industry Standard |
|---|---|---|
| **Hardcoded linear pipeline** | Every query runs all three agents regardless of complexity. A "what is Apple's current P/E?" shouldn't run the full pipeline | Conditional routing: skip research for simple factual queries; skip editor for internal comparison tables |
| **No agent-level retry** | If the Quant analyst fails, the whole pipeline fails. The researcher's work is lost | Node-level retry with state preservation: re-run only the failed node with the existing state |
| **No parallelism for multi-ticker** | AAPL and MSFT are researched sequentially. At 15 RPM free tier, each ticker takes 60–120s | Pipeline can be parallelized per ticker if RPM allows; or pipelined: start MSFT quant while AAPL editor runs |
| **Error types collapse to same output** | `PartialStateError`, `CircuitBreakerError`, generic `Exception` all produce the same generic error response | Graduated degradation: circuit breaker → return whatever was computed; timeout → return partial |
| **SQLite checkpoint in production** | LangGraph's SQLite checkpointer is single-process only | Use the Postgres checkpointer (`langgraph-checkpoint-postgres`) in production |

**Severity: MEDIUM** — Sequential multi-ticker is the most visible user-facing limitation.

---

### 6.2 Comparison Agent (`agents/comparison_agent.py`)

**Current:** Runs full 3-agent pipeline for every ticker, then calls Flash to write a comparison table.

| Issue | Detail | Industry Standard |
|---|---|---|
| **Full pipeline per ticker for comparison** | A head-to-head comparison of 4 stocks runs 4 full pipelines + a comparison call | Cache single-ticker analyses; a comparison request should combine cached analyses |
| **Hard-truncates data at 4000/2000 chars** | Important metrics for the 2nd and 3rd tickers may be in the truncated tail | Structured extraction: pull only the key comparison metrics explicitly, not raw truncated text |
| **No validation of output table** | Comparison result is whatever the LLM returns. A 3-ticker request might get a 2-row table | Post-validate: parse markdown table, check all requested tickers are present, flag missing rows |
| **Comparison dimensions fixed** | Always the same columns (price, P/E, CAGR, etc.) | Let the user specify what dimensions to compare: "compare their cloud revenue growth" |

**Severity: MEDIUM** — The redundant pipeline execution is the most expensive issue.

---

### 6.3 Refinement Handler (`agents/refinement_handler.py`)

**Current:** LLM outputs `old_string` + `new_string`; literal `str.replace(old, new, 1)`.

| Issue | Detail | Industry Standard |
|---|---|---|
| **No fuzzy matching fallback** | If the LLM quotes the text with a minor difference (extra space, different quote mark), fails after only 2 attempts | Edit distance / difflib approximate matching: find the closest passage; confirm before applying |
| **No version history** | Each edit overwrites the previous version permanently | Git-style versioning: every edit is a new version; rollback is always available |
| **No section-aware editing** | The LLM receives the entire document and must return exact text. Error-prone for long reports | Parse the document into sections first; send only the relevant section to the editing LLM |
| **Concurrent edit race condition** | If two browser tabs trigger edits simultaneously, the second write wins silently | Optimistic locking: include a `version_hash` in the edit request; reject if document was modified |

**Severity: LOW-MEDIUM** — Works for single-user demo; would fail under real concurrent use.

---

## 7. Report Generation

### 7.1 Report Writer (`tools/report_writer.py`)

**Current:** Single Flash call with a long prompt dictating section structure. Returns markdown string.

| Issue | Detail | Industry Standard |
|---|---|---|
| **No structured output enforcement** | The LLM is asked to "follow this format" in natural language. Section order and numeric format are advisory, not enforced | Use structured generation: define a `ReportSchema` Pydantic model; use Gemini JSON mode to fill it; render from schema |
| **9000-char truncation of analysis JSON** | The input to the report writer is truncated at 9000 chars. For 3+ tickers, important data may be dropped | Extract the key fields for each ticker explicitly before passing to the LLM |
| **Disclaimer is checked twice, differently** | `editor.py` checks for `"This is not financial advice"` (exact string); `report_writer.py` has its own disclaimer logic | Append disclaimer as a post-processing step after report generation, not in the LLM prompt |
| **No quality scoring of output** | A 200-word report and an 800-word report both pass the editor | Output quality check: verify section count, approximate word count, presence of key metrics |
| **Module-level `_primary_llm` state** | `configure()` sets a module-level variable. Not thread-safe in async context | Inject the LLM as a parameter to the function, not a global |

**Severity: MEDIUM** — Structured output would guarantee section completeness.

---

## 8. Content Security

### 8.1 Sanitizer (`core/sanitizer.py`)

**Current:** Regex pattern list for injection detection, LLM-based extraction for web content, static canary token.

| Issue | Detail | Industry Standard |
|---|---|---|
| **Regex injection filter is evadable** | Homoglyphs (`ΙgnоrΕ` looks like "Ignore" but fails the regex), encoding tricks, and split phrases bypass pattern matching | Unicode normalisation (NFKC) before matching; semantic injection detection via a small classifier |
| **High false-positive rate** | `(output\|reveal\|expose\|print)\s+(your\s+)?(api\s+key...)` fires on security-awareness articles legitimately discussing API key protection | Context-aware filtering: reject only if injection language appears in imperative-to-AI context |
| **Static canary token** | `"CANARY_XQ7Z_SENTINEL"` is hardcoded. An adversary who reads the codebase can craft an attack that avoids this string | Session-generated canary tokens: `str(uuid.uuid4())` injected into system prompt; if it appears in tool output, alert |
| **Canary only checked post-generation** | The canary is only checked on final agent output, not on intermediate LLM calls | Check canary after every LLM call in the pipeline, not just the final response |
| **Extraction fallback leaks raw text** | When LLM extraction is unavailable, the first 200 chars of raw web content go directly to the agent | Hard fail (return empty content, log warning) rather than leak unvalidated content |

**Severity: HIGH** — Static canary and regex-only filtering are known-weak patterns.

---

## 9. Editor Agent (`agents/editor.py`)

**Current:** SOP checklist (5 hardcoded keys), grounding check (regex number extraction with tolerance), disclaimer injection.

| Issue | Detail | Industry Standard |
|---|---|---|
| **Grounding check has false negatives** | Returns `True` (ungrounded) for unparseable values; returns `True` for exact zeros; doesn't handle thousands separators (`1,234`) or parenthetical negatives (`(4,200)`) | Number extraction should use a proper financial number parser |
| **Tolerance is flat 1.5%** | A $1B market-cap company off by 1.5% is a $15M error — material. The same tolerance applied to a percentage is too loose | Scale-relative tolerance configured per metric type |
| **SOP failure is binary** | One missing data field fails the entire SOP check | Weighted SOP: critical fields (price, market cap) fail; optional fields (analyst coverage) warn |
| **Disclaimer is hardcoded string** | Checks for exact `"This is not financial advice"` — fragile to LLM phrasing variations | Append the standard disclaimer programmatically rather than verifying the LLM added it correctly |
| **No validation of report completeness** | A 3-line report passes the editor if it has all SOP keys | Minimum section length thresholds per section type |

**Severity: MEDIUM** — Grounding check false negatives mean hallucinated numbers can pass review.

---

## 10. Charts System

### 10.1 Charts Module (`charts/`)

Already substantially improved (13 chart types, interactive, date ranges, Bollinger Bands, combined panels). Remaining gaps:

| Issue | Detail | Industry Standard |
|---|---|---|
| **No intraday data** | All charts use daily or weekly closing prices. Intraday moves are invisible | TradingView, Bloomberg show 1m/5m/15m/1h candles for recent periods |
| **No analyst price targets on charts** | No visual overlay of consensus price targets | Bloomberg displays analyst targets as horizontal bands on price charts |
| **No volume profile** | Volume bars show total volume; Volume Profile shows price levels where volume concentrated | Volume Profile (horizontal histogram of volume at each price level) is standard on professional charts |
| **Hardcoded P/E coloring thresholds** | Red if >20% premium, green if >10% discount, blue otherwise. Thresholds are arbitrary | Percentile-based coloring: red if P/E is in top decile vs sector history, green if bottom decile |

**Severity: LOW** — Charts are already professional-grade; these are incremental enhancements.

---

## 11. PageIndex System

### 11.1 PageIndex (`pageindex/`)

Just built; architecture is sound. Early-stage gaps:

| Issue | Detail | Industry Standard |
|---|---|---|
| **No re-ranking step** | RRF merges vector + FTS results but doesn't apply a cross-encoder re-ranker | BGE-Reranker, Cohere Rerank, or a small cross-encoder pass on top-K candidates improves precision by 15–30% |
| **No query expansion** | User queries are embedded as-is. Short, ambiguous queries retrieve poorly | HyDE (Hypothetical Document Embedding): ask the LLM to generate a hypothetical answer, embed that as the query |
| **No chunking strategy for long pages** | A 10,000-char PDF page is embedded as one vector. The embedding model's effective range is ~512–768 tokens | Sentence-level chunking with parent-child retrieval: store small chunks for retrieval; return full page as context |
| **Embeddings not refreshed on model change** | If Gemini changes `text-embedding-004`, all stored embeddings become incompatible | Store the `model_version` alongside each embedding; detect version mismatches and trigger re-embedding |

**Severity: LOW-MEDIUM** — Re-ranking would meaningfully improve retrieval precision.

---

## Priority Matrix

| # | Component | Issue | Severity | Implementation Effort |
|---|---|---|---|---|
| 1 | `web_search.py` | 4-hour TTL for financial news | **CRITICAL** | Very Low (change one constant, add per-type TTL) |
| 2 | `core/llm.py` | Circuit breaker has no recovery; streaming mismatch | **HIGH** | Low (fix half-open state + timeout) |
| 3 | `core/sanitizer.py` | Static canary; regex-only injection filter; extraction fallback leaks | **HIGH** | Medium (session canary + Unicode normalisation) |
| 4 | `quant_analyst.py` | No risk-adjusted metrics; brittle LLM parsing | **HIGH** | Medium (add Sharpe/Sortino; switch to structured output) |
| 5 | `benchmark_lookup.py` | Completely static data; single multiple | **HIGH** | Medium (live fetch from Damodaran; add EV/EBITDA, P/B) |
| 6 | `yahoo_finance.py` | No cash flow; no adjusted close; stale `info` | **HIGH** | Medium (add `.cashflow`; use `.history(auto_adjust=True)`) |
| 7 | `editor.py` | Grounding check false negatives; flat tolerance | **MEDIUM** | Low (fix number parser + scale-relative tolerance) |
| 8 | `report_writer.py` | No structured output; 9000-char truncation | **MEDIUM** | Medium (Gemini JSON mode + explicit field extraction) |
| 9 | `memory/long_term.py` | Keyword-only retrieval for summaries | **MEDIUM** | Medium (use PageIndex embeddings for memory retrieval) |
| 10 | `orchestrator.py` | No per-ticker parallelism; linear for all queries | **MEDIUM** | Medium (parallel research with asyncio semaphore) |
| 11 | `quant_analyst.py` | LLM-written bull/bear; no structured output | **MEDIUM** | Low (switch to Gemini JSON mode for bull/bear) |
| 12 | `memory/short_term.py` | Char-based token estimation; drops full messages | **MEDIUM** | Low (use real tokenizer; turn-pair-aware windowing) |
| 13 | `refinement_handler.py` | No version history; no fuzzy match | **LOW-MEDIUM** | Medium |
| 14 | `pageindex/retriever.py` | No re-ranking; no HyDE query expansion | **LOW-MEDIUM** | Medium |
| 15 | `budget_tracker.py` | Request-counting, not token-counting | **LOW** | Low |
| 16 | Charts | No analyst targets; no intraday | **LOW** | Low-Medium |

---

*This document is the basis for the next implementation cycle. Each item will be reviewed, feasibility-assessed, and — where approved — redesigned and implemented.*
