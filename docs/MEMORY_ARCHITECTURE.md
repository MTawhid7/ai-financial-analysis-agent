# Memory Architecture & Authentication Design

**Status:** Design document — not yet implemented  
**Scope:** Covers the memory bug fix, conversation persistence, Google authentication, per-user isolation, and long-term memory retrieval strategy.

---

## 1. Root Cause of the Current Memory Bug

**Symptom:** Asking *"What did we find out about AAPL earlier?"* in a new session re-runs the full analysis pipeline instead of retrieving the stored summary.

**Root cause — missing intent:**

The classifier has four intents: `financial_analysis`, `financial_question`, `off_topic`, `clarification_needed`. Classifier Rule 1 says:

> *"If the user mentions a company name or stock symbol and seems to want information or analysis about it, use `financial_analysis`."*

"AAPL" appears in the message → `financial_analysis` fires → `_handle_financial_analysis(tickers=["AAPL"])` is called → pipeline runs. The memory context **was** built and **was** retrieved, but is then silently discarded because the `financial_analysis` handler ignores it.

**Fix — add `memory_query` intent:**

A fifth intent is needed:

| Intent | Trigger | Handler |
|--------|---------|---------|
| `memory_query` | User asks about a past interaction, previous result, or what was said/found before | Answer directly from stored summaries; no pipeline |

Retrospective signal phrases that bypass the `financial_analysis` route:
- "what did we find", "what did you say", "what was the", "what were the"
- "earlier", "last time", "previously", "before", "last session", "remind me"
- "what you found", "the report you wrote", "your previous analysis"
- "do you remember", "recall"

**How the handler works:** look up `analysis_summaries` by ticker/keyword, format the match as a conversational response, optionally offer to re-run a fresh analysis.

---

## 2. What "Memory" Means at Each Level

There are four distinct memory concerns in this system. Conflating them causes bugs like the one above.

| Memory tier | Scope | Storage | Lifetime |
|---|---|---|---|
| **Context window** | Current LLM call | In-process (list of LangChain messages) | Single request |
| **Conversation history** | Current chat thread | SQLite `messages` table | Until user deletes it |
| **User preferences** | All of a user's conversations | SQLite `preferences` table | Until user clears |
| **Analysis summaries** | All of a user's past pipeline runs | SQLite `analysis_summaries` table | Until user clears |

The current Phase 2 implementation conflates context-window management with long-term retrieval, and uses a single shared `.memory/memory.db` file that has no concept of users or conversation threads.

---

## 3. Memory Inheritance Rules

These rules define explicitly what carries over from one context to another.

| Source → Destination | What carries over | What does not |
|---|---|---|
| Previous conversation → New conversation | User preferences (always) · Top-3 most relevant analysis summaries (retrieved by query) | Full message history of old conversation |
| Previous session → Same conversation resumed | Full message history · Preferences · All summaries | Nothing — everything is available |
| Old analysis → New analysis (same ticker) | Previous summary is surfaced in memory context | Raw pipeline data (re-fetched from yfinance/Tavily) |
| User A → User B | Nothing — complete isolation per user_id | N/A |

**Key decision: new conversations should inherit preferences but NOT full history.**
Showing all prior conversation history in a new chat would be overwhelming and irrelevant. Instead, the `MemoryManager` retrieves the most relevant 2-3 past analysis summaries based on the current query. Full history is only available by explicitly reopening the original conversation.

---

## 4. Authentication Design: Google Sign-In

### Why Streamlit is insufficient

Streamlit re-runs the entire script on every interaction. There is no built-in session persistence, no redirect URI support, and no bidirectional communication. Implementing Google OAuth properly in Streamlit requires workarounds that are fragile and non-standard.

**Authentication is the trigger to execute Phase 4 (FastAPI + React) earlier than planned.**

### OAuth flow

```
1. User clicks "Sign in with Google" in the React frontend
2. Frontend: Google Identity Services SDK returns an ID token (client-side)
3. Frontend: POST /auth/google  { id_token: "..." }
4. Backend: validates token using google-auth Python library
5. Backend: upserts user in `users` table, issues a signed JWT (30-day expiry)
6. Frontend: stores JWT in httpOnly cookie (XSS-safe) or memory
7. All subsequent API calls include JWT in Authorization header
```

### Dependencies

| Package | Role |
|---|---|
| `google-auth>=2.29` | Server-side ID token validation |
| `python-jose[cryptography]>=3.3` | JWT signing and verification |
| `@react-oauth/google` (npm) | Google Identity Services button component |

### User model

```sql
CREATE TABLE users (
    id           TEXT PRIMARY KEY,  -- Google sub (stable user identifier)
    email        TEXT NOT NULL UNIQUE,
    display_name TEXT,
    picture_url  TEXT,
    created_at   REAL NOT NULL,
    last_seen_at REAL NOT NULL
);
```

---

## 5. Persistent Chat History: Database Schema

All memory tables gain a `user_id` foreign key. The current shared `.memory/memory.db` is replaced by a single multi-tenant database at `.memory/db.sqlite`.

```sql
-- Users (populated by Google OAuth)
CREATE TABLE users (
    id           TEXT PRIMARY KEY,
    email        TEXT NOT NULL UNIQUE,
    display_name TEXT,
    picture_url  TEXT,
    created_at   REAL NOT NULL,
    last_seen_at REAL NOT NULL
);

-- Conversations (one per chat thread)
CREATE TABLE conversations (
    id         TEXT PRIMARY KEY,  -- UUID
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title      TEXT,              -- auto-generated from first message, editable
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX idx_conversations_user ON conversations(user_id, updated_at DESC);

-- Messages (each turn saved in real time)
CREATE TABLE messages (
    id              TEXT PRIMARY KEY,  -- UUID
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,     -- 'user' | 'assistant'
    content         TEXT NOT NULL,
    intent          TEXT,              -- classified intent for this user turn
    tickers         TEXT,              -- comma-separated, set when intent=financial_analysis
    created_at      REAL NOT NULL
);
CREATE INDEX idx_messages_conv ON messages(conversation_id, created_at);

-- Preferences (per user, upserted by key)
CREATE TABLE preferences (
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (user_id, key)
);

-- Analysis summaries (per user, linked to the originating conversation)
CREATE TABLE analysis_summaries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT REFERENCES conversations(id),
    tickers         TEXT NOT NULL,
    summary_text    TEXT NOT NULL,
    run_id          TEXT DEFAULT '',
    created_at      REAL NOT NULL
);
CREATE INDEX idx_summaries_user ON analysis_summaries(user_id, created_at DESC);
```

### Conversation lifecycle

1. User opens the app → sidebar shows list of their conversations (title + last updated)
2. User clicks "New Chat" → new `conversations` row inserted, session starts fresh
3. Each user message → `messages` row inserted immediately (before agent processes it)
4. Each assistant response → `messages` row inserted
5. After a pipeline run → `analysis_summaries` row inserted, conversation `updated_at` bumped
6. Conversation title → auto-generated from the first user message using Flash-Lite (one call), capped at 60 chars; user can rename it

---

## 6. Per-User Memory Isolation

All `LongTermMemory` and `MemoryManager` operations gain a required `user_id` parameter. No query is ever executed without scoping to `user_id`.

```python
# Before (Phase 2 — shared, single-user)
await lt.get_all_preferences()
await lt.save_analysis_summary(session_id, tickers, summary, run_id)

# After (Phase 4 — per-user, conversation-aware)
await lt.get_preferences(user_id=user_id)
await lt.save_analysis_summary(user_id=user_id, conversation_id=conv_id, tickers=tickers, ...)
```

The `ConversationalAgent` receives the `user_id` as a constructor argument (injected by the session manager). All downstream calls are scoped automatically.

---

## 7. Long-Term Memory Retrieval Strategy (Evolution)

### Phase 2.5 → Phase 4: Keyword search (current approach, extended)

`LIKE '%query%'` across `tickers` and `summary_text`. Fast, zero dependencies. Works well when the user references a specific ticker or company name. Fails for semantic queries ("what stocks had good momentum?").

**Limitation:** Does not understand synonyms, paraphrases, or thematic queries.

### Phase 5: Hybrid keyword + recency ranking

Combine LIKE search with a recency boost: `score = match_count + decay(created_at)`. Simple to implement with SQLite, no new dependencies. Handles "what did we look at recently?" better.

### Phase 7+: Vector embeddings (semantic search)

Use `sentence-transformers` (the `all-MiniLM-L6-v2` model, ~80MB, CPU-only, no API cost) to embed each analysis summary at save time. Store the embedding as a BLOB in a `summary_embeddings` table. At retrieval time, embed the query and find the k-nearest summaries by cosine similarity.

```python
from sentence_transformers import SentenceTransformer
_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# On save:
embedding = _encoder.encode(summary_text).tobytes()

# On query:
query_vec = _encoder.encode(query_text)
# SQLite doesn't support vector ops natively — load candidate summaries, 
# compute cosine in Python, return top-k
```

This enables queries like "what was the most bullish analysis I ran?" or "remind me about defensive stocks I looked at."

**Trigger for vector search upgrade:** when `count_summaries()` exceeds ~200 rows OR when user feedback indicates recall quality is poor.

### Retrieval rules by query type

| Query type | Strategy |
|---|---|
| Ticker-specific ("what did we find about AAPL") | Exact LIKE match on `tickers` column — fastest |
| Thematic ("show me my tech stock analyses") | Hybrid LIKE + recency |
| Temporal ("what did I ask about last week") | `created_at` range filter |
| Semantic ("find analyses where I was bullish") | Vector similarity (Phase 7+) |

---

## 8. Full API Surface (FastAPI — Phase 4)

```
POST   /auth/google                Validate Google ID token, issue JWT
GET    /auth/me                    Return current user profile
POST   /auth/logout                Invalidate session

GET    /conversations              List user's conversations (title, updated_at, message_count)
POST   /conversations              Create new conversation, return id
GET    /conversations/{id}         Load full message history for a conversation
DELETE /conversations/{id}         Delete conversation + all messages

POST   /chat/{conversation_id}     Send message, stream response via SSE
GET    /stream/{conversation_id}   SSE stream for tool events during pipeline

GET    /memory/preferences         Return all stored preferences
PATCH  /memory/preferences         Upsert a preference key-value pair
DELETE /memory/preferences/{key}   Delete one preference
GET    /memory/summaries           List analysis summaries (paginated)
DELETE /memory/summaries/{id}      Delete one summary
POST   /memory/clear               Delete all memory for the current user
```

---

## 9. React Frontend Requirements (Phase 4)

### Conversation sidebar

```
┌─────────────────────────┐
│ [+] New conversation    │
├─────────────────────────┤
│ ▪ AAPL Analysis         │ ← today
│ ▪ What is CAGR?         │ ← yesterday
│ ▪ NVDA vs AMD compare   │ ← 3 days ago
│ ▪ Tesla valuation       │ ← last week
│   [Load more...]        │
└─────────────────────────┘
```

- Clicking a past conversation loads full message history via `GET /conversations/{id}`
- "New conversation" clears the chat area and creates a new conversation record
- Conversations are sorted by `updated_at DESC`
- Title is editable (double-click to rename)

### Memory panel (accessible via sidebar toggle)

```
┌──────────────────────────────┐
│ 🧠 Your Memory               │
├──────────────────────────────┤
│ Preferences                  │
│  - investment_style: conservative │
│  - summary_length: brief     │
│                              │
│ Past analyses (14)           │
│  [AAPL] Apple showed 17% ... │
│  [NVDA] NVIDIA P/E 55x vs .. │
│  [TSLA] Tesla revenue grew.. │
│  [More...]                   │
│                              │
│ [Clear all memory]           │
└──────────────────────────────┘
```

---

## 10. Phased Implementation Plan

### Phase 2.5 — Memory Bug Fix + Conversation Persistence (Streamlit)

**Goal:** Fix the immediate bug and add persistent conversation history without changing the UI framework.

**Changes:**
- `intent_classifier.py` — add `memory_query` to the taxonomy; add retrospective signal phrases to the prompt
- `conversational_agent.py` — add `_handle_memory_query()` that searches `analysis_summaries` and returns a formatted response; if no match found, offers to re-run the analysis
- `memory/long_term.py` — add `conversations` and `messages` tables; auto-save each message turn; `get_conversations_list(user_id)` for sidebar
- `ui/chat_app.py` — conversation list in sidebar (session-keyed for now, no auth); persist messages to DB across restarts

**Effort:** ~2 days · No new dependencies · No auth required

---

### Phase 4A — FastAPI Backend + Google Authentication

**Goal:** Production-grade backend with multi-user auth and per-user data isolation.

**New files:**
```
backend/
  main.py              FastAPI app, CORS, middleware
  routers/
    auth.py            POST /auth/google, GET /auth/me, POST /auth/logout
    conversations.py   CRUD for conversations + messages
    chat.py            POST /chat/{conv_id}, SSE /stream/{conv_id}
    memory.py          Preferences + summaries CRUD
  core/
    auth.py            JWT signing, Google ID token validation
    session_manager.py LRU cache of ConversationalAgent per user (30-min TTL)
    database.py        aiosqlite connection pool + schema migration
```

**Dependencies added:** `fastapi>=0.115`, `uvicorn[standard]>=0.30`, `google-auth>=2.29`, `python-jose[cryptography]>=3.3`, `python-multipart>=0.0.9`

**Memory module changes:** all methods gain `user_id` parameter; schema migrated to multi-tenant (single `.memory/db.sqlite` with `user_id` columns and foreign keys)

---

### Phase 4B — React Frontend with Auth + Conversation UI

**Goal:** Full frontend with Google Sign-In, conversation history, and memory panel.

**Key components:**
- `AuthProvider.tsx` — Google OAuth context, JWT storage, auto-refresh
- `ConversationSidebar.tsx` — conversation list, new chat button, delete/rename
- `ChatInterface.tsx` — message thread for the active conversation
- `MemoryPanel.tsx` — preferences editor, analysis summaries list, clear all

**SSE streaming:** `EventSource` connects to `/stream/{conversation_id}` and renders `ToolCallBubble` events inline as the pipeline runs.

---

### Phase 7 — Vector Memory (Semantic Retrieval)

**Goal:** Replace LIKE search with vector similarity for thematic and semantic memory queries.

**New dependency:** `sentence-transformers>=3.3` (CPU-only, ~80MB model download on first use)

**Changes:**
- `memory/long_term.py` — add `summary_embeddings(summary_id, embedding BLOB)` table; embed at save time
- `memory/retrieval.py` — `semantic_search(user_id, query, k=5)` function
- `MemoryManager.build_memory_context` — use semantic search when keyword search returns 0 results

**Trigger:** activate when `count_summaries() > 200` OR explicit user-facing "memory quality" feedback is negative.

---

## 11. Memory Inheritance — Definitive Rules

| Scenario | Preferences | Analysis summaries | Conversation messages |
|---|---|---|---|
| Resuming the same conversation | ✅ All | ✅ All | ✅ Full history loaded |
| Opening a new conversation (same user) | ✅ All | ✅ Top-3 most relevant to first query | ❌ Not shown |
| After Streamlit restart (no auth) | ✅ (from SQLite) | ✅ (from SQLite) | ✅ (from SQLite, Phase 2.5+) |
| Different browser / device (post-Phase 4) | ✅ (server-side, tied to user_id) | ✅ | ✅ |
| New user account | ❌ | ❌ | ❌ |
| User clears memory | ❌ Deleted | ❌ Deleted | ✅ Conversation history preserved (separate action) |
| User deletes a conversation | Unaffected | Unaffected | ❌ Messages deleted |

**Design rationale for new conversations:** Injecting the full history of past conversations into every new chat would flood the context window and feel intrusive. The right behaviour is: preferences always carry over (they define who the user is), and relevant analysis summaries surface only when the query is related to past work. This creates the "it knows me" feeling without the "it's reading my old emails" creepiness.
