# CloudNest Support Router — Change Log

All changes dated 2026-07-03. Line counts are raw lines of `support_router/app.py`
(hard constraint: under 150). Every change below was verified by re-running the spec's
test checklist: billing/technical/vague queries → responder/responder/clarify, message
history growing 2→4→6 across turns, and no LLM in routing.

## 1. Initial build
Graph `START → router → retriever → [confidence ≥ 0.25] → responder | clarify → END`.
Rule-based keyword router (no LLM/function-calling — hard constraint), lexical TF
retrieval over `##`-chunked docs, Claude responder with extractive fallback, MemorySaver
checkpointer for multi-turn state, CLI loop. Follow-ups: UTF-8 console output; startup
line prints the responder mode so a missing API key is visible immediately.

## 2. API key via project `.env`
**Problem:** key was set at Windows User scope; shells opened before that never inherit
it, so the app silently fell back to extractive answers.
**Change:** git-ignored `.env` at project root; `app.py` loads it at startup
(`os.environ.setdefault`, so a shell-set key still wins). Added `.gitignore`.

## 3. Line-budget compaction (169 → 149 lines)
The `.env` loader pushed the raw line count over 150. Compacted with no logic changes:
single blank line between definitions, tighter docstring, node registration loop keyed
on `fn.__name__`, conditional-edge lambda became a named `picker` function.

## 4. Router vocabulary expansion
**Problem:** "Where can I configure bandwidth limits?" routed as *general* — none of its
words were in `TECH_WORDS`. (Answer was still correct: general searches all docs.)
**Change:** added bandwidth, network, configure, settings, preferences, speed, limit,
limits, proxy. Keyword sets match exact tokens, so plurals must be listed explicitly.

## 5. Soft scoping in the retriever (the significant design change)
**Problem:** "Which plans include API access, and how much do they cost?" → routed
*billing*, and the billing category hard-filtered retrieval to the two billing docs.
The answer lives in `04_technical_setup.md` ("API Access"), which was excluded before
scoring — the responder truthfully reported its context said nothing about APIs.
Keyword tweaks cannot fix this: billing words legitimately dominate the query.
**Change:** score all docs; chunks from the routed category's docs get a 2× boost
instead of exclusivity. On-category behavior unchanged; cross-category answers stay
reachable. Design spec updated to match. Retriever got 2 lines shorter (147 total).

## 6. Tokenizer: stopwords + synonym normalization (145 lines)
**Problem (residual from #5):** the pricing table says "Price"; the query says "cost" —
lexical matching can't bridge synonyms, so prices were still missing from the answer.
Also, filler words ("which", "much", "they") counted as unmatched terms and dragged
confidence down.
**Change:** added they/them/which/much to `STOPWORDS`; `SYNONYMS` map normalizes
cost/costs/pricing → price during tokenization (applied to both queries and doc text, so
matching stays consistent and `BILLING_WORDS` routing still works). Docstring cut to one
line to fund the additions. The API-access question now answers completely (conf
0.25 → 0.40); "how much does the pro plan cost?" confidence rose 0.50 → 1.00.

## 7. Submission note
Added `SUBMISSION_NOTE.md` (½ page): why LangGraph over a simple chain, what state
persists across nodes, what happens on retrieval failure.

## 8. Synonym map extended with word-form variants (147 lines)
**Problem:** "Does CloudNest automatically merge conflicting files?" → wrong "docs don't
cover it" answer. Exact-token matching treats "conflicting" ≠ "conflict"/"conflicted"
and "files" ≠ "file", so the Conflict Resolution chunk lost to chunks that repeat
generic words; the router also missed the technical label for the same reason.
**Change:** `SYNONYMS` gained conflicting/conflicted/conflicts → conflict,
files → file, merging/merges → merge. Query now routes technical with confidence
0.75 and answers correctly ("creates a conflicted copy; does not auto-merge").
**Known limit (deliberate):** this is per-word-family patching; the general fix is
embedding-based retrieval (~10 lines with sentence-transformers, same graph), kept out
of scope to preserve the transparent, defensible TF scoring for the live walkthrough.

## 9. Vercel deployment: stateless API, client-carried history
**Problem:** Vercel Python functions are stateless — the in-process MemorySaver cannot
persist turns between requests.
**Change:** `api/index.py` replaces `server.py`: the browser sends the conversation
history with each request and the graph is invoked once per request on a fresh thread.
`app.py` is untouched — the same messages-list input drives router/retriever/responder.
Added `vercel.json` (frontend build, `/api/*` rewrite, `includeFiles` bundling
`support_router/` + `cloudnest_docs/`) and root `requirements.txt`. The same API file
serves local dev (`python -m uvicorn api.index:app`). Multi-turn verified over the
stateless protocol.

## Operational lesson (bit us twice)
Python reads `app.py` once at launch. After any code or environment change, restart the
chat session — a running `you>` prompt keeps executing the old version.
