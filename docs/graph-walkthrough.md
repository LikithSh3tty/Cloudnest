# CloudNest Support Router — Live Walkthrough Guide

A presenter's script for demoing the graph. Pair it with `CHANGELOG.md` (evolution
story) and `SUBMISSION_NOTE.md` (the ½-page written deliverable).

## The graph

```
START ──> router ──> retriever ──┬─ confidence ≥ 0.25 ──> responder ──> END
                                 └─ confidence < 0.25 ──> clarify ────> END
```

Four nodes, one conditional edge, compiled with a `MemorySaver` checkpointer.
All of it lives in `support_router/app.py` (145 lines).

## Node-by-node (talking points)

**router** — pure Python, no LLM (hard constraint). Tokenizes the latest user message
and intersects with two keyword sets: `BILLING_WORDS` and `TECH_WORDS`. Larger overlap
wins (ties → billing); zero overlap on both → "general". ~6 lines.

**retriever** — lexical RAG, no embeddings. At startup the 7 docs are split into chunks
on `##` headings. Query terms (stopwords removed, synonyms normalized) are TF-scored
against every chunk; chunks from the routed category's preferred docs get a 2× boost
(*soft scoping* — see changelog #5 for why this replaced hard filtering). Emits top-3
chunks and `confidence` = fraction of query terms matched by the best chunk.

**conditional edge** — the `picker` function in `build_app()`: confidence ≥ 0.25 →
responder, else → clarify. This is the reason a graph beats a chain: the path is chosen
at runtime from state.

**responder** — the only place an LLM appears. Claude gets the retrieved chunks plus
conversation history and must answer only from that context. If the call fails (no key,
no network) it degrades to an extractive answer stitched from the chunks — never crashes.

**clarify** — no answer attempted; asks the user to rephrase with detail. This is the
honesty valve: weak retrieval never becomes a hallucinated answer.

**State** — one `TypedDict` flows through everything: `messages` (append-only via the
`add` reducer; persisted across turns by thread_id), plus per-turn `category`, `context`,
`confidence`. The CLI reuses one thread, so history survives across turns.

## Live demo script

Run `python support_router\app.py` — first confirm the banner says
`mode: Claude responder`. Then, in order:

| # | Type this | What it demonstrates |
|---|-----------|----------------------|
| 1 | `how much does the pro plan cost?` | Happy path: routes billing, confidence 1.00, answer from the pricing table |
| 2 | `Where can I configure bandwidth limits?` | Technical routing (keyword classifier), scoped retrieval |
| 3 | `Which plans include API access, and how much do they cost?` | The showpiece: cross-category question. Routes billing, but soft scoping still pulls the API section from the *technical* doc, and cost→price synonym pulls the pricing table. Conf 0.40 |
| 4 | `hi` | Clarify path: confidence 0.00 → no answer attempted, asks for detail |
| 5 | `does that plan include API access?` (right after #1) | Multi-turn state: "that plan" isn't in the query — the responder resolves it to Pro from persisted history. Conf 0.75 |

All five are verified working. Caveat for #5: the follow-up must still contain
doc-matching words ("plan", "API", "access"). A fully anaphoric follow-up like "what
about the second one?" scores 0 at retrieval and routes to clarify — retrieval sees only
the current turn, and the confidence edge fires before the history-aware responder runs.
If asked, that's a known design limit (fix would be rewriting the query from history
before retrieval — a fifth node).

## Questions to expect

- **Why LangGraph over a chain?** The clarify branch is runtime control flow — a chain
  is a fixed pipeline. Plus free checkpointing (multi-turn) and cheap extensibility
  (an escalation node is one node + one edge).
- **Why rule-based routing?** Assignment constraint: no function-calling shortcuts.
  Trade-off shown honestly in demo #3 — keyword routers mislabel cross-category
  questions, which is exactly why retrieval uses soft scoping rather than trusting the
  label absolutely.
- **What if retrieval fails?** Two layers: low confidence → clarify (never answers from
  weak context); LLM failure → extractive fallback (never crashes, never fabricates).
- **Known limits?** Lexical matching needs the synonym map for vocabulary gaps
  (cost/price); real fix at scale is embeddings, out of scope at 150 lines. Confidence
  is term-overlap, not semantic relevance.
