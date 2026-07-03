# CloudNest Support Router — Design (2026-07-03)

## Objective
Multi-agent customer support router built on LangGraph (Python), answering from the 7
CloudNest markdown docs in `cloudnest_docs/`.

## Architecture
Graph: `START → router → retriever → [conditional] → responder | clarify → END`

- **router** — rule-based keyword classifier. Labels the latest user query as
  `billing`, `technical`, or `general`. No LLM, no function-calling (hard constraint).
- **retriever** — pure-Python lexical retrieval. Docs are chunked by `##` headings at
  startup. Query terms (stopwords removed) are scored against chunks with TF weighting,
  restricted to the category's doc subset:
  - billing → `02_pricing_billing.md`, `03_account_management.md`
  - technical → `04_technical_setup.md`, `05_troubleshooting.md`
  - general → all docs
  Emits top-3 chunks and a `confidence` score (fraction of query terms matched in the
  best chunk).
- **conditional edge** — `confidence >= 0.25` → responder; otherwise → clarify.
- **responder** — generates the answer from retrieved context. Uses Claude
  (`ANTHROPIC_API_KEY`) when the key is present; otherwise falls back to an extractive
  answer assembled from the top chunks. The LLM is never used for routing.
- **clarify** — asks the user to rephrase / add detail; no answer is attempted.

## State & multi-turn
`TypedDict` state: accumulated `messages`, plus per-turn `category`, `context`,
`confidence`. Persistence via LangGraph `MemorySaver` checkpointer keyed by
`thread_id`; the CLI chat loop reuses one thread so history survives across turns.

## Constraints (hard)
1. No OpenAI/Anthropic native function-calling for routing — router is rule-based.
2. Core logic (`app.py`) stays under ~150 lines.

## Files
- `support_router/app.py` — graph + CLI loop (all core logic)
- `support_router/requirements.txt` — `langgraph`, `anthropic`

## Error handling
- Missing API key → extractive fallback (never crashes).
- Empty/low-signal query → clarify node via the confidence edge.

## Testing
- Scripted multi-turn session: a billing query, a technical query, and a vague query
  must route to responder/responder/clarify respectively.
- Verify state persistence: message history grows across turns within one thread.
- Verify `app.py` line count < 150.
