# CloudNest Support Router — Submission Note

**Why LangGraph over a simple chain.** The flow is not linear: after retrieval, the system
decides at runtime whether to answer (responder) or ask for clarification (clarify), based on
retrieval confidence. A simple chain hard-codes one fixed path, so this branch would live in
if/else glue outside the chain. LangGraph makes it a first-class conditional edge in the graph
(`retriever → responder | clarify`). LangGraph also provides checkpointing out of the box:
compiling with a `MemorySaver` checkpointer persists state per `thread_id`, so multi-turn
memory requires no hand-rolled session store. Finally, the graph is extensible — adding an
escalation or human-handoff node later is a new node and edge, not a rewrite.

**What state persists across nodes.** A shared `TypedDict` flows through every node with four
fields: `messages` (the full conversation; merged with an `add` reducer so each node appends
rather than overwrites), `category` (the router's label: billing / technical / general),
`context` (the top-3 retrieved doc chunks), and `confidence` (the fraction of query terms
matched by the best chunk). Within a turn: the router writes `category`, the retriever writes
`context` and `confidence`, and responder/clarify append the assistant reply to `messages`.
Across turns: the checkpointer restores `messages` for the thread, so the responder sees prior
turns as conversation history, while `category`, `context`, and `confidence` are recomputed
fresh for each new query.

**What happens on retrieval failure.** Two independent safety layers. (1) *Low relevance:*
if retrieval confidence falls below 0.25 — including the case where no chunk matches at all —
the conditional edge routes to the clarify node, which asks the user to rephrase or add detail
(plan, device, exact error). The system never generates an answer from weak context. (2) *LLM
failure:* if the Claude call fails (missing API key, network error), the responder falls back
to an extractive answer assembled verbatim from the retrieved chunks, clearly attributed to the
source docs. The app never crashes and never fabricates content beyond the documentation.
