# CloudNest Support Agent

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-1C3C3C?logo=langchain&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?logo=vite&logoColor=white)
![Vercel](https://img.shields.io/badge/Deploy-Vercel-000000?logo=vercel&logoColor=white)

**Live:** [cloudnest-nine.vercel.app](https://cloudnest-nine.vercel.app)

A support chatbot for a fictional cloud-storage product called CloudNest. You type a question in plain English, and it figures out whether you're asking about billing or something technical, pulls the relevant bits out of the product docs, and answers you like a real support engineer would. It runs on a small LangGraph state machine on the backend and a React chat window on the front.

Retrieval is real semantic search: every doc section is embedded ahead of time by a small local model, no vector database or embedding provider involved, and matched against your question by cosine similarity. See [How retrieval actually works](#how-retrieval-actually-works) below for the full story, including the keyword-based fallback it degrades to if the model or index can't load. If the API key is present it uses Claude to phrase the final answer, written to sound like a person rather than an AI model; if not, it hands you back the matching sections as clean Markdown. Either way you always get an answer.

## What it does

- **Routes each question** to billing, technical, or general before it does anything else, so the retriever knows which docs to favor.
- **Retrieves by meaning, not keywords.** Doc sections are embedded once ahead of time and ranked by cosine similarity against the question, so "how much will I be charged if I add a teammate mid-cycle" finds the pricing section even though it shares no words with it. Falls back to keyword matching if the embedding model or index isn't available.
- **Answers with Claude when a key is set**, grounded strictly in the retrieved sections and written in a natural, human support-agent voice: no mention of documentation, retrieval, confidence, or routing, no emoji, no em dashes. No key, and it falls back to the retrieved sections as clean Markdown instead of failing.
- **Formats for reading**, not just talking: a direct answer up front, Markdown tables for comparisons, bullets for options, numbered steps for procedures.
- **Bails out honestly** when it isn't confident. Below a threshold calibrated against real questions (see `backend/calibrate.py`), it asks you to add detail rather than guessing.
- **Keeps retrieval internals out of the UI.** The API still returns the route, confidence, and cited sources for every answer, but the chat window shows only the answer itself.
- **Remembers the conversation.** The CLI keeps context across turns via a LangGraph checkpointer; the web version carries history in the browser since the serverless function is stateless.
- **Understands follow-ups.** A reply like "does it cost extra" carries no topic word of its own — retrieval folds in the previous turn before embedding the question, so it still finds the right section instead of scoring below the confidence threshold or matching the wrong one.

## How it's wired

The backend is a LangGraph `StateGraph` with eight nodes: a router, two retrievers that run in parallel, a fusion step, and a four-way gate to a responder, a clarify, a deflect, or an escalate node. A question comes in, gets routed, gets matched against the docs by both a semantic and a lexical branch at once, and the two rankings are merged by Reciprocal Rank Fusion before the gate decides whether to answer, ask for more detail, offer to help in place of a human, or hand off to a human.

```
                         question
                            │
                            ▼
                        ┌────────┐
                        │ router │
                        └───┬────┘
                 ┌──────────┴──────────┐   (run in parallel)
                 ▼                     ▼
         ┌───────────────┐   ┌──────────────┐
         │ semantic_retr │   │ lexical_retr │
         └───────┬───────┘   └──────┬───────┘
                 └──────────┬────────┘
                            ▼
                        ┌────────┐
                        │ fusion │   RRF merge, cosine confidence
                        └───┬────┘
                            ▼
                    ┌───────────────┐
                    │ gate (2 floor)│
                    └──┬──┬──┬──┬────┘
              conf≥0.30│  │  │  │ 1st human request ─► deflect (try me first)
              or rescue│  │  │  └ insisted / 2nd borderline miss ─► escalate ─► ticket
                       ▼  ▼  ▼
              responder clarify deflect
```

A first explicit request for a human ("let me talk to a person", "get me an agent", "connect me to a representative") doesn't escalate straight away. It routes to the deflect node, which offers to help right there first, since the bot usually can. Only if the user still asks for a person after that does the gate escalate. Escalation also fires on a repeated borderline miss (a real, in-scope question the bot couldn't get confident about two turns running). The escalate node builds a support ticket (id, timestamp, question, category, confidence, the full conversation, and a reason) and hands off rather than guessing further. That ticket comes back from the API today but isn't persisted anywhere yet — see [Things I'd add next](#things-id-add-next).

`index.py` wraps that same graph in a FastAPI app so the React frontend can talk to it over `/api/chat`. `app.py` can also be run on its own as a command-line chat loop, which is the quickest way to poke at the logic without touching the frontend.

## Project layout

```
Alliedworks/
├── backend/
│   ├── app.py               # the LangGraph agent + CLI entry point
│   ├── index.py             # FastAPI wrapper (/api/chat, /api/health)
│   ├── embed.py             # local ONNX embedding function
│   ├── build_index.py       # offline indexer — embeds cloudnest_docs/ into index.npz
│   ├── calibrate.py         # measures the confidence threshold from real questions
│   ├── index.npz            # committed embedding index (vectors + chunk metadata)
│   ├── model/                # vendored int8 embedding model + tokenizer
│   ├── tests/                # pytest suite (44 tests)
│   ├── requirements.txt
│   └── requirements-dev.txt  # requirements.txt + pytest
├── cloudnest_docs/          # the knowledge base — plain markdown
│   ├── 01_product_overview.md
│   ├── 02_pricing_billing.md
│   ├── 03_account_management.md
│   ├── 04_technical_setup.md
│   ├── 05_troubleshooting.md
│   ├── 06_security_privacy.md
│   └── 07_general_faq.md
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # the chat UI
│   │   ├── main.jsx
│   │   └── app.css
│   ├── index.html
│   ├── package.json
│   └── vite.config.js       # dev proxy to the backend on :8000
├── vercel.json              # builds both halves for deployment
└── .env                     # ANTHROPIC_API_KEY lives here (gitignored)
```

## Running it locally

You'll need Python 3.10+ and Node 18+.

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
```

Drop your key in a `.env` file at the project root (one level up from `backend/`):

```
ANTHROPIC_API_KEY=your_key_here
```

The key is optional. Without it the agent still runs, it just returns the matched doc sections as Markdown instead of a Claude-written answer. Handy if you want to see exactly what retrieval is pulling.

Then start the API:

```bash
uvicorn index:app --reload --port 8000
```

Or skip the server entirely and chat in the terminal:

```bash
python app.py
```

The CLI prints the route and confidence before each answer, which is useful when you're tuning the retrieval threshold or checking whether the lexical fallback kicked in.

### 2. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Vite serves the UI on `http://localhost:5173` and proxies `/api` calls through to the backend on `:8000`, so you don't have to deal with CORS. The header shows "Support available" when a key is present and "Limited support" otherwise.

Once it's running, [`EXAMPLE_QUESTIONS.md`](./EXAMPLE_QUESTIONS.md) has a curated set of questions to try, from single-fact lookups to ones that need several doc sections combined.

## API

Two endpoints, both under `/api`.

**`POST /api/chat`**

```json
{
  "message": "How much does the Pro plan cost?",
  "history": []
}
```

`history` is a list of `{ "role": "user" | "assistant", "content": "..." }` objects. The frontend keeps track of it and sends the running conversation with each request, since the backend doesn't hold session state of its own.

Response:

```json
{
  "answer": "Pro is $9.99/month, or $99/year if you pay annually...",
  "category": "billing",
  "confidence": 0.67,
  "clarified": false,
  "escalated": false,
  "ticket": null,
  "sources": ["Plans and Pricing"]
}
```

`clarified` and `escalated` tell you which branch of the graph answered (`clarified: true` means it asked you to add detail instead of answering; `escalated: true` means it handed the conversation off instead). `sources` lists the titles of the doc sections the answer was cited from — empty on either a clarify or an escalate, since neither is a doc-grounded answer. `ticket` carries the escalation record (id, timestamp, question, category, confidence, the full conversation, and a reason) when `escalated` is true, otherwise `null`. None of these are shown in the chat window; all exist for logging and debugging. The ticket is returned by the API but not yet written anywhere durable — see [Things I'd add next](#things-id-add-next).

**`GET /api/health`** — returns `{ "mode": "claude" }` if a key is configured, `{ "mode": "extractive" }` otherwise, plus a `retrieval` field: `"semantic"` when the embedding index is loaded, or `"lexical"` when it has fallen back to keyword retrieval. The UI uses `mode` to set the status badge.

## How retrieval actually works

Worth a note since it's the core of the thing and there's nothing hidden.

Each doc gets split into sections on its markdown headings. Every section is embedded once, ahead of time, by a small local model (`all-MiniLM-L6-v2`, an int8 ONNX export — no API key, no per-query cost, runs offline) and the resulting vectors are committed as `backend/index.npz`. When a question comes in, only the question is embedded, and sections are ranked by cosine similarity against it. All 46 sections (`TOP_K` in `app.py`, set to the corpus size while it's under `SMALL_CORPUS_LIMIT`) become the context, and confidence is the similarity of the best one. Because it matches on meaning rather than shared words, "how much will I be charged if I add a teammate mid-cycle" finds the pricing section even though it shares no keywords with it.

The whole corpus rather than a fixed cutoff, because a question can genuinely need several sections spread far apart in the ranking: "I need API access, EU residency, bank transfer, and CLI automation" touches 4 sections, and 2 of them ranked outside any small top-N against that query (10th and 22nd out of 46). At ~1,800 tokens for the entire corpus, sending everything costs nothing meaningful, and it's simpler than tuning a cutoff that would just need raising again for the next multi-fact question. `SMALL_CORPUS_LIMIT` (150 chunks) marks where that stops being true; past it, `TOP_K` falls back to a fixed `FALLBACK_TOP_K`, which is an unvalidated placeholder rather than a tuned value, on the same reasoning that any fixed cutoff can drop a section the way the old `TOP_K=5` did. This only changes what an already-confident answer gets to draw from — `CONFIDENCE_THRESHOLD` routing still looks at just the single best score, and the API still cites only the best 3 by score, so the wider recall doesn't clutter the sources it reports.

If that top score clears `0.30` (a constant near the top of `app.py`, chosen by measuring the gap between in-scope and out-of-scope questions — see `backend/calibrate.py`), the responder runs. If it doesn't, you get the clarify prompt instead, or — see below — an escalation. It keeps the agent from confidently answering questions the docs don't actually cover.

Retrieval actually runs as two parallel branches: the semantic ranking above, and a lexical (token-overlap) ranking of the same corpus. `fuse_results` merges the two full rankings with Reciprocal Rank Fusion (RRF) — a chunk scores `1 / (60 + rank)` per list it appears in, so a section ranked highly in either list surfaces even if the other list buries it — and that fused order is what's sent to the LLM and cited from. Confidence itself does not change: it's still the semantic cosine top-1 score, exactly as above. Fusion only reorders context and citations; it can also *rescue* a borderline-confidence answer when the lexical branch has a strong exact-keyword hit (two or more distinct query terms matched in the top chunk), which is how a query like "cloudnest-cli" still gets answered even if the embedding alone scored it under `0.30`.

Below the gate sits a second, lower floor: `ESCALATE_FLOOR` (0.27, set in `app.py` just above the measured out-of-scope ceiling from `backend/calibrate.py`). A question scoring between the floor and `0.30` is borderline and in-scope — the first time, it gets the clarify prompt; if the *next* turn is still borderline after that, or if the user explicitly asks for a human ("agent", "person", "representative", "ticket", "escalate"), it escalates to a support ticket instead of clarifying again. Anything scoring below the floor is treated as off-topic noise and can only clarify, never escalate, no matter how many times it repeats — so two turns of "cat mouse banana" never opens a ticket. That's about score-driven escalation only: an explicit request to reach a human (a handoff phrase like "talk to a person" or "get me an agent") always hands off regardless of score, since the floor guards against off-topic noise, not against a deliberate ask for a person.

Before any of that, both `router()` and `retriever()` run the question through `contextualize_query()`, which folds the previous user turn onto the current one (capped at `CONTEXT_CHAR_CAP`, 200 characters) before it's embedded. A bare follow-up like "does it cost extra" carries no topic word of its own — alone it scores 0.285 (just under the threshold) and points at the wrong section; folded onto the turn before it ("I want to add two-factor authentication"), it scores 0.55 and finds the right one. Only the retrieval-facing query is folded — Claude still receives the full, unfolded conversation for generation, since it was never confused about "it"; only retrieval was.

If the embedding model or index can't be loaded, retrieval falls back to a keyword scorer (the previous approach — token overlap with a small synonym map), so the app still answers with no model and no API key at all.

### Rebuilding the search index

The section embeddings are generated ahead of time. After editing anything in `cloudnest_docs/`, regenerate the index:

    python backend/build_index.py

Skipping this is safe but degrading: `app.py` fingerprints the docs and, on a mismatch, falls back to keyword retrieval rather than searching outdated vectors. Run `python backend/build_index.py --calibrate` to also re-check the confidence threshold against the new content.

## Deployment

`vercel.json` is set up to build the frontend as a static site and run `backend/index.py` as a Python serverless function, with `/api/*` routed to the backend and everything else falling through to the SPA. `ANTHROPIC_API_KEY` is set as an environment variable in the Vercel project (Preview and Production). The project is linked to GitHub, so every push to `main` triggers a fresh production deployment automatically.

Live at **[cloudnest-nine.vercel.app](https://cloudnest-nine.vercel.app)**. `/api/health` there currently reports `{"mode": "claude", "retrieval": "semantic"}`, so the embedding index and the API key are both loading correctly in production. The deployed function, model and all, measures 79.83 MB per Vercel's own build output, comfortably inside the 250 MB serverless function limit.

## Things I'd add next

- Persist escalation tickets. The escalate node builds a full ticket dict and the API returns it, but nothing writes it to a database yet — it's generated per-request and discarded once the response goes out. A real datastore plus a small admin view to work the queue is the next subsystem.
- A stronger embedding model, and a real retrieval strategy (a stronger fusion/rerank than plain RRF, not just a bigger fixed cutoff) once the corpus outgrows `SMALL_CORPUS_LIMIT`. `FALLBACK_TOP_K` in `app.py` is a placeholder, not a tuned value — a fixed cutoff has the same failure mode the whole-corpus change just fixed, just at a different scale.
- Persist conversations server-side so history doesn't have to round-trip through the browser.
- Re-run `backend/calibrate.py` against real production questions once there's traffic. The current threshold is calibrated from a 16-question probe set, which is a reasonable start but not the same as live data.
- Frontend tests. The backend has 44 pytest cases around the router, the parallel retrievers, fusion, the gate, and the index; the React side is only checked by hand.
