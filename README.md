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

## How it's wired

The backend is a LangGraph `StateGraph` with four nodes. A question comes in, gets routed, gets matched against the docs, and then either goes to the responder or the clarify node depending on how confident the retriever was.

```
                    question
                       │
                       ▼
                  ┌─────────┐
                  │ router  │   billing / technical / general
                  └────┬────┘
                       ▼
                  ┌───────────┐
                  │ retriever │   embed + cosine-rank doc sections, compute confidence
                  └─────┬─────┘
                        │
             confidence ≥ 0.30 ?
              ┌─────────┴─────────┐
              │ yes               │ no
              ▼                   ▼
        ┌───────────┐       ┌──────────┐
        │ responder │       │ clarify  │   "can you rephrase / add detail?"
        └─────┬─────┘       └────┬─────┘
              │                  │
       Claude if key set,        │
       else Markdown of the      │
       retrieved sections        │
              └────────┬─────────┘
                       ▼
                     answer
```

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
│   ├── tests/                # pytest suite (19 tests)
│   ├── requirements.txt
│   ├── requirements-dev.txt  # requirements.txt + pytest
│   └── cloudnest_docs/      # the knowledge base — plain markdown
│       ├── 01_product_overview.md
│       ├── 02_pricing_billing.md
│       ├── 03_account_management.md
│       ├── 04_technical_setup.md
│       ├── 05_troubleshooting.md
│       ├── 06_security_privacy.md
│       └── 07_general_faq.md
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
  "sources": ["Plans and Pricing"]
}
```

`clarified` tells you which branch of the graph answered (`true` means it asked you to add detail instead of answering). `sources` lists the titles of the doc sections the answer was cited from, empty on a clarify. Neither is shown in the chat window; both exist for logging and debugging.

**`GET /api/health`** — returns `{ "mode": "claude" }` if a key is configured, `{ "mode": "extractive" }` otherwise, plus a `retrieval` field: `"semantic"` when the embedding index is loaded, or `"lexical"` when it has fallen back to keyword retrieval. The UI uses `mode` to set the status badge.

## How retrieval actually works

Worth a note since it's the core of the thing and there's nothing hidden.

Each doc gets split into sections on its markdown headings. Every section is embedded once, ahead of time, by a small local model (`all-MiniLM-L6-v2`, an int8 ONNX export — no API key, no per-query cost, runs offline) and the resulting vectors are committed as `backend/index.npz`. When a question comes in, only the question is embedded, and sections are ranked by cosine similarity against it. The top 5 (`TOP_K` in `app.py`) become the context, and confidence is the similarity of the best one. Because it matches on meaning rather than shared words, "how much will I be charged if I add a teammate mid-cycle" finds the pricing section even though it shares no keywords with it.

Five rather than three because some answers genuinely need two sections at once, e.g. "does emptying Trash free up enough space to resume syncing" needs both the Trash-recovery section and the storage-quota section, and the second one doesn't always rank in the top 3 under a model this small. The API still only cites the best 3 by score, so the wider recall doesn't clutter the sources it reports.

If that top score clears `0.30` (a constant near the top of `app.py`, chosen by measuring the gap between in-scope and out-of-scope questions — see `backend/calibrate.py`), the responder runs. If it doesn't, you get the clarify prompt instead. It keeps the agent from confidently answering questions the docs don't actually cover.

If the embedding model or index can't be loaded, retrieval falls back to a keyword scorer (the previous approach — token overlap with a small synonym map), so the app still answers with no model and no API key at all.

### Rebuilding the search index

The section embeddings are generated ahead of time. After editing anything in `backend/cloudnest_docs/`, regenerate the index:

    python backend/build_index.py

Skipping this is safe but degrading: `app.py` fingerprints the docs and, on a mismatch, falls back to keyword retrieval rather than searching outdated vectors. Run `python backend/build_index.py --calibrate` to also re-check the confidence threshold against the new content.

## Deployment

`vercel.json` is set up to build the frontend as a static site and run `backend/index.py` as a Python serverless function, with `/api/*` routed to the backend and everything else falling through to the SPA. `ANTHROPIC_API_KEY` is set as an environment variable in the Vercel project (Preview and Production). The project is linked to GitHub, so every push to `main` triggers a fresh production deployment automatically.

Live at **[cloudnest-nine.vercel.app](https://cloudnest-nine.vercel.app)**. `/api/health` there currently reports `{"mode": "claude", "retrieval": "semantic"}`, so the embedding index and the API key are both loading correctly in production. The deployed function, model and all, measures 79.83 MB per Vercel's own build output, comfortably inside the 250 MB serverless function limit.

## Things I'd add next

- A stronger embedding model once the doc set outgrows what a 384-dim int8 model can rank well; `TOP_K` and the calibrated threshold are both compensating for its ceiling.
- Persist conversations server-side so history doesn't have to round-trip through the browser.
- Re-run `backend/calibrate.py` against real production questions once there's traffic. The current threshold is calibrated from a 16-question probe set, which is a reasonable start but not the same as live data.
- Frontend tests. The backend has 19 pytest cases around the router, retriever, and index; the React side is only checked by hand.
