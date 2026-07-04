# CloudNest Support Agent

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-1C3C3C?logo=langchain&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?logo=vite&logoColor=white)
![Vercel](https://img.shields.io/badge/Deploy-Vercel-000000?logo=vercel&logoColor=white)

A support chatbot for a fictional cloud-storage product called CloudNest. You type a question in plain English, and it figures out whether you're asking about billing or something technical, pulls the relevant bits out of the product docs, and answers you. It runs on a small LangGraph state machine on the backend and a React chat window on the front.

The whole thing is built around a folder of markdown docs. There's no vector database and no embeddings — retrieval is plain keyword matching with a bit of scoring on top. That was a deliberate choice: it's easy to read, easy to debug, and it's more than enough for a doc set this size. If the API key is present it uses Claude to phrase the final answer; if not, it just hands you back the matching doc sections. Either way you always get an answer.

## What it does

- **Routes each question** to billing, technical, or general before it does anything else, so the retriever knows which docs to favor.
- **Retrieves from local markdown**, splitting each doc into sections by heading and scoring them against your question's keywords (with synonym folding and stopword removal, so "cost" and "price" land in the same place).
- **Answers with Claude when a key is set**, grounded strictly in the retrieved sections — no key, and it falls back to returning the raw doc snippets instead of failing.
- **Bails out honestly** when it isn't confident. Below a set threshold it asks you to rephrase or add detail rather than making something up.
- **Shows its work** — every reply comes back tagged with the route it took and a confidence score, both of which are visible in the UI.
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
                  │ retriever │   score doc sections, compute confidence
                  └─────┬─────┘
                        │
             confidence ≥ 0.25 ?
              ┌─────────┴─────────┐
              │ yes               │ no
              ▼                   ▼
        ┌───────────┐       ┌──────────┐
        │ responder │       │ clarify  │   "can you rephrase / add detail?"
        └─────┬─────┘       └────┬─────┘
              │                  │
       Claude if key set,        │
       else raw doc sections     │
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
│   ├── requirements.txt
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

The key is optional. Without it the agent still runs — it just returns the matched doc sections verbatim instead of a rephrased answer. Handy if you want to see exactly what retrieval is pulling.

Then start the API:

```bash
uvicorn index:app --reload --port 8000
```

Or skip the server entirely and chat in the terminal:

```bash
python app.py
```

The CLI prints the route and confidence before each answer, which is useful when you're tuning the keyword lists or the threshold.

### 2. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Vite serves the UI on `http://localhost:5173` and proxies `/api` calls through to the backend on `:8000`, so you don't have to deal with CORS. The header shows whether the assistant is online (key present) or running in fallback mode.

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
  "answer": "The Pro plan is ...",
  "category": "billing",
  "confidence": 0.67
}
```

**`GET /api/health`** — returns `{ "mode": "claude" }` if a key is configured, `{ "mode": "extractive" }` otherwise. The UI uses this to set the status badge.

## How retrieval actually works

Worth a note since it's the core of the thing and there's nothing hidden.

Each doc gets split into sections on its markdown headings. When a question comes in, both the question and every section get tokenized — lowercased, stripped of stopwords, with a small synonym map so a handful of common variants collapse together (`cost`/`costs`/`pricing` → `price`, and so on). Sections are then scored by how many query terms they contain and how often, with a 2× boost for docs that match the routed category. The top few sections above zero become the context, and confidence is roughly the share of your query's terms that the best section covered.

If that top score clears `0.25` (a constant near the top of `app.py`, easy to tweak), the responder runs. If it doesn't, you get the clarify prompt instead. It's a blunt heuristic, but it keeps the agent from confidently answering questions the docs don't actually cover.

## Deployment

`vercel.json` is set up to build the frontend as a static site and run `backend/index.py` as a Python serverless function, with `/api/*` routed to the backend and everything else falling through to the SPA. Add `ANTHROPIC_API_KEY` as an environment variable in the Vercel project settings and push.

## Things I'd add next

- Swap the keyword scoring for real embeddings once the doc set outgrows keyword matching.
- Make the confidence threshold configurable instead of hardcoded.
- Persist conversations server-side so history doesn't have to round-trip through the browser.
- A proper test suite around the router and retriever — right now it's verified by hand through the CLI.
