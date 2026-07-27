# CloudNest Support Agent

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-1C3C3C?logo=langchain&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?logo=vite&logoColor=white)
![Vercel](https://img.shields.io/badge/Deploy-Vercel-000000?logo=vercel&logoColor=white)

**Live:** [cloudnest-nine.vercel.app](https://cloudnest-nine.vercel.app)

A support chatbot for a fictional cloud-storage product called CloudNest. You type a question in plain English, and it figures out whether you're asking about billing or something technical, pulls the relevant bits out of the product docs, and answers you like a real support engineer would. It runs on a small LangGraph state machine on the backend and a React chat window on the front.

Retrieval is real semantic search, and it's self-contained: every doc section is embedded ahead of time by a small local model that runs inside the function itself, then matched against your question by cosine similarity. Search costs nothing per query, works offline, and has no third-party service in the path to go down. See [How retrieval actually works](#how-retrieval-actually-works) below for the full story, including the keyword-based fallback it degrades to if the model or index can't load. If the API key is present it uses Claude to phrase the final answer, written to sound like a person rather than an AI model; if not, it hands you back the matching sections as clean Markdown. Either way you always get an answer.

## What it does

- **Routes each question** to billing, technical, or general before it does anything else, so the retriever knows which docs to favor.
- **Retrieves by meaning, not keywords.** Doc sections are embedded once ahead of time and ranked by cosine similarity against the question, so "how much will I be charged if I add a teammate mid-cycle" finds the pricing section even though it shares no words with it. Falls back to keyword matching if the embedding model or index isn't available.
- **Answers with Claude when a key is set**, grounded strictly in the retrieved sections and written in a natural, human support-agent voice: no mention of documentation, retrieval, confidence, or routing, no emoji, no em dashes. No key, and it falls back to the retrieved sections as clean Markdown instead of failing.
- **Formats for reading**, not just talking: a direct answer up front, Markdown tables for comparisons, bullets for options, numbered steps for procedures.
- **Bails out honestly** when it isn't confident. Below a threshold calibrated against real questions (see `backend/calibrate.py`), it asks you to add detail rather than guessing.
- **Keeps retrieval internals out of the UI.** The API still returns the route, confidence, and cited sources for every answer, but the chat window shows only the answer itself.
- **Remembers the conversation.** The CLI keeps context across turns via a LangGraph checkpointer; the web version carries history in the browser since the serverless function is stateless.
- **Understands follow-ups.** A reply like "does it cost extra" carries no topic word of its own. Retrieval folds in the previous turn before embedding the question, so it still finds the right section instead of scoring below the confidence threshold or matching the wrong one. Greetings are skipped when folding, since a hello has no topic to inherit and mixing one in only drags the score down.
- **Says hello back.** A message that is only a greeting gets greeted and invited to ask, rather than the out-of-scope reply. The check is narrow enough that "hi, my sync keeps failing" is still answered as the real question it is.

## How it's wired

The backend is a LangGraph `StateGraph` with eight nodes: a router, two retrievers that run in parallel, a fusion step, and four terminal nodes. What picks between those four isn't a node at all. It's a conditional edge off `fusion`, which is why the count is eight and not nine.

```
                            question
                               │
                               ▼
                          ┌─────────┐
                          │ router  │  billing / technical / general
                          └────┬────┘
                  ┌────────────┴────────────┐        (in parallel)
                  ▼                         ▼
        ┌───────────────────┐     ┌───────────────────┐
        │ semantic_retriever│     │ lexical_retriever │
        │   cosine on 46    │     │   token overlap   │
        │  embedded sections│     │  + synonym map    │
        └─────────┬─────────┘     └─────────┬─────────┘
                  └────────────┬────────────┘
                               ▼
                          ┌─────────┐
                          │ fusion  │  RRF merge; confidence = cosine top-1
                          └────┬────┘
                               │  picker (a conditional edge, not a node)
            ┌────────────┬─────┴──────┬────────────┐
            ▼            ▼            ▼            ▼
      ┌───────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐
      │ responder │ │ clarify │ │ deflect │ │ escalate │
      └───────────┘ └─────────┘ └─────────┘ └────┬─────┘
                                                 ▼
                                            ticket ─► Postgres ─► /admin
```

Which branch the picker takes:

| Branch | When |
|---|---|
| `responder` | confidence ≥ `0.30`, or a lexical rescue pulls a borderline score up |
| `clarify` | under `0.30`. A greeting gets greeted, anything under the `0.27` floor gets the out-of-scope reply, and a borderline in-scope question gets asked for detail |
| `deflect` | a first explicit request for a human; offers to help before handing over |
| `escalate` | the user insisted after a deflect, or missed borderline twice running |

A first explicit request for a human ("let me talk to a person", "get me an agent", "connect me to a representative") doesn't escalate straight away. It routes to the deflect node, which offers to help right there first, since the bot usually can. Only if the user still asks for a person after that does the gate escalate. Escalation also fires on a repeated borderline miss (a real, in-scope question the bot couldn't get confident about two turns running). The escalate node builds a support ticket (id, timestamp, question, category, confidence, the full conversation, and a reason) and hands off rather than guessing further. That ticket comes back from the API and is written to a Postgres `tickets` table on the way out, so the handoff survives the request. Support admins read the queue at [`/admin`](#admin-view).

### Admin view

Every escalated ticket is persisted to Postgres by `backend/tickets_store.py`, the only module that touches the database. `/admin` is a separate page (its own Vite entry and bundle, so the chat bundle is untouched) showing the tickets newest-first, each row expandable to the full conversation that led to the handoff. It's read-only: there's no way to edit or delete a ticket from it.

### Signing in

There's **one login page**, and it serves both roles. The support desk is closed until you sign in; whoever you sign in as is the name the admin later sees on your ticket.

The trick is that the page doesn't decide who you are. The server does. Whatever you type is tried against `GET /api/tickets`: a 200 means those were the admin credentials and you land on the ticket queue, anything else signs you in as an ordinary user and opens the chat. So the admin username never appears in the frontend bundle, and there's no "are you an admin?" checkbox to lie to.

Admin credentials are `ADMIN_USERNAME` (default `admin`) and `ADMIN_PASSWORD`, sent as `X-Admin-User` and `X-Admin-Token` headers, never in the URL, and compared with `hmac.compare_digest` in a single expression, so a wrong username and a wrong password are indistinguishable from outside, both just 401. `/admin` has no login form of its own: without an admin session it bounces you to the front page.

**Ordinary sign-in is deliberately fake.** Any username and password is accepted and no account exists anywhere on the server. It exists to put a name on a ticket so the admin view has something real to show. It is not authentication, and nothing a user sees is protected by it. Only the admin half is checked, and only server-side.

The session is held in `localStorage` under `cloudnest.session`, so you stay signed in across tabs and browser restarts until you press Sign out. For an admin that means the password is written to disk rather than kept for the tab, which is the trade for not retyping it every visit, and worth knowing before using a real password on a shared machine.

### What the admin sees

Each ticket row carries the username that raised it, and expanding one shows who they were, when they signed in, and the full conversation with their name on their turns. Tickets raised before the login flow existed, or by anyone hitting the API directly, show *not signed in* rather than a fake name. The user columns are nullable and `init_db()` adds them to an existing table with `ALTER TABLE … ADD COLUMN IF NOT EXISTS`, so an existing `tickets` table upgrades in place.

With no database configured at all, nothing breaks: `save_ticket` no-ops, `list_tickets` returns an empty list, and chat runs exactly as before. Persistence is also best-effort inside `/api/chat`: if the write fails, the user still gets their answer.

`index.py` wraps that same graph in a FastAPI app so the React frontend can talk to it over `/api/chat`. `app.py` can also be run on its own as a command-line chat loop, which is the quickest way to poke at the logic without touching the frontend.

## Project layout

```
Alliedworks/
├── backend/
│   ├── app.py               # the LangGraph agent + CLI entry point
│   ├── index.py             # FastAPI wrapper (/api/chat, /api/health, /api/tickets)
│   ├── tickets_store.py     # Postgres persistence for escalation tickets
│   ├── embed.py             # local ONNX embedding function
│   ├── build_index.py       # offline indexer; embeds cloudnest_docs/ into index.npz
│   ├── calibrate.py         # measures the confidence threshold from real questions
│   ├── index.npz            # committed embedding index (vectors + chunk metadata)
│   ├── model/                # vendored int8 embedding model + tokenizer
│   ├── tests/                # pytest suite (78 tests)
│   ├── requirements.txt
│   └── requirements-dev.txt  # requirements.txt + pytest + httpx
├── cloudnest_docs/          # the knowledge base, plain markdown
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
│   │   ├── Login.jsx        # the one sign-in page, for users and admins alike
│   │   ├── session.js       # session storage + the server-decides-the-role check
│   │   ├── main.jsx
│   │   ├── app.css
│   │   ├── admin.jsx        # read-only admin view of escalated tickets
│   │   └── admin.css
│   ├── index.html
│   ├── admin.html           # second Vite entry, served at /admin
│   ├── package.json
│   └── vite.config.js       # two build inputs + dev proxy to the backend on :8000
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

Both `.env` and `.env.local` are read, `.env.local` taking precedence, and a real shell variable beating both. That means `vercel env pull`, which writes `.env.local`, drops the Neon connection string straight into local dev with nothing to copy by hand. Surrounding quotes are stripped, since the CLI writes `KEY="value"` and a quoted DSN can never connect.

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

Three endpoints, all under `/api`.

**`POST /api/chat`**

```json
{
  "message": "How much does the Pro plan cost?",
  "history": []
}
```

`history` is a list of `{ "role": "user" | "assistant", "content": "..." }` objects. The frontend keeps track of it and sends the running conversation with each request, since the backend doesn't hold session state of its own. Two optional fields, `username` and `signed_in_at`, carry the signed-in user; they're stitched onto the ticket on escalation so the admin can see who raised it, and omitting them just leaves the ticket unattributed.

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

`clarified` and `escalated` tell you which branch of the graph answered (`clarified: true` means it asked you to add detail instead of answering; `escalated: true` means it handed the conversation off instead). `sources` lists the titles of the doc sections the answer was cited from, and it's empty on either a clarify or an escalate, since neither is a doc-grounded answer. `ticket` carries the escalation record (id, timestamp, question, category, confidence, the full conversation, a reason, and the signed-in `username` / `signed_in_at`) when `escalated` is true, otherwise `null`. None of these are shown in the chat window; all exist for logging and debugging. The ticket is also persisted to Postgres on escalate and readable at [`/admin`](#admin-view).

**`GET /api/health`** returns `{ "mode": "claude" }` if a key is configured, `{ "mode": "extractive" }` otherwise, plus a `retrieval` field: `"semantic"` when the embedding index is loaded, or `"lexical"` when it has fallen back to keyword retrieval. The UI uses `mode` to set the status badge.

**`GET /api/tickets`** is the escalated ticket list behind [`/admin`](#admin-view), newest first: `{ "tickets": [ ... ] }`, each entry shaped like the `ticket` object above. Authenticated with `X-Admin-User` (matched against `ADMIN_USERNAME`, default `admin`) and `X-Admin-Token` (matched against `ADMIN_PASSWORD`); 401 if either is missing or wrong, 503 if `ADMIN_PASSWORD` isn't set at all. Returns an empty list when no database is configured.

## How retrieval actually works

Worth a note since it's the core of the thing and there's nothing hidden.

Each doc gets split into sections on its markdown headings. Every section is embedded once, ahead of time, by a small local model (`all-MiniLM-L6-v2`, an int8 ONNX export that runs offline at zero per-query cost) and the resulting vectors are committed as `backend/index.npz`. When a question comes in, only the question is embedded, and sections are ranked by cosine similarity against it. All 46 sections (`TOP_K` in `app.py`, set to the corpus size while it's under `SMALL_CORPUS_LIMIT`) become the context, and confidence is the similarity of the best one. Because it matches on meaning rather than shared words, "how much will I be charged if I add a teammate mid-cycle" finds the pricing section even though it shares no keywords with it.

The whole corpus rather than a fixed cutoff, because a question can genuinely need several sections spread far apart in the ranking: "I need API access, EU residency, bank transfer, and CLI automation" touches 4 sections, and 2 of them ranked outside any small top-N against that query (10th and 22nd out of 46). At ~1,800 tokens for the entire corpus, sending everything costs nothing meaningful, and it's simpler than tuning a cutoff that would just need raising again for the next multi-fact question. `SMALL_CORPUS_LIMIT` (150 chunks) marks where that stops being true; past it, `TOP_K` falls back to a fixed `FALLBACK_TOP_K`, which is an unvalidated placeholder rather than a tuned value, on the same reasoning that any fixed cutoff can drop a section the way the old `TOP_K=5` did. This only changes what an already-confident answer gets to draw from. `CONFIDENCE_THRESHOLD` routing still looks at just the single best score, and the API still cites only the best 3 by score, so the wider recall doesn't clutter the sources it reports.

If that top score clears `0.30` (a constant near the top of `app.py`, chosen by measuring the gap between in-scope and out-of-scope questions; see `backend/calibrate.py`), the responder runs. If it doesn't, you get the clarify prompt instead, or an escalation (see below). It keeps the agent from confidently answering questions the docs don't actually cover.

Retrieval actually runs as two parallel branches: the semantic ranking above, and a lexical (token-overlap) ranking of the same corpus. `fuse_results` merges the two full rankings with Reciprocal Rank Fusion (RRF). A chunk scores `1 / (60 + rank)` per list it appears in, so a section ranked highly in either list surfaces even if the other list buries it, and that fused order is what's sent to the LLM and cited from. Confidence itself does not change: it's still the semantic cosine top-1 score, exactly as above. Fusion only reorders context and citations; it can also *rescue* a borderline-confidence answer when the lexical branch has a strong exact-keyword hit (two or more distinct query terms matched in the top chunk), which is how a query like "cloudnest-cli" still gets answered even if the embedding alone scored it under `0.30`.

Below the gate sits a second, lower floor: `ESCALATE_FLOOR` (0.27, set in `app.py` just above the measured out-of-scope ceiling from `backend/calibrate.py`). A question scoring between the floor and `0.30` is borderline and in-scope. The first time, it gets the clarify prompt; if the *next* turn is still borderline after that, it escalates to a support ticket instead of clarifying again. Anything scoring below the floor is treated as off-topic noise and can only clarify, never escalate, no matter how many times it repeats, so two turns of "cat mouse banana" never opens a ticket.

One exception to that below-the-floor path: a plain hello. "hello my name is likith" scores 0.08 and used to get the out-of-scope guard, which is technically correct and reads like a door closing. A message that is *only* a greeting or an introduction gets greeted back and invited to ask instead. The check is deliberately narrow: any billing or technical word anywhere in the message, or more than eight words, and it falls through to the normal path, so "hi, my sync keeps failing" is answered (0.64) rather than waved at. That's about score-driven escalation only: an explicit request to reach a human is handled separately and ignores the score entirely, since the floor guards against off-topic noise, not against a deliberate ask for a person. The first such request deflects, and a second one hands off.

A request for a human is matched by intent rather than by a fixed phrase list: a word meaning *a human* (`human`, `agent`, `representative`, `person`, `someone`…) together with a word meaning *hand me over* (`connect`, `talk`, `speak`, `transfer`, `get`, `put`…). Exact-phrase matching kept missing natural variants, since "connect to human" and "connect me to a representative" are the same ask, while a question that merely contains "person" with no handoff verb ("am I the only person seeing this") stays answerable.

Before any of that, both `router()` and `retriever()` run the question through `contextualize_query()`, which folds the previous user turn onto the current one (capped at `CONTEXT_CHAR_CAP`, 200 characters) before it's embedded. A bare follow-up like "does it cost extra" carries no topic word of its own. Alone it scores 0.285 (just under the threshold) and points at the wrong section; folded onto the turn before it ("I want to add two-factor authentication"), it scores 0.55 and finds the right one. Only the retrieval-facing query is folded. Claude still receives the full, unfolded conversation for generation, since it was never confused about "it"; only retrieval was.

If the embedding model or index can't be loaded, retrieval falls back to a keyword scorer (the previous approach: token overlap with a small synonym map), so the app still answers with no model and no API key at all.

### Rebuilding the search index

The section embeddings are generated ahead of time. After editing anything in `cloudnest_docs/`, regenerate the index:

    python backend/build_index.py

Skipping this is safe but degrading: `app.py` fingerprints the docs and, on a mismatch, falls back to keyword retrieval rather than searching outdated vectors. Run `python backend/build_index.py --calibrate` to also re-check the confidence threshold against the new content.

## Deployment

`vercel.json` is set up to build the frontend as a static site and run `backend/index.py` as a Python serverless function, with `/api/*` routed to the backend and everything else falling through to the SPA. `ANTHROPIC_API_KEY` is set as an environment variable in the Vercel project (Preview and Production). The project is linked to GitHub, so every push to `main` triggers a fresh production deployment automatically.

Two more environment variables turn on ticket persistence and the admin view. Both are optional. Without them the app runs exactly as it did before, it just stores nothing and keeps `/admin` closed.

- **`DATABASE_URL`** (or **`POSTGRES_URL`**): add a Neon Postgres integration from the Vercel Marketplace, which injects the connection string into the project. Either name works; `DATABASE_URL` wins if both are set. The `tickets` table is created automatically on first boot by `init_db()`, so there's no migration step.
- **`ADMIN_PASSWORD`**: set it in the Vercel project env (and in the local `.env` for dev) to enable `/admin`. Without it, `GET /api/tickets` returns 503 and the admin page says so.
- **`ADMIN_USERNAME`**: optional, defaults to `admin`. Set it only if you want a different login name.

Post-deploy check: open `/admin`, sign in with `ADMIN_PASSWORD`, and confirm the list loads. Then insist on a human in the chat (ask once, decline the offer to help, ask again) and confirm the new ticket appears.

Live at **[cloudnest-nine.vercel.app](https://cloudnest-nine.vercel.app)**. `/api/health` there currently reports `{"mode": "claude", "retrieval": "semantic"}`, so the embedding index and the API key are both loading correctly in production. The deployed function, model and all, measures 56.99 MB per `vercel inspect`, comfortably inside the 250 MB serverless function limit.

## Things I'd add next

- Make the admin queue workable rather than readable. Tickets persist and `/admin` lists them, but there's no way to assign, annotate, or close one, and no per-admin login, just a single shared password. Real accounts and ticket state are the next step.
- A stronger embedding model, and a real retrieval strategy (a stronger fusion/rerank than plain RRF, not just a bigger fixed cutoff) once the corpus outgrows `SMALL_CORPUS_LIMIT`. `FALLBACK_TOP_K` in `app.py` is a placeholder, not a tuned value: a fixed cutoff has the same failure mode the whole-corpus change just fixed, just at a different scale.
- Persist conversations server-side so history doesn't have to round-trip through the browser.
- Re-run `backend/calibrate.py` against real production questions once there's traffic. The current threshold is calibrated from a 16-question probe set, which is a reasonable start but not the same as live data.
- Frontend tests. The backend has 78 pytest cases around the router, the parallel retrievers, fusion, the gate, the index, the ticket store, and the API's auth gate; the React side, chat and admin both, is only checked by hand.
