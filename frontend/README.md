# CloudNest Support — chat interface

React (Vite) chat UI over the LangGraph support router. The core `support_router/app.py`
is unchanged; `api/index.py` wraps it with a two-endpoint FastAPI service that runs
identically on Vercel and locally. The API is stateless — the browser carries the
conversation history with each request — so multi-turn follow-ups work on serverless.

## Run locally (two terminals, from the repo root)

```powershell
# 1. API on :8000
python -m uvicorn api.index:app --port 8000

# 2. UI on :5173 (proxies /api -> :8000)
cd frontend
npm run dev
```

Open http://localhost:5173. The header shows "assistant online" when the API key is
loaded. Each reply displays the router's category and retrieval confidence; amber
replies are the clarify path. Reload the page to start a fresh conversation.

## Deploy on Vercel

Import the GitHub repo in Vercel (framework preset: Other — `vercel.json` drives the
build). Set one environment variable in the project settings:

- `ANTHROPIC_API_KEY` — without it the responder falls back to extractive answers.

`vercel.json` builds the frontend to `frontend/dist`, serves `api/index.py` as a Python
function, and rewrites `/api/*` to it. `support_router/` and `cloudnest_docs/` are
bundled with the function via `includeFiles`.
