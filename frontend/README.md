# CloudNest Support — chat interface

React (Vite) chat UI over the LangGraph support router. The core `support_router/app.py`
is unchanged; `support_router/server.py` wraps it with a two-endpoint FastAPI service.

## Run (two terminals)

```powershell
# 1. API on :8000
cd support_router
uvicorn server:api --port 8000

# 2. UI on :5173 (proxies /api -> :8000)
cd frontend
npm run dev
```

Open http://localhost:5173. The header shows "assistant online" when the API key is
loaded. Each reply displays the router's category and retrieval confidence; amber
replies are the clarify path. Multi-turn state is per browser tab (one thread_id per
page load — reload to start a fresh conversation).
