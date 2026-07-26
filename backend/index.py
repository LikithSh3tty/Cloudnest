import hmac
import os
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

import tickets_store
from app import build_app, sources_from_context

graph = build_app()
tickets_store.init_db()  # idempotent; no-op when no database is configured
app = FastAPI(title="CloudNest support API")


class ChatIn(BaseModel):
    message: str
    history: list[dict] = []


@app.post("/api/chat")
def chat(body: ChatIn):
    messages = body.history + [{"role": "user", "content": body.message}]
    config = {"configurable": {"thread_id": uuid4().hex}}
    result = graph.invoke({"messages": messages}, config)
    # Retrieve wide for the LLM but cite narrow: show only the 3 best-ranked
    # sections as chips. No sources on a clarify (sub-threshold) or an
    # escalate (the answer is a handoff, not a doc-grounded reply).
    escalated = result.get("escalate", False)
    ticket = result.get("ticket")
    if escalated and ticket:
        # Best-effort: a persistence failure must never turn a chat reply into
        # a 500. save_ticket already swallows its own errors; this guards
        # against anything unexpected too.
        try:
            tickets_store.save_ticket(ticket)
        except Exception as exc:
            print(f"ticket persist failed: {type(exc).__name__}")
    sources = ([] if result["clarified"] or escalated
               else sources_from_context(result["context"])[:3])
    return {"answer": result["messages"][-1]["content"],
            "category": result["category"],
            "confidence": result["confidence"],
            "clarified": result["clarified"],
            "escalated": escalated,
            "ticket": ticket,
            "sources": sources}


@app.get("/api/health")
def health():
    from app import INDEX
    return {"mode": "claude" if os.environ.get("ANTHROPIC_API_KEY") else "extractive",
            "retrieval": "semantic" if INDEX is not None else "lexical"}


@app.get("/api/tickets")
def tickets(x_admin_user: str | None = Header(default=None),
            x_admin_token: str | None = Header(default=None)):
    password = os.environ.get("ADMIN_PASSWORD")
    if not password:
        # Closed by default: no admin password configured means no admin access.
        raise HTTPException(status_code=503, detail="admin not configured")
    username = os.environ.get("ADMIN_USERNAME", "admin")
    # Both halves must match, and both are compared in constant time. Checking
    # them in one expression avoids answering "was the username right?"
    # separately from "was the password right?".
    ok = (x_admin_user and x_admin_token
          and hmac.compare_digest(x_admin_user, username)
          and hmac.compare_digest(x_admin_token, password))
    if not ok:
        raise HTTPException(status_code=401, detail="unauthorized")
    return {"tickets": tickets_store.list_tickets()}
