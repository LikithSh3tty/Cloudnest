import os
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastapi import FastAPI
from pydantic import BaseModel

from app import build_app, sources_from_context

graph = build_app()
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
    # sections as chips, so extra recall doesn't clutter the UI with sections
    # the answer barely touched. No sources on a clarify (chunks were sub-threshold).
    sources = [] if result["clarified"] else sources_from_context(result["context"])[:3]
    return {"answer": result["messages"][-1]["content"],
            "category": result["category"],
            "confidence": result["confidence"],
            "clarified": result["clarified"],
            "sources": sources}


@app.get("/api/health")
def health():
    from app import INDEX
    return {"mode": "claude" if os.environ.get("ANTHROPIC_API_KEY") else "extractive",
            "retrieval": "semantic" if INDEX is not None else "lexical"}
