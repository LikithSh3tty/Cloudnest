"""HTTP wrapper for the CloudNest support graph. Reuses app.py unchanged.

Run from the support_router directory:  uvicorn server:api --port 8000
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import build_app

graph = build_app()
api = FastAPI(title="CloudNest support API")
api.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])


class ChatIn(BaseModel):
    message: str
    thread_id: str = "web"


@api.post("/chat")
def chat(body: ChatIn):
    config = {"configurable": {"thread_id": body.thread_id}}
    result = graph.invoke({"messages": [{"role": "user", "content": body.message}]}, config)
    return {"answer": result["messages"][-1]["content"],
            "category": result["category"],
            "confidence": result["confidence"]}


@api.get("/health")
def health():
    return {"mode": "claude" if os.environ.get("ANTHROPIC_API_KEY") else "extractive"}
