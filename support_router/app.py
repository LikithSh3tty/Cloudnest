"""CloudNest multi-agent support router (LangGraph) — design: docs/superpowers/specs/."""
import os
import re
from operator import add
from pathlib import Path
from typing import Annotated, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if ENV_FILE.exists():
    for _line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        _key, _, _value = _line.partition("=")
        if _key.strip() and not _line.lstrip().startswith("#"):
            os.environ.setdefault(_key.strip(), _value.strip())
DOCS_DIR = Path(__file__).resolve().parent.parent / "cloudnest_docs"
CONFIDENCE_THRESHOLD = 0.25
CATEGORY_DOCS = {
    "billing": {"02_pricing_billing.md", "03_account_management.md"},
    "technical": {"04_technical_setup.md", "05_troubleshooting.md"},
}
BILLING_WORDS = {
    "price", "pricing", "plan", "plans", "pay", "payment", "bill", "billing",
    "invoice", "refund", "subscription", "upgrade", "downgrade", "charge",
    "charged", "cost", "cancel", "renewal", "discount", "trial", "card",
}
TECH_WORDS = {
    "install", "installation", "sync", "syncing", "error", "crash", "setup", "backup",
    "restore", "version", "versioning", "upload", "download", "slow", "fail", "failed",
    "bug", "app", "device", "login", "log", "connect", "encrypted", "encryption",
    "vault", "folder", "file", "conflict", "bandwidth", "network", "configure",
    "settings", "preferences", "speed", "limit", "limits", "proxy",
}
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "can",
    "i", "my", "me", "you", "your", "it", "its", "on", "in", "of", "to", "and",
    "or", "for", "with", "how", "what", "why", "when", "where", "not", "no",
    "have", "has", "be", "will", "about", "this", "that", "there", "they",
    "them", "which", "much", "cloudnest",
}
SYNONYMS = {"cost": "price", "costs": "price", "pricing": "price"}

def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return [SYNONYMS.get(w, w) for w in words if w not in STOPWORDS]

def load_chunks() -> list[dict]:
    chunks = []
    for doc in sorted(DOCS_DIR.glob("*.md")):
        for section in re.split(r"\n(?=#{1,3} )", doc.read_text(encoding="utf-8")):
            title = section.strip().splitlines()[0].lstrip("# ").strip()
            chunks.append({"doc": doc.name, "title": title, "text": section.strip()})
    return chunks

CHUNKS = load_chunks()

class State(TypedDict):
    messages: Annotated[list[dict], add]
    category: str
    context: list[dict]
    confidence: float

def router(state: State) -> dict:
    words = set(tokenize(state["messages"][-1]["content"]))
    billing, technical = len(words & BILLING_WORDS), len(words & TECH_WORDS)
    if billing == technical == 0:
        return {"category": "general"}
    return {"category": "billing" if billing >= technical else "technical"}

def retriever(state: State) -> dict:
    terms = tokenize(state["messages"][-1]["content"])
    preferred = CATEGORY_DOCS.get(state["category"], set())
    scored = []
    for chunk in CHUNKS:
        chunk_terms = tokenize(chunk["text"])
        matched = {t for t in terms if t in chunk_terms}
        boost = 2 if chunk["doc"] in preferred else 1
        scored.append((boost * sum(chunk_terms.count(t) for t in matched), len(matched), chunk))
    scored.sort(key=lambda s: (-s[0], s[2]["doc"]))
    top = [c for score, _, c in scored[:3] if score > 0]
    confidence = scored[0][1] / len(terms) if terms and scored else 0.0
    return {"context": top, "confidence": confidence}

def llm_answer(question: str, context: str, history: list[dict]) -> str | None:
    try:
        import anthropic
        client = anthropic.Anthropic()
        prompt = f"Documentation context:\n{context}\n\nCustomer question: {question}"
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1024,
            system="You are CloudNest's customer support agent. Answer using only "
                   "the provided documentation context. Be concise and friendly; plain "
                   "text, no emoji. If the context does not cover the question, say so.",
            messages=history[:-1] + [{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text") or None
    except Exception:
        return None  # no credentials / network — extractive fallback answers

def responder(state: State) -> dict:
    question = state["messages"][-1]["content"]
    context = "\n\n".join(f"[{c['doc']} - {c['title']}]\n{c['text']}" for c in state["context"])
    answer = llm_answer(question, context, state["messages"])
    if answer is None:
        parts = [f"From our docs ({c['doc']} - {c['title']}):\n{c['text']}" for c in state["context"]]
        answer = "\n\n".join(parts)
    return {"messages": [{"role": "assistant", "content": answer}]}

def clarify(_state: State) -> dict:
    msg = ("I couldn't find a confident answer for that in our documentation. Could you "
           "rephrase or add detail (your plan, your device, or the exact error you see)?")
    return {"messages": [{"role": "assistant", "content": msg}]}

def build_app():
    graph = StateGraph(State)
    for fn in (router, retriever, responder, clarify):
        graph.add_node(fn.__name__, fn)
    graph.add_edge(START, "router")
    graph.add_edge("router", "retriever")
    def picker(s: State) -> str:
        return "responder" if s["confidence"] >= CONFIDENCE_THRESHOLD else "clarify"
    graph.add_conditional_edges("retriever", picker, ["responder", "clarify"])
    graph.add_edge("responder", END)
    graph.add_edge("clarify", END)
    return graph.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    app = build_app()
    config = {"configurable": {"thread_id": "cli-session"}}  # persists across turns
    mode = "Claude responder" if os.environ.get("ANTHROPIC_API_KEY") else \
        "extractive fallback (no ANTHROPIC_API_KEY in shell or .env)"
    print(f"CloudNest support (type 'quit' to exit) — mode: {mode}")
    while True:
        try:
            query = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in {"quit", "exit"}:
            break
        result = app.invoke({"messages": [{"role": "user", "content": query}]}, config)
        print(f"\n[route: {result['category']} | confidence: {result['confidence']:.2f}]")
        print(result["messages"][-1]["content"])
