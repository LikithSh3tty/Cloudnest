import hashlib
import json
import os
import re
from operator import add
from pathlib import Path
from typing import Annotated, TypedDict
import numpy as np
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if ENV_FILE.exists():
    for _line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        _key, _, _value = _line.partition("=")
        if _key.strip() and not _line.lstrip().startswith("#"):
            os.environ.setdefault(_key.strip(), _value.strip())
DOCS_DIR = Path(__file__).resolve().parent / "cloudnest_docs"
# cosine similarity, chosen from backend/calibrate.py output
CONFIDENCE_THRESHOLD = 0.30
# sections sent to the LLM. Wider than it looks necessary because an answer can
# span two sections (e.g. Trash recovery + Storage Full quota) that don't both
# rank in the top 3 under a small embedding model.
TOP_K = 5
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
SYNONYMS = {"cost": "price", "costs": "price", "pricing": "price", "files": "file",
            "conflicting": "conflict", "conflicted": "conflict", "conflicts": "conflict",
            "merging": "merge", "merges": "merge"}

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

INDEX_PATH = Path(__file__).resolve().parent / "index.npz"


def docs_hash() -> str:
    """Fingerprint the corpus so a stale index can be detected at load time."""
    digest = hashlib.sha256()
    for doc in sorted(DOCS_DIR.glob("*.md")):
        digest.update(doc.read_bytes())
    return digest.hexdigest()

CHUNKS = load_chunks()

# Guarded so a missing model or index degrades to lexical retrieval instead of
# breaking import. The app must still answer with neither model nor API key.
try:
    from embed import MODEL_ID, embed
except Exception as exc:
    print(f"embed unavailable: {type(exc).__name__}")
    embed = None


def load_index():
    """Return (vectors, meta), or None if absent, unreadable, or stale.

    Rejects the index if either the docs or the embedding model changed since
    it was built — vectors built by one model are meaningless against queries
    embedded by another, and cosine scores would silently misroute.
    """
    try:
        data = np.load(INDEX_PATH, allow_pickle=False)
        if data["docs_hash"].item() != docs_hash():
            print("index is stale: re-run backend/build_index.py")
            return None
        if data["model_id"].item() != MODEL_ID:
            print("index built by a different model: re-run backend/build_index.py")
            return None
        return data["vectors"], json.loads(data["meta"].item())
    except Exception as exc:
        print(f"index unavailable: {type(exc).__name__}")
        return None


INDEX = load_index() if embed is not None else None

class State(TypedDict):
    messages: Annotated[list[dict], add]
    category: str
    context: list[dict]
    confidence: float
    clarified: bool

def router(state: State) -> dict:
    words = set(tokenize(state["messages"][-1]["content"]))
    billing, technical = len(words & BILLING_WORDS), len(words & TECH_WORDS)
    if billing == technical == 0:
        return {"category": "general"}
    return {"category": "billing" if billing >= technical else "technical"}

def lexical_retrieve(question: str) -> dict:
    """Token-overlap fallback used when the model or index is unavailable.

    Its confidence is a matched-terms ratio, not a cosine similarity, so
    CONFIDENCE_THRESHOLD is only approximately meaningful on this path. That is
    acceptable for a degraded mode; it is not a scale worth calibrating twice.
    """
    terms = tokenize(question)
    scored = []
    for chunk in CHUNKS:
        chunk_terms = tokenize(chunk["text"])
        matched = {t for t in terms if t in chunk_terms}
        scored.append((sum(chunk_terms.count(t) for t in matched), len(matched), chunk))
    scored.sort(key=lambda s: (-s[0], s[2]["doc"]))
    top = [c for score, _, c in scored[:TOP_K] if score > 0]
    confidence = scored[0][1] / len(terms) if terms and scored else 0.0
    return {"context": top, "confidence": confidence}


def retriever(state: State) -> dict:
    question = state["messages"][-1]["content"]
    if embed is None or INDEX is None:
        return lexical_retrieve(question)
    try:
        vectors, meta = INDEX
        # rows and the query are both L2-normalized, so this dot product is cosine
        scores = vectors @ embed([question])[0]
        top = np.argsort(-scores)[:TOP_K]
        return {
            "context": [meta[i] for i in top],
            "confidence": float(scores[top[0]]),
        }
    except Exception as exc:
        print(f"semantic retrieval failed: {type(exc).__name__}")
        return lexical_retrieve(question)

SUPPORT_SYSTEM_PROMPT = """You are the official AI support assistant for CloudNest.

Provide fast, accurate, friendly, professional support. Every response should feel \
like it comes from an experienced CloudNest support engineer, not an AI model or a \
search system.

# Core principles
- Prioritize accuracy over guessing.
- Answer the question directly before adding context.
- Solve the problem, don't just answer it: explain why it happens, how to fix it, \
what happens next, and the logical next step.
- Keep it concise unless more detail is asked for.
- Sound natural, conversational, and confident.

# Human voice
- Write like a real person chatting, not a search engine. Use contractions \
(you're, it's, don't, you'll).
- Vary your openings. Good: "Yes." "You can do that." "Here's how." "That usually \
happens when..." "The easiest way is..." Avoid: "Certainly!" "According to..." \
"Based on..." "I'd be happy to help." "I can confirm..."
- Don't be overly formal or scripted. Don't overuse emoji.

# Never reveal internal implementation
Never mention or imply documentation, docs, a knowledge base, context, retrieved \
information, sources, search results, confidence, routing, or AI limitations. Never \
say "According to the documentation", "Based on the context", "The docs don't \
mention", "I only have access to", or anything like them.

# When information exists
Give the answer immediately and confidently; don't explain how you know it. Where \
useful, add why it happens, what happens next, common mistakes, and tips.

# When information can't be confirmed
Don't expose limitations. Say it naturally instead: "That isn't currently specified." \
"That hasn't been officially confirmed." "At the moment I can't confirm that." "For a \
definitive answer, our support team can help." Never say "the documentation doesn't \
say", "my context doesn't contain", or "I couldn't retrieve".

# Multi-step reasoning
Many questions need several pieces of product information combined. Merge them into \
one complete answer. Never mention that multiple sources were used.

# Hallucination prevention
Never invent features, pricing, storage limits, APIs, security policies, release \
dates, roadmap items, or integrations. If something truly isn't specified, say "That \
isn't currently specified" rather than guessing.

# Personalization
Tailor recommendations when you have enough to go on, e.g. "Since you're on six \
devices and need API access, the Team plan is the best fit."

# Clarify only when necessary
If the question is genuinely ambiguous, ask one short follow-up (e.g. "Are you on the \
desktop app or the web app?") rather than assuming.

# Be proactive
Where it helps, offer the likely next thing: "Want help upgrading?" "I can walk you \
through the API setup." "If that doesn't fix it, I can dig further."

# Formatting
Simple questions: under 100 words. Instructions: numbered steps. Comparisons: bullets \
or a table. Long answers: short sections. Avoid walls of text. Use Markdown.

# Grounding (internal — never mention this to the user)
Answer only from the CloudNest product information provided in the conversation. If it \
doesn't cover something, treat it as not specified and use the natural "isn't \
currently specified" phrasing above. Never quote or reference that information as a \
source."""

def llm_answer(question: str, context: str, history: list[dict]) -> str | None:
    try:
        import anthropic
        client = anthropic.Anthropic()
        prompt = f"CloudNest product reference:\n{context}\n\nCustomer question: {question}"
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1024,
            system=SUPPORT_SYSTEM_PROMPT,
            messages=history[:-1] + [{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text") or None
    except Exception as e:
        # log only the exception type: messages/reprs can embed secrets (e.g. API keys)
        print(f"llm_answer failed: {type(e).__name__}")
        return None

def sources_from_context(context: list[dict]) -> list[str]:
    """Unique section titles behind an answer, in retrieved order, for citation."""
    seen, out = set(), []
    for c in context:
        if c["title"] not in seen:
            seen.add(c["title"])
            out.append(c["title"])
    return out

def responder(state: State) -> dict:
    question = state["messages"][-1]["content"]
    context = "\n\n".join(f"[{c['doc']} - {c['title']}]\n{c['text']}" for c in state["context"])
    answer = llm_answer(question, context, state["messages"])
    if answer is None:
        # no key / API down: hand back the retrieved sections as clean Markdown
        parts = [f"**{c['title']}**\n\n{c['text']}" for c in state["context"]]
        answer = "Here's what should help:\n\n" + "\n\n---\n\n".join(parts)
    return {"messages": [{"role": "assistant", "content": answer}], "clarified": False}

def clarify(_state: State) -> dict:
    msg = ("I want to make sure I get this right — could you tell me a bit more? Your "
           "plan, the device you're on, or the exact message you're seeing all help. Our "
           "support team can also confirm the specifics if you'd rather go straight there.")
    return {"messages": [{"role": "assistant", "content": msg}], "clarified": True}

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
