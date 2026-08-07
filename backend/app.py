import hashlib
import json
import os
import re
from datetime import datetime, timezone
from operator import add
from pathlib import Path
from typing import Annotated, TypedDict
from uuid import uuid4
import numpy as np
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

_ROOT = Path(__file__).resolve().parent.parent
# .env.local first: `vercel env pull` writes there, and the more specific file
# should win (setdefault means whoever is read first sticks). A real shell
# variable still beats both.
ENV_FILES = (_ROOT / ".env.local", _ROOT / ".env")


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        # `vercel env pull` writes KEY="value"; the quotes are delimiters, not
        # part of the secret, and leaving them on quietly breaks a DSN.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


for _env_file in ENV_FILES:
    _load_env_file(_env_file)
DOCS_DIR = Path(__file__).resolve().parent.parent / "cloudnest_docs"
# cosine similarity, chosen from backend/calibrate.py output
CONFIDENCE_THRESHOLD = 0.30
ESCALATE_FLOOR = 0.27  # just above the measured out-of-scope ceiling (0.265);
                       # below this a question is off-topic noise and never tickets
                       # In fully-degraded (no-index) mode confidence is the
                       # lexical matched-terms ratio from lexical_retrieve, not
                       # a cosine similarity, so this floor - like
                       # CONFIDENCE_THRESHOLD - is only approximately
                       # meaningful on that path.
# Human-handoff detection. Exact-phrase matching kept missing natural variants
# ("connect to human" vs "connect me to a representative"), so this looks for the
# intent instead: a word meaning "a human" together with a word meaning "hand me
# over". Ordinary questions that merely contain "person"/"someone" but no handoff
# verb ("am I the only person seeing this") stay answerable.
HUMAN_NOUNS_STRONG = {"human", "agent", "representative", "rep", "operator", "advisor"}
HUMAN_NOUNS = HUMAN_NOUNS_STRONG | {"person", "someone", "somebody"}
HANDOFF_VERBS = {
    "connect", "talk", "speak", "transfer", "reach", "chat", "call",
    "get", "put", "want", "need",
}
CLARIFY_BORDERLINE_MSG = (
    "I want to make sure I get this right, so could you tell me a bit more? "
    "Your plan, the device you're on, or the exact message you're seeing all "
    "help. Our support team can also confirm the specifics if you'd rather go "
    "straight there."
)
CLARIFY_OFFTOPIC_MSG = (
    "I'd like to help with that, though I should be straight with you: "
    "CloudNest is really all I know about, so your plan and billing, your "
    "account, or getting syncing and setup working. If any of that is what "
    "you're after, tell me what's going on and I'll take it from there."
)
GREETING_MSG = (
    "Hey, good to meet you. I look after CloudNest support, so anything to do "
    "with your plan and billing, your account, or getting syncing and setup "
    "behaving is fair game. What can I help you with?"
)
DEFLECT_MSG = (
    "I can almost certainly sort this out for you right here, and it's usually "
    "quicker than waiting for someone to get back to you. Tell me what you're "
    "running into and I'll take care of it. If you'd still rather talk to a "
    "person after that, just say so and I'll connect you."
)
CONTEXT_CHAR_CAP = 200  # bound how much of the prior turn we fold in
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

from variant import index_filename

INDEX_PATH = Path(__file__).resolve().parent / index_filename()


def docs_hash() -> str:
    """Fingerprint the corpus so a stale index can be detected at load time."""
    digest = hashlib.sha256()
    for doc in sorted(DOCS_DIR.glob("*.md")):
        digest.update(doc.read_bytes())
    return digest.hexdigest()

CHUNKS = load_chunks()
# Sections sent to the LLM for grounding. Below SMALL_CORPUS_LIMIT we retrieve
# everything rather than slicing to a fixed cutoff: a query can legitimately
# need facts from many sections at once (e.g. "I need API access, EU
# residency, bank transfer, and CLI automation" - 4 sections, 2 of them ranked
# outside any small top-N), and at ~1,800 tokens for all 46 of today's CHUNKS,
# sending everything costs nothing meaningful. This does not affect
# CONFIDENCE_THRESHOLD routing (still the single top score) or citations
# (still capped separately, see sources_from_context in index.py) - only how
# much context an already-confident answer gets to draw from.
#
# Past SMALL_CORPUS_LIMIT, "send everything" stops being free, so TOP_K falls
# back to a fixed cutoff. FALLBACK_TOP_K is an unvalidated placeholder, not a
# tuned value - a fixed cutoff is the same shape of bug this file just fixed
# (see the regression test for the rank-10/rank-22 failure), so recalibrate it
# against real questions (backend/calibrate.py) before this branch ever runs,
# rather than trusting the number below.
SMALL_CORPUS_LIMIT = 150
FALLBACK_TOP_K = 15
TOP_K = len(CHUNKS) if len(CHUNKS) <= SMALL_CORPUS_LIMIT else FALLBACK_TOP_K
# The no-API-key fallback prints retrieved sections as Markdown; cap it so it
# never dumps the whole corpus. 5 covers the documented multi-fact worst case
# (a question spanning ~4 sections) while still bounding the offline answer. The
# LLM path (with a key) still receives all of context - this bounds presentation
# only.
EXTRACTIVE_SECTIONS = 5

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
    semantic: dict          # written by semantic_retriever (parallel branch)
    lexical: dict           # written by lexical_retriever (parallel branch)
    context: list[dict]     # fused, sent to the LLM and used for citations
    confidence: float
    lexical_rescue: bool
    clarified: bool
    escalate: bool          # set by the gate/escalate node (Task 4)
    ticket: dict | None     # built by escalate (Task 4); None otherwise

def contextualize_query(messages: list[dict]) -> str:
    """Fold the previous user turn into the retrieval query.

    A follow-up like "does it cost extra" carries no topic word of its own,
    so embedding it alone often misses or triggers a needless clarify. Only
    retrieval sees the folded text - the full conversation still goes to
    the LLM for the actual answer, so nothing changes about how Claude
    reads the conversation.

    Greeting-only turns are skipped, not folded. They carry no topic to
    inherit, and mixing one in actively hurts: "what are the plans" scores
    0.397 on its own but 0.256 as "hello my name is likith. what are the
    plans", which drops an in-scope question under the off-topic floor. So we
    keep looking back for a turn that actually had a subject.
    """
    current = messages[-1]["content"]
    for msg in reversed(messages[:-1]):
        if msg["role"] == "user" and msg["content"].strip():
            if _is_greeting(msg["content"]):
                continue
            prior = msg["content"].strip()[:CONTEXT_CHAR_CAP]
            return f"{prior}. {current}"
    return current

def _explicit_human_request(text: str) -> bool:
    """True when the user is asking to be handed to a human.

    Detects the intent rather than fixed phrases: "escalate", or a human-noun
    ("agent"/"person"/...) alongside a handoff verb ("connect"/"talk"/...), or
    with "real"/"live" ("real person"), or a strong human-noun standing nearly
    alone ("agent", "human please"). A human-noun with no handoff cue stays
    answerable, so "am I the only person seeing this" is not a request.
    """
    words = set(re.findall(r"[a-z]+", text.lower()))
    if "escalate" in words:
        return True
    human = words & HUMAN_NOUNS
    if human and (words & HANDOFF_VERBS or words & {"real", "live", "actual"}):
        return True
    if (words & HUMAN_NOUNS_STRONG) and len(words) <= 3:
        return True
    return False

def router(state: State) -> dict:
    words = set(tokenize(contextualize_query(state["messages"])))
    billing, technical = len(words & BILLING_WORDS), len(words & TECH_WORDS)
    if billing == technical == 0:
        return {"category": "general"}
    return {"category": "billing" if billing >= technical else "technical"}

def lexical_retrieve(question: str) -> dict:
    """Token-overlap scorer. Used both as a parallel retrieval branch and,
    when the embedding model or index is unavailable, as the sole retriever.

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
    ranking = [c for _, _, c in scored]
    top = [c for score, _, c in scored[:TOP_K] if score > 0]
    confidence = scored[0][1] / len(terms) if terms and scored else 0.0
    # A strong exact-keyword signal: the top chunk hit >= 2 distinct query
    # terms. Consumed by the gate to rescue a borderline-confidence answer
    # that the embedding underscored (e.g. an exact product/CLI name).
    rescue = bool(scored and scored[0][1] >= 2)
    return {"context": top, "confidence": confidence, "ranking": ranking, "rescue": rescue}


RRF_K = 60  # Reciprocal Rank Fusion damping; 60 is the standard default


def rrf_fuse(rankings: list[list[dict]]) -> list[dict]:
    """Merge several ranked chunk-lists into one by Reciprocal Rank Fusion.

    Each chunk scores 1 / (RRF_K + rank) for every list it appears in (rank
    0-based), so a chunk ranked highly in either list surfaces even if the
    other list buries it. Chunks are identified by (doc, title), so the same
    section coming from the semantic index and the lexical corpus counts once.
    """
    scores: dict[tuple, float] = {}
    first_seen: dict[tuple, dict] = {}
    order: list[tuple] = []
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            key = (chunk["doc"], chunk["title"])
            if key not in first_seen:
                first_seen[key] = chunk
                order.append(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
    order.sort(key=lambda k: -scores[k])
    return [first_seen[k] for k in order]


def semantic_retrieve(question: str) -> dict:
    """Cosine-rank the whole corpus against the question.

    Returns the full ranking and the top-1 cosine as confidence. On a missing
    or broken model/index, returns an empty ranking and 0.0 - the lexical
    branch then carries the turn and supplies confidence (see fuse_results).
    """
    if embed is None or INDEX is None:
        return {"ranking": [], "confidence": 0.0}
    try:
        vectors, meta = INDEX
        # rows and the query are both L2-normalized, so this dot product is cosine
        scores = vectors @ embed([question])[0]
        order = np.argsort(-scores)
        return {
            "ranking": [meta[i] for i in order],
            "confidence": float(scores[order[0]]),
        }
    except Exception as exc:
        print(f"semantic retrieval failed: {type(exc).__name__}")
        return {"ranking": [], "confidence": 0.0}


def fuse_results(semantic: dict, lexical: dict) -> dict:
    """Merge the semantic and lexical branches into the retrieval result.

    context: RRF of the two rankings, capped at TOP_K, sent to the LLM and
    used for citations. confidence: the semantic cosine top-1 when semantic
    ran (preserves the calibrated 0.30 gate), else the lexical ratio so the
    bot still answers in fully-degraded mode. lexical_rescue: passed through
    for the gate's borderline rescue rule.
    """
    context = rrf_fuse([semantic["ranking"], lexical["ranking"]])[:TOP_K]
    confidence = semantic["confidence"] if semantic["ranking"] else lexical["confidence"]
    return {"context": context, "confidence": confidence,
            "lexical_rescue": lexical["rescue"]}


def retriever(state: State) -> dict:
    """Non-graph convenience: run both branches and fuse, in one call.

    The compiled graph runs the branches as parallel nodes instead (see
    build_app), but this keeps a single synchronous entry point for tests and
    the CLI.
    """
    question = contextualize_query(state["messages"])
    return fuse_results(semantic_retrieve(question), lexical_retrieve(question))


def semantic_retriever(state: State) -> dict:
    return {"semantic": semantic_retrieve(contextualize_query(state["messages"]))}


def lexical_retriever(state: State) -> dict:
    return {"lexical": lexical_retrieve(contextualize_query(state["messages"]))}


def fusion(state: State) -> dict:
    return fuse_results(state["semantic"], state["lexical"])

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
- Don't be overly formal or scripted. Never use emoji, anywhere, for any reason.
- Never use an em dash (—). It's a giveaway of AI-generated text. Write two shorter \
sentences instead, or use a comma or parentheses.
- Never use a horizontal rule / divider line (---, ***, or similar) to separate \
sections. It reads as generated, not written. Use a short heading, a blank line, or \
just start the next paragraph instead.

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
        # no key / API down: hand back the retrieved sections as clean Markdown.
        # No horizontal rules between sections — a bold title is separation enough.
        parts = [f"**{c['title']}**\n\n{c['text']}" for c in state["context"][:EXTRACTIVE_SECTIONS]]
        answer = "Here's what should help:\n\n" + "\n\n".join(parts)
    return {"messages": [{"role": "assistant", "content": answer}],
            "clarified": False, "escalate": False, "ticket": None}

def picker_for_test(confidence: float, messages: list[dict],
                    lexical_rescue: bool) -> str:
    """The gate's routing decision, as a pure function so it is unit-testable.

    Order matters: an explicit human request wins over confidence, but the
    first one deflects (offer to help before handing off) and only a repeat
    request after that deflection escalates; a rescued borderline answer beats
    a clarify; a repeat borderline miss escalates; a first borderline miss
    clarifies; anything below the floor is off-topic and can only clarify - it
    never escalates, so noise cannot create tickets.
    """
    current = messages[-1]["content"]
    if _explicit_human_request(current):
        # Try to help before handing off: the first request deflects, and only
        # a request made after the user has already been deflected escalates.
        return "escalate" if _deflected_before(messages) else "deflect"
    if confidence >= CONFIDENCE_THRESHOLD:
        return "responder"
    if ESCALATE_FLOOR <= confidence < CONFIDENCE_THRESHOLD:
        if lexical_rescue:
            return "responder"
        if _prev_assistant_was_borderline_clarify(messages):
            return "escalate"
        return "clarify"
    return "clarify"  # below the floor: off-topic, clarify only, never escalate


def _prev_assistant_was_borderline_clarify(messages: list[dict]) -> bool:
    for msg in reversed(messages[:-1]):
        if msg["role"] == "assistant":
            return msg["content"] == CLARIFY_BORDERLINE_MSG
    return False


def _deflected_before(messages: list[dict]) -> bool:
    """True if the bot has already offered to help in place of a human handoff.

    Scans the whole history (not just the previous turn) so a user who was
    deflected, asked a question or two, and then still wants a person is
    recognized as insisting and gets escalated.
    """
    return any(m["role"] == "assistant" and m["content"] == DEFLECT_MSG
               for m in messages[:-1])


GREETING_WORDS = {
    "hi", "hii", "hiya", "hello", "helo", "hey", "heya", "yo", "howdy",
    "greetings", "morning", "afternoon", "evening", "namaste", "sup",
}
# "my name is ..." is an introduction, which is a greeting by another route
GREETING_OPENERS = ("my name is", "this is ", "i am ", "i'm ", "how are you",
                    "how's it going", "hows it going", "nice to meet")


def _is_greeting(text: str) -> bool:
    """True when the message is *only* a hello, with nothing to answer in it.

    A greeting that carries a real question ("hi, my sync is broken") must not
    match: saying hello back would drop the actual problem on the floor. So a
    single product word anywhere disqualifies it, as does any message long
    enough to be carrying content.
    """
    lowered = text.lower()
    words = re.findall(r"[a-z']+", lowered)
    if not words or len(words) > 8:
        return False
    if any(w in BILLING_WORDS or w in TECH_WORDS for w in words):
        return False
    return bool(GREETING_WORDS.intersection(words)) or any(
        opener in lowered for opener in GREETING_OPENERS)


def clarify(state: State) -> dict:
    if state["confidence"] < ESCALATE_FLOOR and _is_greeting(state["messages"][-1]["content"]):
        # Someone saying hello isn't off-topic noise, they just haven't got to
        # the point yet. Greet them and invite the question.
        return {"messages": [{"role": "assistant", "content": GREETING_MSG}],
                "clarified": True, "escalate": False, "ticket": None}
    msg = CLARIFY_OFFTOPIC_MSG if state["confidence"] < ESCALATE_FLOOR else CLARIFY_BORDERLINE_MSG
    return {"messages": [{"role": "assistant", "content": msg}],
            "clarified": True, "escalate": False, "ticket": None}


def deflect(state: State) -> dict:
    """First response to a human-handoff request: offer to help first.

    Not a ticket and not a refusal - it invites the user to state their issue
    so the bot can try. If they ask for a person again after this, the gate
    routes to escalate (see _deflected_before).
    """
    return {"messages": [{"role": "assistant", "content": DEFLECT_MSG}],
            "clarified": True, "escalate": False, "ticket": None}


def escalate(state: State) -> dict:
    reason = ("user_requested"
              if _explicit_human_request(state["messages"][-1]["content"])
              else "repeated_low_confidence")
    ticket = {
        "id": uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": state["messages"][-1]["content"],
        "category": state.get("category", "general"),
        "confidence": state.get("confidence", 0.0),
        "conversation": state["messages"],
        "reason": reason,
    }
    msg = ("I've passed this to our support team and someone will follow up "
           "with you directly. Is there anything else I can help with in the "
           "meantime?")
    return {"messages": [{"role": "assistant", "content": msg}],
            "clarified": False, "escalate": True, "ticket": ticket}

def build_app():
    graph = StateGraph(State)
    for fn in (router, semantic_retriever, lexical_retriever, fusion,
               responder, clarify, escalate, deflect):
        graph.add_node(fn.__name__, fn)
    graph.add_edge(START, "router")
    # fan out: both retrieval branches run in parallel off the router
    graph.add_edge("router", "semantic_retriever")
    graph.add_edge("router", "lexical_retriever")
    # fan in: fusion waits for both branches (it has an edge from each)
    graph.add_edge("semantic_retriever", "fusion")
    graph.add_edge("lexical_retriever", "fusion")

    def picker(s: State) -> str:
        return picker_for_test(s["confidence"], s["messages"], s["lexical_rescue"])
    graph.add_conditional_edges("fusion", picker,
                                ["responder", "clarify", "escalate", "deflect"])
    graph.add_edge("responder", END)
    graph.add_edge("clarify", END)
    graph.add_edge("escalate", END)
    graph.add_edge("deflect", END)
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
