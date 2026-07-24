# Local ONNX Embedding Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the lexical token-count retriever in `backend/app.py` with semantic retrieval driven by a local ONNX embedding model, so the bot answers questions whose wording doesn't overlap the docs.

**Architecture:** All 46 doc chunks are embedded ahead of time by `build_index.py` and committed as `backend/index.npz`. At request time only the query is embedded; retrieval is a dot product against the committed matrix. The model (`all-MiniLM-L6-v2`, int8 ONNX) runs locally via ONNX Runtime, so there is no embedding provider, no API key, and no per-query cost. The existing lexical scorer is retained as a fallback for when the model or index is unavailable.

**Tech Stack:** Python 3.11, ONNX Runtime 1.27, HuggingFace `tokenizers` 0.23, NumPy, LangGraph, FastAPI, pytest, React/Vite.

## Global Constraints

- **Deploy size ceiling: 250 MB unzipped** (Vercel Python function limit). Projected ~145 MB. Verified in Task 5.
- **Use the int8 model** `model_quint8_avx2.onnx` (23 MB). The fp32 export is 90 MB and pushes the total to ~210 MB.
- **Never install `sentence-transformers` or `torch`** — torch alone is ~800 MB and blows the ceiling. ONNX Runtime replaces it.
- **Embeddings are always L2-normalized at creation**, so cosine similarity is a plain dot product everywhere downstream.
- **`np.load(..., allow_pickle=False)` always.** Loading pickled arrays executes arbitrary code; `index.npz` ships in the deploy bundle.
- **Exception logging: type name only, never the message or repr** — matches the existing convention at `backend/app.py:99-102`, which exists because exception reprs can embed API keys.
- **Commit messages: subject and body only.** No `Co-Authored-By` or `Claude-Session` trailers.
- **Work directly on `main`.** Do not create feature branches.
- Run all commands from the repo root: `C:\Users\HP\OneDrive\Documents\!Github projects\Alliedworks`.

## Verified Facts

Confirmed by direct inspection on 2026-07-24 — do not re-derive:

| Fact | Value |
|---|---|
| ONNX inputs | `input_ids`, `attention_mask`, `token_type_ids` — all `tensor(int64)` |
| ONNX output | `last_hidden_state`, shape `(batch, sequence, 384)` |
| Model download base | `https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/` |
| Already installed | `numpy`, `pytest`, `langgraph`, `anthropic`, `fastapi`, `onnxruntime` 1.27.0, `tokenizers` 0.23.1 |
| Corpus | 7 markdown files, 46 chunks after heading split |

Reference similarity, measured with the pooling in Task 1 (sanity target, not a test assertion):
`"how much am I charged mid cycle"` scores **0.347** against `"pricing and billing plans"` and **0.026** against `"my files will not sync"` — despite zero shared tokens with either.

## File Structure

| File | Responsibility |
|---|---|
| `backend/model/model_quint8_avx2.onnx` | New, committed. The int8 embedding model. |
| `backend/model/tokenizer.json` | New, committed. WordPiece vocab + config. |
| `backend/embed.py` | New. Sole owner of tokenization, ONNX inference, pooling, normalization. Exports `embed()`. |
| `backend/build_index.py` | New. Offline indexer + threshold calibration CLI. |
| `backend/index.npz` | New, committed, generated. Vectors + chunk metadata + staleness hash. |
| `backend/app.py` | Modified. `docs_hash()` added; `retriever()` rewritten; old scorer preserved as `lexical_retrieve()`; `CATEGORY_DOCS` and the 2x boost deleted; `State` gains `clarified`. |
| `backend/index.py` | Modified. `/api/chat` returns `clarified`; `/api/health` reports index status. |
| `backend/requirements.txt` | Modified. Adds `onnxruntime`, `tokenizers`, `numpy`. |
| `backend/requirements-dev.txt` | New. `pytest`. |
| `backend/tests/conftest.py` | New. Puts `backend/` on `sys.path`. |
| `backend/tests/test_embed.py` | New. |
| `backend/tests/test_build_index.py` | New. |
| `backend/tests/test_retrieval.py` | New. |
| `frontend/src/App.jsx` | Modified. Line 98's hardcoded `0.25` replaced by the server's `clarified` flag. |

---

### Task 1: Embedding function

Vendors the model and builds the single embedding primitive everything else calls.

**Files:**
- Create: `backend/model/model_quint8_avx2.onnx`, `backend/model/tokenizer.json`
- Create: `backend/embed.py`
- Create: `backend/tests/conftest.py`, `backend/tests/test_embed.py`
- Create: `backend/requirements-dev.txt`
- Modify: `backend/requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: `embed(texts: list[str]) -> np.ndarray` of shape `(len(texts), 384)`, float32, L2-normalized rows. Also module constants `MODEL_ID: str`, `DIM: int = 384`, `MAX_TOKENS: int = 256`.

- [ ] **Step 1: Download the model artifacts**

```bash
mkdir -p backend/model
python -c "
import urllib.request
B = 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/'
for src, dst in [('onnx/model_quint8_avx2.onnx', 'backend/model/model_quint8_avx2.onnx'),
                 ('tokenizer.json', 'backend/model/tokenizer.json')]:
    urllib.request.urlretrieve(B + src, dst)
    print(dst)
"
```

Verify sizes — the model must be 23.0 MB, not 90 MB (that would be the fp32 export):

```bash
ls -lh backend/model/
```

- [ ] **Step 2: Add dependencies**

Append to `backend/requirements.txt`:

```
onnxruntime
tokenizers
numpy
```

Create `backend/requirements-dev.txt`:

```
-r requirements.txt
pytest
```

- [ ] **Step 3: Create the test path shim**

Create `backend/tests/conftest.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 4: Write the failing tests**

Create `backend/tests/test_embed.py`:

```python
import numpy as np

from embed import DIM, embed


def test_rows_are_unit_norm():
    vectors = embed(["pricing and billing", "syncing files"])
    assert vectors.shape == (2, DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_empty_input_returns_empty_matrix():
    assert embed([]).shape == (0, DIM)


def test_paraphrase_scores_above_unrelated():
    charged, pricing, sync = embed([
        "how much am I charged mid cycle",
        "pricing and billing plans",
        "my files will not sync",
    ])
    assert charged @ pricing > charged @ sync
    # zero shared tokens with either; this is the case the lexical scorer misses
    assert charged @ pricing > 0.2


def test_pooling_ignores_padding():
    """A text embedded alone must equal the same text embedded in a padded batch.

    Regression guard for unmasked mean pooling. Without the attention mask,
    padding tokens are averaged into the vector and a text's embedding silently
    changes depending on what else is in the batch.
    """
    alone = embed(["refund policy"])[0]
    padded = embed(["refund policy", "a considerably longer sentence " * 12])[0]
    assert np.allclose(alone, padded, atol=1e-4)
```

- [ ] **Step 5: Run the tests to verify they fail**

```bash
python -m pytest backend/tests/test_embed.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'embed'`.

- [ ] **Step 6: Implement `embed.py`**

Create `backend/embed.py`:

```python
"""Local sentence embeddings via ONNX Runtime.

No API key and no torch: the int8 all-MiniLM-L6-v2 export plus onnxruntime is
~42 MB, where sentence-transformers would drag in ~800 MB of torch and blow
Vercel's 250 MB function limit.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_ID = "model_quint8_avx2.onnx"
MAX_TOKENS = 256
DIM = 384

# Built once per process, not per call: loading the session costs ~1-3s and a
# warm serverless instance should pay it only on cold start.
_session = ort.InferenceSession(
    str(MODEL_DIR / MODEL_ID), providers=["CPUExecutionProvider"]
)
_input_names = {i.name for i in _session.get_inputs()}
_tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
_tokenizer.enable_truncation(MAX_TOKENS)
_tokenizer.enable_padding()


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts. Returns (len(texts), DIM) float32 with L2-normalized rows.

    Rows are normalized here so every caller can treat a dot product as cosine
    similarity.
    """
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    encoded = _tokenizer.encode_batch(texts)
    ids = np.array([e.ids for e in encoded], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in _input_names:
        feed["token_type_ids"] = np.zeros_like(ids)
    tokens = _session.run(None, feed)[0]  # (n, sequence, DIM)
    # Mask-weighted mean pooling: the export emits per-token vectors, not a
    # sentence vector. Averaging padding in here degrades retrieval silently.
    weights = mask[..., None].astype(np.float32)
    pooled = (tokens * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
    return (pooled / np.linalg.norm(pooled, axis=1, keepdims=True)).astype(np.float32)
```

- [ ] **Step 7: Run the tests to verify they pass**

```bash
python -m pytest backend/tests/test_embed.py -v
```

Expected: 4 passed.

- [ ] **Step 8: Commit**

```bash
git add backend/model backend/embed.py backend/tests backend/requirements.txt backend/requirements-dev.txt
git commit -m "Add local ONNX embedding function

Vendors the int8 all-MiniLM-L6-v2 export and wraps it in embed(), which
tokenizes, runs inference, applies mask-weighted mean pooling, and
L2-normalizes so callers can treat dot products as cosine similarity.

Uses onnxruntime rather than sentence-transformers: the latter depends on
torch, which alone exceeds Vercel's 250 MB function limit."
```

---

### Task 2: Offline indexer

Embeds the corpus once and writes the committed index.

**Files:**
- Modify: `backend/app.py` (add `docs_hash()` only — the retriever rewrite is Task 3)
- Create: `backend/build_index.py`
- Create: `backend/index.npz` (generated)
- Create: `backend/tests/test_build_index.py`

**Interfaces:**
- Consumes: `embed()`, `MODEL_ID` from Task 1; `load_chunks()` and `DOCS_DIR` from `app.py`.
- Produces: `docs_hash() -> str` and `INDEX_PATH: Path` in `app.py`. `build(path: Path = INDEX_PATH) -> int` in `build_index.py`, returning the chunk count.

`docs_hash()` and `INDEX_PATH` live in `app.py`, not `build_index.py`, to avoid a circular import: `build_index` imports `load_chunks` from `app`, so `app` must not import from `build_index`. Task 3 consumes `INDEX_PATH` from the same place — it is defined once, here.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_build_index.py`:

```python
import json

import numpy as np

from app import docs_hash, load_chunks
from build_index import build


def test_docs_hash_is_stable_and_content_sensitive(tmp_path, monkeypatch):
    import app

    first = docs_hash()
    assert first == docs_hash()  # deterministic across calls

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# Title\noriginal", encoding="utf-8")
    monkeypatch.setattr(app, "DOCS_DIR", docs)
    before = docs_hash()
    (docs / "a.md").write_text("# Title\nedited", encoding="utf-8")
    assert docs_hash() != before


def test_build_writes_aligned_index(tmp_path):
    from embed import DIM, MODEL_ID

    path = tmp_path / "index.npz"
    count = build(path)
    assert count == len(load_chunks())

    data = np.load(path, allow_pickle=False)
    meta = json.loads(data["meta"].item())
    assert data["vectors"].shape == (count, DIM)
    assert data["vectors"].dtype == np.float32
    assert len(meta) == count
    assert {"doc", "title", "text"} <= meta[0].keys()
    assert data["docs_hash"].item() == docs_hash()
    assert data["model_id"].item() == MODEL_ID


def test_index_rows_align_with_meta(tmp_path):
    """Row i of vectors must be the embedding of meta[i]['text']."""
    from embed import embed

    path = tmp_path / "index.npz"
    build(path)
    data = np.load(path, allow_pickle=False)
    meta = json.loads(data["meta"].item())
    assert np.allclose(embed([meta[3]["text"]])[0], data["vectors"][3], atol=1e-5)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest backend/tests/test_build_index.py -v
```

Expected: collection error, `ImportError: cannot import name 'docs_hash' from 'app'`.

- [ ] **Step 3: Add `docs_hash()` and `INDEX_PATH` to `app.py`**

Add `import hashlib` to the imports at the top of `backend/app.py`, then insert this immediately after `load_chunks()` (currently ends at line 54) and before the `CHUNKS = load_chunks()` line:

```python
INDEX_PATH = Path(__file__).resolve().parent / "index.npz"


def docs_hash() -> str:
    """Fingerprint the corpus so a stale index can be detected at load time."""
    digest = hashlib.sha256()
    for doc in sorted(DOCS_DIR.glob("*.md")):
        digest.update(doc.read_bytes())
    return digest.hexdigest()
```

- [ ] **Step 4: Implement `build_index.py`**

Create `backend/build_index.py`:

```python
"""Build the embedding index. Re-run after editing cloudnest_docs/.

    python backend/build_index.py
    python backend/build_index.py --calibrate
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import INDEX_PATH, docs_hash, load_chunks
from embed import MODEL_ID, embed


def build(path: Path = INDEX_PATH) -> int:
    """Embed every chunk and write the index. Returns the chunk count."""
    chunks = load_chunks()
    vectors = embed([c["text"] for c in chunks])
    np.savez(
        path,
        vectors=vectors,
        meta=json.dumps(chunks),
        docs_hash=docs_hash(),
        model_id=MODEL_ID,
    )
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true",
                        help="report the similarity gap and suggest a threshold")
    args = parser.parse_args()
    count = build()
    print(f"indexed {count} chunks -> {INDEX_PATH}")
    if args.calibrate:
        from calibrate import report  # added in Task 4
        report()


if __name__ == "__main__":
    main()
```

Note: the `--calibrate` import is deliberately deferred to call time so this task's tests pass before Task 4 exists.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python -m pytest backend/tests/test_build_index.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Generate and inspect the real index**

```bash
python backend/build_index.py
```

Expected: `indexed 46 chunks -> ...backend\index.npz`

```bash
ls -lh backend/index.npz
```

Expected: roughly 80-150 KB. If the chunk count is not 46, the corpus changed since planning — that is fine, but note the new number.

- [ ] **Step 7: Commit**

```bash
git add backend/build_index.py backend/index.npz backend/app.py backend/tests/test_build_index.py
git commit -m "Add offline embedding indexer

build_index.py embeds all doc chunks via load_chunks() so build-time and
runtime chunking cannot drift, and writes vectors, row-aligned chunk
metadata, and a corpus hash to index.npz.

Chunk metadata is stored as a JSON string rather than an object array so
the index can be loaded with allow_pickle=False."
```

---

### Task 3: Semantic retriever with lexical fallback

**Files:**
- Modify: `backend/app.py` (lines 16-20 constants, lines 71-83 `retriever`)
- Create: `backend/tests/test_retrieval.py`

**Interfaces:**
- Consumes: `embed()` from Task 1; `INDEX_PATH` from Task 2; `docs_hash()` from Task 2.
- Produces: `lexical_retrieve(question: str) -> dict` and a rewritten `retriever(state) -> dict`, both returning `{"context": list[dict], "confidence": float}`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_retrieval.py`:

```python
import numpy as np
import pytest

import app


def test_semantic_retrieval_finds_billing_docs_without_shared_tokens():
    """The motivating failure: no token overlap with the pricing docs."""
    state = {
        "messages": [{"role": "user", "content": "how much am I charged mid cycle"}],
        "category": "billing",
    }
    result = app.retriever(state)
    assert result["context"], "expected at least one chunk"
    docs = {c["doc"] for c in result["context"]}
    assert "02_pricing_billing.md" in docs
    assert result["confidence"] > 0.0


def test_out_of_scope_question_scores_below_in_scope():
    def confidence(question):
        return app.retriever({
            "messages": [{"role": "user", "content": question}],
            "category": "general",
        })["confidence"]

    assert confidence("how do I get a refund") > confidence(
        "what is the capital of Peru"
    )


def test_falls_back_to_lexical_when_index_missing(monkeypatch):
    monkeypatch.setattr(app, "INDEX", None)
    result = app.retriever({
        "messages": [{"role": "user", "content": "refund"}],
        "category": "billing",
    })
    assert result["context"]


def test_falls_back_to_lexical_when_embed_unavailable(monkeypatch):
    monkeypatch.setattr(app, "embed", None)
    result = app.retriever({
        "messages": [{"role": "user", "content": "refund"}],
        "category": "billing",
    })
    assert result["context"]


def test_stale_index_is_rejected(monkeypatch, tmp_path):
    """A docs_hash mismatch must not be searched."""
    monkeypatch.setattr(app, "docs_hash", lambda: "0" * 64)
    assert app.load_index() is None


def test_lexical_retrieve_still_works():
    result = app.lexical_retrieve("refund policy")
    assert result["context"]
    assert 0.0 <= result["confidence"] <= 1.0


def test_category_boost_is_gone():
    assert not hasattr(app, "CATEGORY_DOCS")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest backend/tests/test_retrieval.py -v
```

Expected: failures including `AttributeError: module 'app' has no attribute 'INDEX'` and `test_category_boost_is_gone` failing because `CATEGORY_DOCS` still exists.

- [ ] **Step 3: Delete `CATEGORY_DOCS`**

In `backend/app.py`, delete these four lines (currently 17-20):

```python
CATEGORY_DOCS = {
    "billing": {"02_pricing_billing.md", "03_account_management.md"},
    "technical": {"04_technical_setup.md", "05_troubleshooting.md"},
}
```

Semantic ranking should not be overridden by a keyword-derived category — a 2x multiplier on a misclassified query actively promotes worse matches. `router()`, `BILLING_WORDS`, and `TECH_WORDS` stay: the category is still rendered as a route badge at `frontend/src/App.jsx:49`.

- [ ] **Step 4: Add the guarded imports and index loader**

In `backend/app.py`, add `import json` to the imports, and add this immediately after the `CHUNKS = load_chunks()` line:

```python
# Guarded so a missing model or index degrades to lexical retrieval instead of
# breaking import. The app must still answer with neither model nor API key.
try:
    from embed import embed
except Exception as exc:
    print(f"embed unavailable: {type(exc).__name__}")
    embed = None


def load_index():
    """Return (vectors, meta), or None if absent, unreadable, or stale."""
    try:
        data = np.load(INDEX_PATH, allow_pickle=False)
        if data["docs_hash"].item() != docs_hash():
            print("index is stale: re-run backend/build_index.py")
            return None
        return data["vectors"], json.loads(data["meta"].item())
    except Exception as exc:
        print(f"index unavailable: {type(exc).__name__}")
        return None


INDEX = load_index() if embed is not None else None
```

Add `import numpy as np` to the imports at the top of the file.

- [ ] **Step 5: Replace `retriever()`**

Replace the whole `retriever` function (currently lines 71-83) with:

```python
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
    top = [c for score, _, c in scored[:3] if score > 0]
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
        top = np.argsort(-scores)[:3]
        return {
            "context": [meta[i] for i in top],
            "confidence": float(scores[top[0]]),
        }
    except Exception as exc:
        print(f"semantic retrieval failed: {type(exc).__name__}")
        return lexical_retrieve(question)
```

- [ ] **Step 6: Run the full test suite**

```bash
python -m pytest backend/tests -v
```

Expected: all tests pass (4 + 3 + 7 = 14).

- [ ] **Step 7: Smoke-test the CLI end to end**

```bash
echo "how much am I charged if I add a teammate mid cycle" | python backend/app.py
```

Expected: a `[route: billing | confidence: 0.NN]` line where confidence is non-zero, followed by an answer drawn from the pricing docs. Record the confidence value — Task 4 needs a feel for the in-scope range.

- [ ] **Step 8: Commit**

```bash
git add backend/app.py backend/tests/test_retrieval.py
git commit -m "Replace lexical retrieval with embedding similarity

retriever() now embeds the question and ranks chunks by cosine similarity
against the committed index, so queries are matched on meaning rather than
shared tokens.

The old scorer is kept as lexical_retrieve() and runs when the model or
index is unavailable, preserving the offline path. Drops CATEGORY_DOCS and
its 2x score boost: with semantic ranking, boosting on a keyword-derived
category promotes worse matches when the category is wrong."
```

---

### Task 4: Calibrate the threshold and remove the duplicated cutoff

`CONFIDENCE_THRESHOLD` is currently `0.25` on a matched-terms ratio, and `frontend/src/App.jsx:98` hardcodes the same `0.25` with the comment "mirrors the graph's conditional edge". Cosine similarities do not live on that scale, so the value must be re-derived — and the duplication removed rather than resynced.

> **Deviation from the spec, flagged for review.** The spec said to expose the threshold via `/api/health` and have the frontend read it. This task instead has `/api/chat` return which node answered (`clarified`), so the frontend needs no threshold at all. Same goal — no duplicated literal — but it deletes the coupling instead of synchronizing it, and removes a startup race where a message sent before `/api/health` resolves would have no threshold to compare against.

**Files:**
- Create: `backend/calibrate.py`
- Modify: `backend/app.py` (`CONFIDENCE_THRESHOLD`, `State`, `responder`, `clarify`)
- Modify: `backend/index.py`
- Modify: `frontend/src/App.jsx:98`

**Interfaces:**
- Consumes: `embed()`, `INDEX_PATH`, `load_index()`.
- Produces: `report() -> None` in `calibrate.py`. `State["clarified"]: bool`. `/api/chat` response gains `"clarified": bool`.

- [ ] **Step 1: Write the calibration script**

Create `backend/calibrate.py`:

```python
"""Report the in-scope/out-of-scope similarity gap to pick CONFIDENCE_THRESHOLD.

Cosine similarity has no natural cutoff, and it is not on the same scale as the
matched-terms ratio the lexical scorer produced. Measure, do not guess.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import load_index
from embed import embed

IN_SCOPE = [
    "how much does the pro plan cost",
    "how do I get a refund",
    "can I change my billing cycle",
    "why won't my files sync",
    "the desktop app keeps crashing on startup",
    "how do I restore a deleted file",
    "is my data encrypted at rest",
    "how do I reset my password",
    "how do I invite a teammate to my account",
    "what happens when I run out of storage",
]

OUT_OF_SCOPE = [
    "what is the capital of Peru",
    "write me a poem about the sea",
    "how do I file my taxes",
    "what is the weather tomorrow",
    "who won the world cup in 1998",
    "how do I change my car's oil",
]


def report() -> None:
    index = load_index()
    if index is None:
        print("no usable index; run: python backend/build_index.py")
        return
    vectors, _ = index

    def top1(questions):
        return (embed(questions) @ vectors.T).max(axis=1)

    hits, misses = top1(IN_SCOPE), top1(OUT_OF_SCOPE)
    for label, scores, questions in [
        ("IN SCOPE", hits, IN_SCOPE),
        ("OUT OF SCOPE", misses, OUT_OF_SCOPE),
    ]:
        print(f"\n{label}  min={scores.min():.3f}  max={scores.max():.3f}  mean={scores.mean():.3f}")
        for score, question in sorted(zip(scores, questions)):
            print(f"  {score:.3f}  {question}")

    floor, ceiling = float(hits.min()), float(misses.max())
    print(f"\nlowest in-scope   {floor:.3f}")
    print(f"highest out-scope {ceiling:.3f}")
    if floor > ceiling:
        print(f"clean separation; suggested CONFIDENCE_THRESHOLD = {(floor + ceiling) / 2:.2f}")
    else:
        print("OVERLAP: no threshold separates these sets cleanly.")
        print("Pick by which error you prefer — a low value answers more and")
        print("risks confident wrong answers; a high value asks to rephrase more.")


if __name__ == "__main__":
    report()
```

- [ ] **Step 2: Run calibration and choose the threshold**

```bash
python backend/calibrate.py
```

Read the output. If it reports clean separation, use the suggested midpoint. If it reports overlap, pick a value above `highest out-scope` and accept that the lowest-scoring in-scope questions will route to `clarify`.

**Write the chosen number down — the next step needs it.** Do not carry `0.25` forward; it is on the wrong scale.

- [ ] **Step 3: Set the threshold**

In `backend/app.py`, replace line 16:

```python
CONFIDENCE_THRESHOLD = 0.25
```

with the calibrated value, e.g.:

```python
# cosine similarity, chosen from backend/calibrate.py output
CONFIDENCE_THRESHOLD = 0.18
```

- [ ] **Step 4: Have the graph report which branch it took**

In `backend/app.py`, add `clarified` to the `State` TypedDict (currently lines 58-62):

```python
class State(TypedDict):
    messages: Annotated[list[dict], add]
    category: str
    context: list[dict]
    confidence: float
    clarified: bool
```

In `responder`, change the return to:

```python
    return {"messages": [{"role": "assistant", "content": answer}], "clarified": False}
```

In `clarify`, change the return to:

```python
    return {"messages": [{"role": "assistant", "content": msg}], "clarified": True}
```

Exactly one of the two nodes runs on every invocation, so `clarified` is always set by the time the graph returns.

- [ ] **Step 5: Return it from the API**

In `backend/index.py`, update the `chat` handler's return (currently lines 26-28):

```python
    return {"answer": result["messages"][-1]["content"],
            "category": result["category"],
            "confidence": result["confidence"],
            "clarified": result["clarified"]}
```

And extend `health` (currently lines 31-33) so index state is visible without reading logs:

```python
@app.get("/api/health")
def health():
    from app import INDEX
    return {"mode": "claude" if os.environ.get("ANTHROPIC_API_KEY") else "extractive",
            "retrieval": "semantic" if INDEX is not None else "lexical"}
```

- [ ] **Step 6: Use the flag in the frontend**

In `frontend/src/App.jsx`, replace line 98:

```javascript
      const isClarify = data.confidence < 0.25; // mirrors the graph's conditional edge
```

with:

```javascript
      const isClarify = data.clarified; // the graph reports which branch it took
```

The threshold now lives in exactly one place.

- [ ] **Step 7: Add a routing test**

Append to `backend/tests/test_retrieval.py`:

```python
def test_graph_reports_which_branch_answered():
    graph = app.build_app()
    config = {"configurable": {"thread_id": "test-routing"}}
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "what is the capital of Peru"}]},
        config,
    )
    assert result["clarified"] is True

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "how do I get a refund"}]},
        {"configurable": {"thread_id": "test-routing-2"}},
    )
    assert result["clarified"] is False
```

- [ ] **Step 8: Run the full suite**

```bash
python -m pytest backend/tests -v
```

Expected: 15 passed. If `test_graph_reports_which_branch_answered` fails, the threshold from Step 2 is misplaced relative to these two questions — re-run calibration and reconsider, do not weaken the test.

- [ ] **Step 9: Verify the frontend end to end**

```bash
npm --prefix frontend install && npm --prefix frontend run dev
```

In a second terminal:

```bash
python -m uvicorn index:app --app-dir backend --port 8000
```

Ask an in-scope question and an out-of-scope one. The out-of-scope answer must show the `clarify-request` badge; the in-scope answer must show `billing` or `technical`. Stop both servers when done.

- [ ] **Step 10: Commit**

```bash
git add backend/calibrate.py backend/app.py backend/index.py backend/tests/test_retrieval.py frontend/src/App.jsx
git commit -m "Calibrate confidence threshold and remove duplicated cutoff

The old 0.25 threshold was a matched-terms ratio and is meaningless against
cosine similarity, so calibrate.py measures the in-scope/out-of-scope
similarity gap and the threshold is set from that.

The frontend previously hardcoded the same 0.25 to mirror the graph's
conditional edge. The graph now reports which branch answered via a
clarified flag on the API response, so the threshold exists in one place."
```

---

### Task 5: Verify the deploy fits

Every size figure so far is projected from wheel sizes. The 250 MB ceiling is a hard failure, not a degradation, so it gets measured.

**Files:** none modified unless the check fails.

- [ ] **Step 1: Measure the installed footprint**

Build a throwaway venv in the repo root, measure it with `backend/`, then delete it.

```bash
python -m venv .sizecheck
./.sizecheck/Scripts/pip install -q -r backend/requirements.txt
python -c "
from pathlib import Path
site = Path('.sizecheck/Lib/site-packages')
if not site.is_dir():
    site = next(Path('.sizecheck/lib').glob('python*/site-packages'))  # posix layout
deps = sum(f.stat().st_size for f in site.rglob('*') if f.is_file())
assets = sum(f.stat().st_size for f in Path('backend').rglob('*') if f.is_file())
print(f'dependencies {deps/1e6:7.1f} MB')
print(f'backend/     {assets/1e6:7.1f} MB')
print(f'TOTAL        {(deps+assets)/1e6:7.1f} MB  (limit 250)')
"
rm -rf .sizecheck
```

Expected: total comfortably under 250 MB. If it exceeds ~200 MB, stop and report before deploying — options are trimming `onnxruntime` provider libraries or moving the model to a fetch-at-build step.

`.sizecheck/` is deleted in the same command and is never staged (every `git add` in this plan names explicit paths), so it needs no `.gitignore` entry.

- [ ] **Step 2: Deploy a preview and confirm**

```bash
vercel deploy
```

If the Vercel CLI is not installed, install it with `npm i -g vercel` first, or push and let the Git integration build.

Expected: build succeeds with no size error. Hit `/api/health` on the preview URL and confirm it reports `"retrieval": "semantic"` — if it reports `"lexical"`, the model or index did not ship and `includeFiles` needs checking.

- [ ] **Step 3: Confirm cold-start latency is tolerable**

Hit the preview `/api/chat` once after several idle minutes, then again immediately. The first request carries the ~1-3s ONNX session load; the second should not. If the cold hit is unacceptable, that is a product decision to raise, not a bug to fix here.

- [ ] **Step 4: Update the README**

Add a note under the setup instructions:

```markdown
### Rebuilding the search index

The retriever uses embeddings generated ahead of time. After editing anything
in `backend/cloudnest_docs/`, regenerate the index:

    python backend/build_index.py

Skipping this is safe but degrading: `app.py` detects the stale hash and falls
back to keyword retrieval rather than searching outdated vectors.
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "Document index rebuild step

Editing the docs without re-running build_index.py silently drops retrieval
to the keyword fallback, which is easy to miss from the outside."
```

---

## Verification Checklist

- [ ] `python -m pytest backend/tests -v` — 15 passed
- [ ] `python backend/build_index.py` regenerates the index without error
- [ ] Deleting `backend/index.npz` still yields answers (lexical fallback), and `/api/health` reports `"retrieval": "lexical"`
- [ ] Editing a doc without rebuilding logs `index is stale` and falls back
- [ ] Unsetting `ANTHROPIC_API_KEY` still yields extractive answers
- [ ] Deployed function is under 250 MB and `/api/health` reports `"retrieval": "semantic"`
- [ ] `grep -rn "0\.25" frontend/src backend` returns nothing related to the threshold
- [ ] `grep -rn "CATEGORY_DOCS" backend` returns nothing
