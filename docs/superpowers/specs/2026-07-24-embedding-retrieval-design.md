# Replace lexical retrieval with local ONNX embeddings

**Date:** 2026-07-24
**Status:** Approved, not yet implemented

## Problem

`backend/app.py` retrieves documentation chunks by counting shared tokens between
the query and each chunk. Three hand-maintained tables prop this up: `SYNONYMS`
(9 entries), `BILLING_WORDS` (23), and `TECH_WORDS` (37). They only work for
words that were predicted in advance.

The failure is vocabulary mismatch. "How much will I be charged if I add a
teammate mid-cycle?" scores against the pricing chunk only if those exact tokens
appear in it. `cost -> price` is handled because someone wrote that mapping down;
`charged -> billing` is not.

Two secondary defects in the same scorer:

- No IDF weighting. A match on "the" counts the same as a match on "refund".
- No length normalization. Longer chunks win by having more tokens to match.

## Goal

Semantic retrieval that matches on meaning rather than shared tokens, with no
new service dependency and no per-query cost.

Explicitly **not** a goal: supporting a larger corpus. See Rejected Alternatives.

## Corpus facts

Measured, not estimated:

| | |
|---|---|
| Documents | 7 markdown files in `backend/cloudnest_docs/` |
| Total size | 8.3 KB / 1,330 words |
| Chunks after heading split | 46 |

These numbers drive most of the decisions below. At 46 chunks, brute-force
cosine over an in-memory matrix is the search index.

## Approach

Embed all 46 chunks ahead of time with a local model, commit the vectors, and at
request time embed only the query and take the top 3 by cosine similarity.

The embedding model runs locally via ONNX Runtime. `sentence-transformers` is not
used: it depends on torch (~800 MB), which exceeds Vercel's 250 MB unzipped
function limit. The model itself is small — torch is the problem, not the weights.

### Deploy size budget

| Component | Wheel size |
|---|---|
| `onnxruntime` 1.27.0 | 18.7 MB |
| `numpy` 2.5.1 | 16.7 MB |
| `tokenizers` 0.23.1 | 3.3 MB |
| `all-MiniLM-L6-v2` int8 ONNX | 23.0 MB |
| `tokenizer.json` | 0.5 MB |

~62 MB compressed, projected ~145 MB installed against a 250 MB cap.

Use the **int8-quantized** model (`model_quint8_avx2.onnx`, 23 MB). The fp32
export is 90 MB and pushes the total to ~210 MB — too close to the limit to be
comfortable.

**The installed size is a projection from wheel sizes and must be verified
against a real deployment before this is considered done.**

## Components

### `backend/model/` (new, committed)

`model_quint8_avx2.onnx` and `tokenizer.json`, downloaded once from
`sentence-transformers/all-MiniLM-L6-v2` and committed.

Committed rather than fetched at build time: `vercel.json` already declares
`includeFiles: "backend/**"`, so no build configuration changes, and deploys stay
reproducible without a network dependency. Cost is 23.5 MB in git history.

### `backend/embed.py` (new)

Single public function, shared by the indexer and the runtime:

```
embed(texts: list[str]) -> np.ndarray   # shape (len(texts), 384), L2-normalized
```

`InferenceSession` and `Tokenizer` are constructed at **module scope**, not per
call, so the ~1-3s model load is paid once per warm instance.

Pipeline:

1. Tokenize with `tokenizers.Tokenizer.from_file()`, truncation at 256 tokens.
2. Run the session. Build the input dict from `session.get_inputs()` rather than
   hardcoding names — exports vary in whether `token_type_ids` is present.
3. **Attention-mask-weighted mean pooling** over `last_hidden_state`:
   sum token vectors where mask is 1, divide by the mask sum.
4. L2-normalize each row.

Steps 3 and 4 are load-bearing. The ONNX export emits per-token vectors, not a
sentence vector. Skipping the pooling, or using an unmasked mean that averages in
padding, degrades retrieval quality badly and silently — the code still runs and
still returns plausible-looking rankings.

Normalizing at embed time means cosine similarity is a plain dot product
everywhere downstream.

### `backend/build_index.py` (new)

Run manually when documentation changes.

Reuses `load_chunks()` from `app.py` so build-time and runtime chunking cannot
drift. Calls `embed()` on all chunk texts. Writes `backend/index.npz`:

| Key | Contents |
|---|---|
| `vectors` | float32, `(n_chunks, 384)`, L2-normalized |
| `meta` | JSON string: list of `{doc, title, text}`, row-aligned with `vectors` |
| `docs_hash` | SHA-256 over the concatenated `cloudnest_docs/*.md` contents |
| `model_id` | model filename, for provenance |

`meta` is stored as a **JSON-encoded string**, not an object array. `np.savez`
cannot serialize a list of dicts without `allow_pickle=True`, and loading pickled
arrays executes arbitrary code on deserialization — not a risk worth taking for a
file that ships in the deploy bundle.

`docs_hash` is the staleness guard. Editing a doc without re-running the indexer
must not silently search stale vectors.

#### `--calibrate` mode

Cosine similarities do not live on the same scale as the current
`matched_terms / len(terms)` score, so the existing `0.25` threshold is
meaningless after this change and cannot be guessed.

`--calibrate` embeds two fixed question sets — in-scope (answerable from the
docs) and out-of-scope ("what's the weather", "how do I file taxes") — prints the
top-1 similarity distribution for each, and reports the separating gap. The
threshold is chosen from that output, not from intuition.

### `backend/app.py` (changed)

**`retriever()` rewritten.** Embed the question, compute `vectors @ q`, take the
top 3 by similarity. `confidence` becomes the top-1 cosine similarity.

**Existing lexical scoring preserved** as `lexical_retrieve()`, retaining
`tokenize`, `SYNONYMS`, and `STOPWORDS`. It runs only when `index.npz` is absent,
unreadable, or its `docs_hash` does not match the docs on disk. Failures log the
exception type only, never the message — matching the existing convention in
`llm_answer` (`app.py:99-102`), which exists because exception reprs can embed
secrets.

**Deleted:** `CATEGORY_DOCS` and the `boost = 2 if chunk["doc"] in preferred`
line. Semantic ranking should not be overridden by a keyword-derived category —
a 2x multiplier on the wrong category actively promotes worse matches.

**Kept:** `router()`, `BILLING_WORDS`, `TECH_WORDS`. The category is rendered as
a route badge at `frontend/src/App.jsx:49` and remains a user-visible feature.
It no longer influences ranking.

**`CONFIDENCE_THRESHOLD`** set from the `--calibrate` result.

### `backend/index.py` (changed)

`/api/health` additionally returns the confidence threshold.

### `frontend/src/App.jsx` (changed)

Line 98 currently hardcodes `0.25` with the comment "mirrors the graph's
conditional edge". Read the threshold from `/api/health` instead. The literal is
duplicated across backend and frontend today; after recalibration a stale
duplicate would put the clarify-request styling out of sync with the graph's
actual routing decision.

### `backend/requirements.txt` (changed)

Add `onnxruntime`, `tokenizers`, `numpy`.

## Data flow

```
build time:  cloudnest_docs/*.md -> load_chunks() -> embed() -> index.npz
                                                                    |
request:     question -> router() -> category ---------------------+
                      -> embed() -> q -> vectors @ q -> top 3 -> context
                                              |
                                        top-1 sim -> confidence -> responder | clarify
```

`load_chunks()` is the single definition of chunking, called from both paths.

## Failure modes

| Condition | Behavior |
|---|---|
| `index.npz` missing or unreadable | Fall back to `lexical_retrieve()` |
| `docs_hash` mismatch | Fall back to `lexical_retrieve()` |
| ONNX session fails to load | Fall back to `lexical_retrieve()` |
| No `ANTHROPIC_API_KEY` | Unchanged — extractive answers from retrieved chunks |

Retrieval and answer generation degrade independently. With neither the model nor
an API key, the app still returns extractive answers via lexical retrieval — the
current offline behavior is preserved.

## Testing

- `embed()` returns unit-norm rows; two paraphrases score higher against each
  other than against an unrelated sentence.
- Pooling is mask-aware: a short text embedded alone and embedded in a batch
  padded to a longer length produce the same vector.
- `docs_hash` mismatch triggers the lexical path.
- Regression set: questions that fail on the current lexical scorer
  (the mid-cycle-charge case above) return the correct chunk.
- Out-of-scope questions score below threshold and route to `clarify`.

## Rejected alternatives

**A vector database (Pinecone, pgvector, Chroma).** At 46 chunks a vector DB is a
network hop wrapped around a 46x384 matrix. Brute-force numpy cosine over that is
sub-millisecond. It would add latency, a service dependency, and cold-start cost
while returning identical results.

**A hosted embedding API (Voyage, OpenAI, Cohere).** Better model quality, but
adds a second provider key and a per-query network call, and breaks offline
operation. MiniLM is sufficient at this corpus size.

**BM25.** Fixes the IDF and length-normalization defects with zero dependencies,
but does not address vocabulary mismatch — the primary problem. Still lexical.

**Whole corpus in the prompt.** At ~1,800 tokens the entire documentation set
fits in a single request, which strictly beats any retrieval scheme on answer
quality (retrieval can only lose information the model would otherwise see) and
costs roughly $0.01 per 1,000 queries. Rejected because it deletes the retriever
node, which is the substance of this project. Recorded here because it is the
correct engineering answer at 8.3 KB and should be revisited if the retrieval
architecture ever stops being the point.

## Known costs

- **Cold start.** Loading the ONNX session adds ~1-3s to the first request on a
  new instance. Vercel Fluid Compute reuses instances, so this amortizes, but the
  first hit after idle is slow.
- **Model quality.** MiniLM (384 dims, 256-token window) is weaker than a current
  hosted embedding model. The 256-token window is adequate for the current chunk
  sizes but constrains how large chunks can grow.
- **Repo size.** 23.5 MB of model artifacts in git history.
