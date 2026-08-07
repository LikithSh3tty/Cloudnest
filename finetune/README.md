# Fine-tuning the retriever

Tools for measuring and improving CloudNest's retrieval embedding model.
This directory does not touch the live backend directly — it reads the
same docs and index format `backend/` uses, and writes its own artifacts
under an `EMBED_VARIANT=finetuned` name that the backend only loads if
that variant is explicitly selected.

## What each script does

- **`eval_retrieval.py`** — the measuring instrument. Loads the hand-labeled
  held-out set (`data/eval_questions.jsonl`), ranks the whole corpus by
  cosine similarity for every question, and reports `recall@1`, `recall@3`,
  and `MRR@10` for a given variant (`baseline` or `finetuned`). This is the
  only script whose numbers should be trusted for a before/after comparison,
  because it is the only one scored against questions a human wrote and
  labeled independently of the model.

  ```bash
  python finetune/eval_retrieval.py --variant baseline
  python finetune/eval_retrieval.py --variant finetuned
  ```

- **`data/eval_questions.jsonl`** — the held-out eval set itself: one JSON
  object per line, each with `question`, `doc` (filename in
  `cloudnest_docs/`), `title` (the section heading that actually answers the
  question), and `kind` (`lookup`, `multi`, or `hard`). Every `(doc, title)`
  pair was verified against `backend.app.load_chunks()` output before being
  committed — see Step 1 of the Task 2 brief for the verification snippet.

- **`make_pairs.py`** — generates `data/train_pairs.jsonl`, synthetic
  (question, section) training pairs written by Claude in a user's
  register rather than the docs' own vocabulary. Resumable: re-running it
  skips any section that already has rows and appends only what's
  missing, and always skips "bare H1 title" chunks (see below) regardless
  of coverage.

  ```bash
  python finetune/make_pairs.py --per-chunk 10
  ```

  **`data/train_pairs.jsonl` now covers 40 of the 46 sections in
  `backend.app.load_chunks()`, 464 rows.** Two things happened to get
  here, both recorded in the task-3 report:

  - The first generation pass hit the Anthropic API credit wall mid-run
    (section 37 of 46) and 61 of its rows had come from 5 near-
    content-free "bare H1 title" chunks — a property of `load_chunks()`
    itself, since 6 of the 7 docs have no intro paragraph before their
    first `##` heading, so that first "chunk" is just the title line.
    Those 61 rows were hand-relabeled to the specific section that
    actually answers each question, or deleted where no section in the
    same document does (32 of the 61).
  - A second, resumed pass (after credits were restored) generated pairs
    for exactly the sections generation never reached the first time —
    skipping every section that already had rows, and skipping bare
    chunks outright rather than ever generating for them again.

  **6 sections have zero training pairs — all 6 are bare-H1 title
  chunks, correctly empty by design, not a coverage gap:**
  ```
  02_pricing_billing.md     CloudNest — Pricing & Billing
  03_account_management.md  CloudNest — Account Management
  04_technical_setup.md     CloudNest — Technical Setup
  05_troubleshooting.md     CloudNest — Troubleshooting
  06_security_privacy.md    CloudNest — Security & Privacy
  07_general_faq.md         CloudNest — General FAQ
  ```

  This is effectively full coverage of the content-bearing corpus (40 of
  46 chunks; the remaining 6 have no content to train on). Full detail,
  including the widened leakage check, the relabel/delete mapping applied
  to the 61 rows, and the resumed top-up run, is in
  `.superpowers/sdd/2026-08-07-finetune-retriever/task-3-report.md`.

- **`train.py`** — fine-tunes `sentence-transformers/all-MiniLM-L6-v2` on
  `data/train_pairs.jsonl` with `MultipleNegativesRankingLoss` (in-batch
  negatives, `BatchSamplers.NO_DUPLICATES`), 3 epochs, batch size 32 (15
  batches/epoch, `ceil(464/32)`). Writes an fp32 checkpoint to
  `artifacts/finetuned-fp32/` (gitignored — reproduce by re-running, not by
  fetching from history).

  ```bash
  python finetune/train.py --epochs 3 --batch-size 32
  ```

- **`export_onnx.py`** — exports the fp32 checkpoint to int8 ONNX, matching
  the serving format the frozen baseline already ships in (`backend/embed.py`
  expects `input_ids`/`attention_mask` in, per-token hidden states out, no
  pooling layer baked into the graph). Writes
  `backend/model/model_finetuned_quint8_avx2.onnx`.

  ```bash
  python finetune/export_onnx.py
  ```

Build the tuned index with `EMBED_VARIANT=finetuned python backend/build_index.py`,
then compare against baseline with `eval_retrieval.py` (see below).

## `data/eval_questions.jsonl` is held out — never train on it

This file exists specifically so there is one measurement that cannot be
gamed by the training process. Synthetic training pairs (Task 3) are
generated *from* a doc chunk and trained to retrieve that same chunk; a
model can memorize the generator's phrasing and score near-perfectly on
its own training pairs while learning nothing about how a real user
asks a question. `eval_questions.jsonl` is hand-labeled independently,
by reading the docs and confirming each section actually answers the
question — not by reusing or paraphrasing anything the training-data
generator produces.

**Rule:** nothing under `finetune/data/eval_questions.jsonl` may appear in,
or be derived into, any training-pair file. If a training-pair generator
ever needs example phrasing, write new examples — do not sample or
paraphrase from this file.

## Command order to reproduce the pipeline

```bash
# 1. Measure the frozen baseline first (this file's job) — always do this
#    before any training exists, so there is an honest reference point.
python finetune/eval_retrieval.py --variant baseline

# 2. (Task 3) Generate synthetic training pairs from the docs.
#    40 of 46 sections; the other 6 are bare-H1 chunks with no
#    content, see the note above.
# 3. (Task 4) Fine-tune MiniLM on those pairs; export int8 ONNX.
# 4. (Task 5) Build the tuned index:
EMBED_VARIANT=finetuned python backend/build_index.py

# 5. Benchmark both variants with the same held-out set:
python finetune/eval_retrieval.py --variant baseline
python finetune/eval_retrieval.py --variant finetuned

# 6. Apply the pre-registered rule: tuned becomes default only if it beats
#    baseline on recall@1 AND does not regress recall@3.
```

## Result

|          | recall@1 | recall@3 | MRR@10 |
|---|---|---|---|
| baseline (frozen int8) | 0.667 | 0.926 | 0.788 |
| fine-tuned (fp32) | 0.648 | 0.963 | 0.793 |
| fine-tuned (int8, served) | 0.648 | 0.963 | 0.792 |

recall@1 regressed (0.667 → 0.648), so by the pre-registered rule **`baseline`
stays the default** — `backend/variant.py`'s fallback is unchanged. At n=54
none of these deltas clear the noise floor (~6.5pp binomial standard error
near p≈0.65) required for conventional confidence, so this is not read as
"fine-tuning made things worse" either — the honest statement is that this
run cannot distinguish fine-tuning's effect from noise on any headline
metric. The full writeup, including the empty-control-slice caveat (every
gold section the 54 eval questions target already has training pairs, so
there is no untrained slice to check for regression against) and the rest
of the limitations, lives in the repo root [`README.md`](../README.md#fine-tuning-the-retriever),
not duplicated here.

The tuned artifacts (`backend/model/model_finetuned_quint8_avx2.onnx`,
`backend/index_finetuned.npz`) stay committed and reachable with
`EMBED_VARIANT=finetuned`; nothing about this result removes them.

## Rollback guarantees

Four independent layers, strongest first:

1. **Default is baseline.** `EMBED_VARIANT` unset resolves to the baseline
   model + baseline index. A fresh clone with no env config behaves
   exactly as it does today.
2. **Baseline files are never written.** Nothing in this pipeline opens
   `model_quint8_avx2.onnx` or `index.npz` for writing. A test asserts
   their SHA-256 digests match the values recorded before any fine-tuning
   work began (`backend/tests/test_variant.py`).
3. **Single-flag revert.** If the tuned variant is ever made default and
   later regresses, set `EMBED_VARIANT=baseline` in the deployment env and
   redeploy. No code change, no rebuild.
4. **Full excision.** To remove this work entirely and return the tree to
   the current commit:

   ```bash
   git checkout -- backend/ README.md .gitignore && rm -rf finetune/ backend/index_finetuned.npz backend/model/model_finetuned_quint8_avx2.onnx
   ```

   Because baseline artifacts are untracked by this work, `git checkout`
   restores them exactly.
