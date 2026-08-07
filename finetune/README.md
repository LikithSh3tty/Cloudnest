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
  register rather than the docs' own vocabulary.

  ```bash
  python finetune/make_pairs.py --per-chunk 10
  ```

  **`data/train_pairs.jsonl` is currently a partial sweep, not the full
  corpus: 398 rows covering 36 of the 46 sections in
  `backend.app.load_chunks()`.** Generation ran out of Anthropic API
  credit mid-run and was not completed. These 10 sections have **zero**
  training pairs:

  ```
  06_security_privacy.md   Compliance
  06_security_privacy.md   Third-Party Sharing
  06_security_privacy.md   Security Incident Reporting
  07_general_faq.md        CloudNest — General FAQ
  07_general_faq.md        Is there a mobile app?
  07_general_faq.md        Can I share files with people who don't have CloudNest?
  07_general_faq.md        What happens to my files if I stop paying?
  07_general_faq.md        Does CloudNest have a referral program?
  07_general_faq.md        Is there a desktop notification for sync errors?
  07_general_faq.md        Who do I contact for enterprise/custom plans above Team tier?
  ```

  Anyone training on this file, or evaluating why a fine-tuned model
  underperforms on those sections, should know that gap exists before
  drawing conclusions. Full detail, including a widened leakage check and
  a separate (unresolved) mislabeling issue in 61 of the 398 rows, is in
  `.superpowers/sdd/2026-08-07-finetune-retriever/task-3-report.md`.

Scripts that fine-tune the model (Task 4 onward: fine-tuning MiniLM,
exporting int8 ONNX, building the tuned index) land in this directory in
later tasks and will be documented here as they're added.

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
#    Currently partial - 36 of 46 sections, see the note above.
# 3. (Task 4) Fine-tune MiniLM on those pairs; export int8 ONNX.
# 4. (Task 5) Build the tuned index:
EMBED_VARIANT=finetuned python backend/build_index.py

# 5. Benchmark both variants with the same held-out set:
python finetune/eval_retrieval.py --variant baseline
python finetune/eval_retrieval.py --variant finetuned

# 6. (Task 6) Only if the tuned int8 variant beats baseline int8 on the
#    held-out set does it become the default. A null result is an
#    acceptable, reportable outcome — the eval set is not adjusted to
#    manufacture a win.
```

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
