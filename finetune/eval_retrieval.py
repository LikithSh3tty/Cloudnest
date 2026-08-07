"""Measure retrieval quality for one variant against the labeled eval set.

    python finetune/eval_retrieval.py --variant baseline
    python finetune/eval_retrieval.py --variant finetuned

Metrics are computed over the hand-labeled held-out set, never over the
generated training pairs: pairs are written from a chunk and would score
near-perfectly regardless of whether the model learned anything useful.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
DEFAULT_EVAL_PATH = Path(__file__).resolve().parent / "data" / "eval_questions.jsonl"


def load_eval_set(path: Path = DEFAULT_EVAL_PATH) -> list[dict]:
    """Read the labeled eval set. Each row: question, doc, title, kind."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def rank_for(question: str, vectors: np.ndarray, meta: list[dict]) -> list[dict]:
    """Order the whole corpus by descending cosine against the question."""
    sys.path.insert(0, str(BACKEND))
    from embed import embed

    scores = vectors @ embed([question])[0]
    return [meta[i] for i in np.argsort(-scores)]


def score(variant: str, eval_path: Path = DEFAULT_EVAL_PATH) -> dict:
    """Compute recall@1, recall@3, and MRR@10 for a variant.

    Loads the variant in a clean interpreter state: embed.py builds its ONNX
    session at import, so switching variants inside one process would keep
    the first model loaded and silently measure it twice.
    """
    os.environ["EMBED_VARIANT"] = variant
    sys.path.insert(0, str(BACKEND))
    for module in ("variant", "embed", "app"):
        sys.modules.pop(module, None)

    from app import load_index

    index = load_index()
    if index is None:
        raise SystemExit(
            f"no usable index for variant '{variant}'; "
            f"run: EMBED_VARIANT={variant} python backend/build_index.py"
        )
    vectors, meta = index

    rows = load_eval_set(eval_path)
    hits_at_1 = hits_at_3 = 0
    reciprocal_ranks = []
    for row in rows:
        ranked = rank_for(row["question"], vectors, meta)
        gold = (row["doc"], row["title"])
        positions = [i for i, c in enumerate(ranked) if (c["doc"], c["title"]) == gold]
        rank = positions[0] if positions else len(ranked)
        hits_at_1 += rank == 0
        hits_at_3 += rank < 3
        reciprocal_ranks.append(1.0 / (rank + 1) if rank < 10 else 0.0)

    n = len(rows)
    return {
        "variant": variant,
        "n": n,
        "recall@1": hits_at_1 / n,
        "recall@3": hits_at_3 / n,
        "mrr@10": float(np.mean(reciprocal_ranks)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="baseline",
                        choices=["baseline", "finetuned"])
    args = parser.parse_args()
    result = score(args.variant)
    print(f"variant     {result['variant']}")
    print(f"questions   {result['n']}")
    print(f"recall@1    {result['recall@1']:.3f}")
    print(f"recall@3    {result['recall@3']:.3f}")
    print(f"MRR@10      {result['mrr@10']:.3f}")


if __name__ == "__main__":
    main()
