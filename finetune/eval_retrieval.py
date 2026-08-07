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

    Also returns a "by_kind" breakout (per-kind recall@1/recall@3/mrr@10/n,
    using the eval set's "kind" field) so open-ended synthesis questions
    that span multiple sections don't silently distort the headline metric
    for plan-selection-style questions that genuinely have one right answer.
    The headline recall@1/recall@3/mrr@10/n/variant keys and how they are
    computed are unchanged — Task 5 compares variants on those, so the
    baseline already recorded from them must stay valid.

    Global state (EMBED_VARIANT and the popped variant/embed/app modules) is
    restored on both the success and exception paths: Task 5 calls this for
    two variants in the same process, and a prior run's env var or module
    left behind would silently corrupt the next call.
    """
    had_env = "EMBED_VARIANT" in os.environ
    prior_env = os.environ.get("EMBED_VARIANT")
    os.environ["EMBED_VARIANT"] = variant
    sys.path.insert(0, str(BACKEND))
    saved_modules = {
        name: sys.modules[name] for name in ("variant", "embed", "app") if name in sys.modules
    }
    for module in ("variant", "embed", "app"):
        sys.modules.pop(module, None)

    try:
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
        by_kind_stats: dict[str, dict] = {}
        for row in rows:
            ranked = rank_for(row["question"], vectors, meta)
            gold = (row["doc"], row["title"])
            positions = [i for i, c in enumerate(ranked) if (c["doc"], c["title"]) == gold]
            rank = positions[0] if positions else len(ranked)
            hit1 = rank == 0
            hit3 = rank < 3
            rr = 1.0 / (rank + 1) if rank < 10 else 0.0

            hits_at_1 += hit1
            hits_at_3 += hit3
            reciprocal_ranks.append(rr)

            bucket = by_kind_stats.setdefault(
                row.get("kind", "unknown"), {"hits_at_1": 0, "hits_at_3": 0, "rr": [], "n": 0}
            )
            bucket["hits_at_1"] += hit1
            bucket["hits_at_3"] += hit3
            bucket["rr"].append(rr)
            bucket["n"] += 1

        n = len(rows)
        by_kind = {
            kind: {
                "recall@1": stats["hits_at_1"] / stats["n"],
                "recall@3": stats["hits_at_3"] / stats["n"],
                "mrr@10": float(np.mean(stats["rr"])),
                "n": stats["n"],
            }
            for kind, stats in sorted(by_kind_stats.items())
        }

        return {
            "variant": variant,
            "n": n,
            "recall@1": hits_at_1 / n,
            "recall@3": hits_at_3 / n,
            "mrr@10": float(np.mean(reciprocal_ranks)),
            "by_kind": by_kind,
        }
    finally:
        for module in ("variant", "embed", "app"):
            sys.modules.pop(module, None)
        sys.modules.update(saved_modules)
        if had_env:
            os.environ["EMBED_VARIANT"] = prior_env
        else:
            os.environ.pop("EMBED_VARIANT", None)


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
    print()
    print("by kind:")
    for kind, stats in result["by_kind"].items():
        print(
            f"  {kind:10s} n={stats['n']:<3d} "
            f"recall@1={stats['recall@1']:.3f} "
            f"recall@3={stats['recall@3']:.3f} "
            f"MRR@10={stats['mrr@10']:.3f}"
        )


if __name__ == "__main__":
    main()
