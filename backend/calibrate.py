"""Report the in-scope/out-of-scope similarity gap to pick CONFIDENCE_THRESHOLD.

Cosine similarity has no natural cutoff, and it is not on the same scale as the
matched-terms ratio the lexical scorer produced. Measure, do not guess.

    python calibrate.py --variant baseline
    EMBED_VARIANT=finetuned python calibrate.py --variant finetuned

--variant only labels the printout so the two thresholds are never confused;
the model actually measured is whichever EMBED_VARIANT selects (see
variant.py), since INDEX_PATH and the ONNX session are both resolved at
import time. Passing --variant finetuned without EMBED_VARIANT=finetuned in
the environment would measure baseline and print a misleading label, so the
mismatch is checked below rather than left to catch a stale caller silently.
"""
import argparse
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


def report(expected_variant: str | None = None) -> None:
    from variant import active_variant

    variant = active_variant()
    print(f"variant: {variant}")
    if expected_variant is not None and expected_variant != variant:
        print(
            f"WARNING: --variant {expected_variant!r} was requested but "
            f"EMBED_VARIANT resolves to {variant!r}; the numbers below are for "
            f"{variant!r}. Set EMBED_VARIANT={expected_variant} to match."
        )

    index = load_index()
    if index is None:
        print("no usable index; run: python backend/build_index.py")
        return
    vectors, _ = index

    def top1(questions):
        # embed one at a time to match production (embed([query])) and the index
        # build: this int8 model's output shifts with batch padding
        return np.array([float((embed([q])[0] @ vectors.T).max()) for q in questions])

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="baseline",
                        choices=["baseline", "finetuned"])
    args = parser.parse_args()
    report(expected_variant=args.variant)


if __name__ == "__main__":
    main()
