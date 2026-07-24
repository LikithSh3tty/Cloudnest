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


if __name__ == "__main__":
    report()
