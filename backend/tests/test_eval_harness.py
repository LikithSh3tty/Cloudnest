import json
import sys
from pathlib import Path

import pytest

FINETUNE = Path(__file__).resolve().parent.parent.parent / "finetune"
sys.path.insert(0, str(FINETUNE))

from eval_retrieval import load_eval_set, rank_for, score  # noqa: E402

from app import load_chunks, load_index  # noqa: E402

EVAL_PATH = FINETUNE / "data" / "eval_questions.jsonl"


def test_eval_set_is_non_trivial():
    rows = load_eval_set(EVAL_PATH)
    assert len(rows) >= 40, "eval set too small to distinguish real gains from noise"


def test_every_label_resolves_to_a_real_section():
    known = {(c["doc"], c["title"]) for c in load_chunks()}
    for row in load_eval_set(EVAL_PATH):
        assert (row["doc"], row["title"]) in known, f"bad label: {row}"


def test_eval_questions_are_unique():
    questions = [r["question"] for r in load_eval_set(EVAL_PATH)]
    assert len(questions) == len(set(questions)), "duplicate eval questions skew the metrics"


def test_ranking_returns_the_whole_corpus_in_order():
    index = load_index()
    if index is None:
        pytest.skip("no index available")
    vectors, meta = index
    ranked = rank_for("how much does the pro plan cost", vectors, meta)
    assert len(ranked) == len(meta)
    assert {(c["doc"], c["title"]) for c in ranked} == {(c["doc"], c["title"]) for c in meta}


def test_score_reports_all_metrics():
    index = load_index()
    if index is None:
        pytest.skip("no index available")
    result = score("baseline")
    for key in ("recall@1", "recall@3", "mrr@10", "n", "variant"):
        assert key in result
    assert 0.0 <= result["recall@1"] <= 1.0
    assert result["recall@1"] <= result["recall@3"], "recall@3 cannot be below recall@1"


def test_score_restores_environment_after_call():
    """score() mutates EMBED_VARIANT and pops sys.modules entries while it
    runs, but Task 5 calls it for two variants back to back in one process
    — leftover env/module state from one call must never leak into the
    next."""
    index = load_index()
    if index is None:
        pytest.skip("no index available")
    import os

    had_env = "EMBED_VARIANT" in os.environ
    prior_value = os.environ.get("EMBED_VARIANT")

    score("baseline")

    if had_env:
        assert os.environ.get("EMBED_VARIANT") == prior_value
    else:
        assert "EMBED_VARIANT" not in os.environ


def test_score_is_idempotent_across_repeated_calls():
    index = load_index()
    if index is None:
        pytest.skip("no index available")
    first = score("baseline")
    second = score("baseline")
    assert first == second
