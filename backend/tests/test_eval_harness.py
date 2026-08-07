import json
import os
import sys
from pathlib import Path

import pytest

FINETUNE = Path(__file__).resolve().parent.parent.parent / "finetune"
sys.path.insert(0, str(FINETUNE))

from eval_retrieval import load_eval_set, rank_for, score  # noqa: E402

from app import load_chunks, load_index  # noqa: E402

EVAL_PATH = FINETUNE / "data" / "eval_questions.jsonl"

_VARIANT_MODULES = ("variant", "embed", "app")


@pytest.fixture
def clean_variant_state():
    """Hard-reset EMBED_VARIANT and the variant/embed/app module cache
    before the test, and restore whatever was actually there afterward.

    Without this, the Finding-3 regression tests below are order-dependent:
    if an earlier test in this file already called score("baseline") and
    left EMBED_VARIANT set (the bug this fixture exists to catch), a later
    test that merely snapshots os.environ at its own start would find
    EMBED_VARIANT already leaked-but-present, making its "restored to the
    pre-call value" assertion trivially true regardless of whether score()
    itself does any restoring. Hard-clearing to a known state before the
    test removes that dependency on collection order and on what any other
    test happened to leave behind.
    """
    had_env = "EMBED_VARIANT" in os.environ
    prior_value = os.environ.get("EMBED_VARIANT")
    saved_modules = {name: sys.modules[name] for name in _VARIANT_MODULES if name in sys.modules}

    os.environ.pop("EMBED_VARIANT", None)
    for name in _VARIANT_MODULES:
        sys.modules.pop(name, None)

    yield

    for name in _VARIANT_MODULES:
        sys.modules.pop(name, None)
    sys.modules.update(saved_modules)
    if had_env:
        os.environ["EMBED_VARIANT"] = prior_value
    else:
        os.environ.pop("EMBED_VARIANT", None)


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


def test_score_restores_environment_after_call(clean_variant_state):
    """score() mutates EMBED_VARIANT and pops sys.modules entries while it
    runs, but Task 5 calls it for two variants back to back in one process
    — leftover env/module state from one call must never leak into the
    next.

    Uses clean_variant_state instead of snapshotting os.environ at the top
    of the test: a self-snapshot would pass trivially if an earlier test
    already leaked EMBED_VARIANT="baseline" into the environment, since
    "restored to the pre-call value" and "leaked value never cleared" look
    identical from inside this test. Hard-clearing first makes the
    pre-condition known (absent), so the post-condition actually exercises
    score()'s own restore logic.
    """
    index = load_index()
    if index is None:
        pytest.skip("no index available")

    # Captured into a plain local before each assert, not inlined as
    # `os.environ.get(...)` in the assert expression itself: pytest's
    # assertion rewriter shows every sub-expression's value on failure,
    # and os.environ in this repo holds real API keys/tokens. Asserting on
    # a local name keeps a failure's introspection to just that one
    # (non-secret) value, never the container it came from.
    embed_variant = os.environ.get("EMBED_VARIANT")
    assert embed_variant is None
    loaded = [sys.modules.get(name) for name in _VARIANT_MODULES]
    assert loaded == [None, None, None]

    score("baseline")

    embed_variant = os.environ.get("EMBED_VARIANT")
    assert embed_variant is None, "score() must not leave EMBED_VARIANT set after returning"
    loaded = [sys.modules.get(name) for name in _VARIANT_MODULES]
    assert loaded == [None, None, None], (
        "score() must not leave variant/embed/app in sys.modules after returning"
    )


def test_cross_variant_calls_do_not_corrupt_each_other(clean_variant_state):
    """The scenario Finding 3 exists to protect: Task 5 calls score() for
    two variants back to back in one process. "finetuned" has no built
    index yet, so that call raises SystemExit from the missing-index
    branch — and a baseline call made afterward must be unaffected by it.

    A same-variant idempotency check (call "baseline" twice) cannot catch
    this: leaked state from a "baseline" call is itself valid "baseline"
    state, so a second "baseline" call produces the same answer whether or
    not anything was restored in between. Only an intervening call for a
    *different* variant can expose leftover EMBED_VARIANT or a stale
    module still configured for that other variant.
    """
    index = load_index()
    if index is None:
        pytest.skip("no index available")

    first = score("baseline")

    with pytest.raises(SystemExit):
        score("finetuned")

    second = score("baseline")

    assert first == second
