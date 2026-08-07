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


def test_score_restores_environment_after_a_failed_call(clean_variant_state):
    """The exception path is the real hazard Finding 3 was about: score()
    sets os.environ["EMBED_VARIANT"] near the top of the function, then
    raises SystemExit from the missing-index branch when a variant has no
    built index. This test needs "finetuned" to genuinely have no built
    index to reach that branch for real, which was true by accident of task
    ordering until Task 5 built backend/index_finetuned.npz (the whole point
    of that task). So the file is hidden for the span of the score() call
    below and restored immediately after — manufacturing the same condition
    on demand instead of depending on it being naturally absent.

    The hide/restore happens *inside* the test body, after the general
    "is any index available" skip guard, not in a fixture: this whole suite
    also runs under EMBED_VARIANT=finetuned (Task 5, Step 7), where
    backend/app.py resolves INDEX_PATH to index_finetuned.npz for the
    process. A fixture that hides that file before the test body runs would
    make the guard's own load_index() call see no index and skip — even
    though the guard's job is only to skip on a fresh checkout with no
    index built at all, not to react to this test's own setup.

    (A same-variant or cross-variant *success* comparison was tried in an
    earlier round and rejected: score() unconditionally pops and
    reimports variant/embed/app at the top of every call regardless of
    the try/finally fix, so a later successful call self-heals any
    leftover module state and the comparison can't fail against the bug.
    The env var set on the line before the raise, never unset because the
    raise skips everything after it, is the part that only a real
    exception path exercises.)
    """
    index = load_index()
    if index is None:
        pytest.skip("no index available")

    # Captured into a plain local before each assert, never asserted on
    # os.environ directly: an earlier round's `"EMBED_VARIANT" not in
    # os.environ` assertion made pytest print the whole environment —
    # including real API keys and tokens on this machine — into the
    # failure traceback. Asserting on a local keeps a failure's
    # introspection to just that one non-secret value.
    embed_variant = os.environ.get("EMBED_VARIANT")
    assert embed_variant is None

    backend = Path(__file__).resolve().parent.parent
    index_path = backend / "index_finetuned.npz"
    hidden_path = backend / "index_finetuned.npz.hidden-for-test"
    index_existed = index_path.exists()
    if index_existed:
        index_path.rename(hidden_path)
    try:
        with pytest.raises(SystemExit):
            score("finetuned")
    finally:
        if index_existed:
            hidden_path.rename(index_path)

    embed_variant = os.environ.get("EMBED_VARIANT")
    assert embed_variant is None, (
        "score() must restore EMBED_VARIANT even when it raises SystemExit"
    )
