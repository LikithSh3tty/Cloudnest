import numpy as np
import pytest

import app


def test_semantic_retrieval_finds_billing_docs_without_shared_tokens():
    """The motivating failure: no token overlap with the pricing docs."""
    state = {
        "messages": [{"role": "user", "content": "how much am I charged mid cycle"}],
        "category": "billing",
    }
    result = app.retriever(state)
    assert result["context"], "expected at least one chunk"
    docs = {c["doc"] for c in result["context"]}
    assert "02_pricing_billing.md" in docs
    assert result["confidence"] > 0.0


def test_out_of_scope_question_scores_below_in_scope():
    def confidence(question):
        return app.retriever({
            "messages": [{"role": "user", "content": question}],
            "category": "general",
        })["confidence"]

    assert confidence("how do I get a refund") > confidence(
        "what is the capital of Peru"
    )


def test_falls_back_to_lexical_when_index_missing(monkeypatch):
    monkeypatch.setattr(app, "INDEX", None)
    result = app.retriever({
        "messages": [{"role": "user", "content": "refund"}],
        "category": "billing",
    })
    assert result["context"]


def test_falls_back_to_lexical_when_embed_unavailable(monkeypatch):
    monkeypatch.setattr(app, "embed", None)
    result = app.retriever({
        "messages": [{"role": "user", "content": "refund"}],
        "category": "billing",
    })
    assert result["context"]


def test_stale_index_is_rejected(monkeypatch, tmp_path):
    """A docs_hash mismatch must not be searched."""
    monkeypatch.setattr(app, "docs_hash", lambda: "0" * 64)
    assert app.load_index() is None


def test_lexical_retrieve_still_works():
    result = app.lexical_retrieve("refund policy")
    assert result["context"]
    assert 0.0 <= result["confidence"] <= 1.0


def test_category_boost_is_gone():
    assert not hasattr(app, "CATEGORY_DOCS")
