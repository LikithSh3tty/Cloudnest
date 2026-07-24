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


def test_index_from_a_different_model_is_rejected(monkeypatch):
    """Vectors built by another model must not be searched against new queries."""
    monkeypatch.setattr(app, "MODEL_ID", "some-other-model.onnx")
    assert app.load_index() is None


def test_lexical_retrieve_still_works():
    result = app.lexical_retrieve("refund policy")
    assert result["context"]
    assert 0.0 <= result["confidence"] <= 1.0


def test_category_boost_is_gone():
    assert not hasattr(app, "CATEGORY_DOCS")


def test_graph_reports_which_branch_answered():
    graph = app.build_app()
    config = {"configurable": {"thread_id": "test-routing"}}
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "what is the capital of Peru"}]},
        config,
    )
    assert result["clarified"] is True

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "how do I get a refund"}]},
        {"configurable": {"thread_id": "test-routing-2"}},
    )
    assert result["clarified"] is False


def test_fully_degraded_path_answers_through_the_graph(monkeypatch):
    """No model, no index, no API key: the graph must still answer.

    This is the feature's headline offline guarantee. The two halves (lexical
    fallback, and graph routing) are tested separately elsewhere; this drives
    the whole compiled flow with retrieval degraded to the keyword scorer.
    """
    monkeypatch.setattr(app, "embed", None)
    monkeypatch.setattr(app, "INDEX", None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    graph = app.build_app()
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "how do I get a refund"}]},
        {"configurable": {"thread_id": "test-degraded"}},
    )
    answer = result["messages"][-1]["content"]
    assert answer  # extractive fallback produced text
    assert result["clarified"] is False
