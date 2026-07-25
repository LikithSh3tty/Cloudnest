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


def test_multi_section_query_surfaces_the_storage_section():
    """Regression: a trash/space/sync question needs both the Trash-recovery and
    the Storage-Full sections to answer correctly, but Storage Full ranks ~5th.
    The old top-3 slice dropped it and the bot wrongly said the docs didn't cover
    whether emptying Trash frees space. TOP_K must be wide enough to include it.
    """
    r = app.retriever({
        "messages": [{"role": "user", "content":
            "if my sync stopped because storage is full, does emptying the trash help it resume"}],
        "category": "technical",
    })
    titles = [c["title"] for c in r["context"]]
    assert any("Storage Full" in t for t in titles), titles
    assert len(r["context"]) <= app.TOP_K


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


def test_sources_are_unique_titles_in_order():
    context = [
        {"doc": "02_pricing_billing.md", "title": "Failed Payments", "text": "..."},
        {"doc": "02_pricing_billing.md", "title": "Failed Payments", "text": "..."},
        {"doc": "03_account_management.md", "title": "Password Reset", "text": "..."},
    ]
    assert app.sources_from_context(context) == ["Failed Payments", "Password Reset"]
    assert app.sources_from_context([]) == []


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


def test_contextualize_query_folds_in_prior_user_turn():
    messages = [
        {"role": "user", "content": "i want to add two-factor authentication"},
        {"role": "assistant", "content": "Sure, here's how to enable 2FA..."},
        {"role": "user", "content": "does it cost extra"},
    ]
    result = app.contextualize_query(messages)
    assert result == "i want to add two-factor authentication. does it cost extra"


def test_contextualize_query_first_turn_has_no_prior_to_fold():
    messages = [{"role": "user", "content": "does it cost extra"}]
    assert app.contextualize_query(messages) == "does it cost extra"


def test_contextualize_query_skips_blank_prior_message():
    messages = [
        {"role": "user", "content": "   "},
        {"role": "user", "content": "does it cost extra"},
    ]
    assert app.contextualize_query(messages) == "does it cost extra"


def test_contextualize_query_caps_prior_turn_length():
    long_prior = "x" * 500
    messages = [
        {"role": "user", "content": long_prior},
        {"role": "user", "content": "does it cost extra"},
    ]
    result = app.contextualize_query(messages)
    assert result == "x" * app.CONTEXT_CHAR_CAP + ". does it cost extra"


def test_ambiguous_followup_now_clears_threshold_and_finds_2fa():
    """Regression pinned to measured evidence: 'does it cost extra' alone scores
    0.285 (below the 0.30 threshold) and points at the wrong section. Folded
    with the turn before it, it must clear the threshold and find 2FA.
    """
    state = {
        "messages": [
            {"role": "user", "content": "i want to add two-factor authentication"},
            {"role": "assistant", "content": "Sure, here's how to enable 2FA..."},
            {"role": "user", "content": "does it cost extra"},
        ],
        "category": "general",
    }
    result = app.retriever(state)
    assert result["confidence"] >= app.CONFIDENCE_THRESHOLD
    assert result["context"][0]["title"] == "Two-Factor Authentication (2FA)"


def test_complex_multi_fact_query_surfaces_all_requested_sections():
    """Regression: a query touching many distinct facts (API, EU residency, bank
    transfer, CLI, unlimited devices) needs sections that rank far apart -
    "Payment Methods" (bank transfer) ranks ~10th and "CLI Tool" ranks ~22nd
    against this exact query, both well outside any small top-N. TOP_K must be
    wide enough that neither gets dropped, or the bot wrongly says it can't
    confirm facts the docs actually state.
    """
    question = (
        "I am a Linux developer with 12 employees. We need API access, EU data "
        "residency, bank transfer payments, CLI automation, and unlimited "
        "devices. Which plan should we choose, and why?"
    )
    result = app.retriever({
        "messages": [{"role": "user", "content": question}],
        "category": "billing",
    })
    titles = {c["title"] for c in result["context"]}
    assert "Payment Methods" in titles, titles
    assert "CLI Tool" in titles, titles
