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


def test_rrf_fuse_ranks_chunk_high_in_both_lists_first():
    a = {"doc": "d1", "title": "A", "text": "a"}
    b = {"doc": "d2", "title": "B", "text": "b"}
    c = {"doc": "d3", "title": "C", "text": "c"}
    # A is 1st in list one and 1st in list two; B and C each strong in only one
    fused = app.rrf_fuse([[a, b, c], [a, c, b]])
    assert fused[0]["title"] == "A"


def test_rrf_fuse_surfaces_chunk_high_in_either_list():
    a = {"doc": "d1", "title": "A", "text": "a"}
    b = {"doc": "d2", "title": "B", "text": "b"}
    # B is buried in list one but 1st in list two; it must not be last after fusion
    long_tail = [{"doc": f"d{i}", "title": f"T{i}", "text": ""} for i in range(10)]
    list_one = [a] + long_tail + [b]
    list_two = [b, a]
    fused = app.rrf_fuse([list_one, list_two])
    titles = [c["title"] for c in fused]
    assert titles.index("B") < titles.index("T9")


def test_rrf_fuse_deduplicates_by_doc_and_title():
    a1 = {"doc": "d1", "title": "A", "text": "from list one"}
    a2 = {"doc": "d1", "title": "A", "text": "from list two"}
    fused = app.rrf_fuse([[a1], [a2]])
    assert len(fused) == 1


def test_rrf_fuse_handles_empty_list():
    a = {"doc": "d1", "title": "A", "text": "a"}
    fused = app.rrf_fuse([[], [a]])
    assert [c["title"] for c in fused] == ["A"]


def test_semantic_retrieve_returns_full_ranking_and_cosine_confidence():
    r = app.semantic_retrieve("how much does the pro plan cost")
    assert len(r["ranking"]) == len(app.CHUNKS)  # full corpus, ranked
    assert 0.0 < r["confidence"] <= 1.0


def test_semantic_retrieve_empty_when_index_missing(monkeypatch):
    monkeypatch.setattr(app, "INDEX", None)
    r = app.semantic_retrieve("refund")
    assert r["ranking"] == []
    assert r["confidence"] == 0.0


def test_lexical_retrieve_now_exposes_ranking_and_rescue():
    r = app.lexical_retrieve("cloudnest-cli linux automation")
    assert r["context"]                       # existing contract preserved
    assert 0.0 <= r["confidence"] <= 1.0       # existing contract preserved
    assert len(r["ranking"]) == len(app.CHUNKS)
    assert isinstance(r["rescue"], bool)


def test_fuse_uses_semantic_confidence_when_available():
    semantic = {"ranking": [{"doc": "d", "title": "A", "text": ""}], "confidence": 0.44}
    lexical = {"ranking": [{"doc": "d", "title": "A", "text": ""}],
               "confidence": 1.0, "context": [], "rescue": True}
    fused = app.fuse_results(semantic, lexical)
    assert fused["confidence"] == 0.44         # cosine wins, not the lexical ratio
    assert fused["lexical_rescue"] is True


def test_fuse_falls_back_to_lexical_confidence_when_degraded():
    semantic = {"ranking": [], "confidence": 0.0}
    lexical = {"ranking": [{"doc": "d", "title": "A", "text": ""}],
               "confidence": 0.8, "context": [], "rescue": False}
    fused = app.fuse_results(semantic, lexical)
    assert fused["confidence"] == 0.8          # degraded mode keeps answering
    assert fused["context"]                    # lexical ranking carries context


def test_extractive_fallback_caps_sections(monkeypatch):
    """No-API-key fallback must show only the top few sections, not all 46.
    context stays the full corpus (for the LLM/citations); only the extractive
    print is bounded.
    """
    monkeypatch.setattr(app, "llm_answer", lambda *a, **k: None)  # force extractive branch
    fake = [{"doc": "d", "title": f"S{i}", "text": f"body{i}"} for i in range(46)]
    state = {"messages": [{"role": "user", "content": "x"}], "context": fake}
    out = app.responder(state)
    answer = out["messages"][-1]["content"]
    shown = [i for i in range(46) if f"S{i}" in answer]
    assert len(shown) <= app.EXTRACTIVE_SECTIONS
    assert len(shown) >= 1


def test_parallel_nodes_write_separate_state_keys():
    q = "how much does the pro plan cost"
    state = {"messages": [{"role": "user", "content": q}]}
    sem = app.semantic_retriever(state)
    lex = app.lexical_retriever(state)
    assert set(sem) == {"semantic"} and "ranking" in sem["semantic"]
    assert set(lex) == {"lexical"} and "ranking" in lex["lexical"]


def test_fusion_node_produces_context_and_confidence():
    q = "how much does the pro plan cost"
    state = {"messages": [{"role": "user", "content": q}]}
    state.update(app.semantic_retriever(state))
    state.update(app.lexical_retriever(state))
    out = app.fusion(state)
    assert out["context"]
    assert out["confidence"] >= app.CONFIDENCE_THRESHOLD
    assert isinstance(out["lexical_rescue"], bool)


def test_graph_still_answers_confident_question_through_parallel_path():
    graph = app.build_app()
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "how do I get a refund"}]},
        {"configurable": {"thread_id": "test-parallel-confident"}},
    )
    assert result["clarified"] is False
    assert result["messages"][-1]["content"]


def _run(messages):
    return app.build_app().invoke(
        {"messages": messages},
        {"configurable": {"thread_id": f"t-{abs(hash(str(messages)))}"}},
    )


def test_explicit_human_request_escalates_and_builds_ticket():
    result = _run([{"role": "user", "content": "this is useless, let me talk to a human"}])
    assert result["escalate"] is True
    assert result["clarified"] is False
    assert result["ticket"]["reason"] == "user_requested"
    assert result["ticket"]["id"] and result["ticket"]["created_at"]
    assert result["ticket"]["conversation"]  # full history captured


def test_offtopic_noise_never_escalates_even_when_repeated():
    prior_clarify = app.CLARIFY_OFFTOPIC_MSG
    result = _run([
        {"role": "user", "content": "cat mouse banana"},
        {"role": "assistant", "content": prior_clarify},
        {"role": "user", "content": "table chair cloud"},
    ])
    assert result["escalate"] is False
    assert result["clarified"] is True
    assert result["ticket"] is None


def test_repeated_borderline_miss_escalates():
    # Force the borderline band deterministically, independent of the corpus.
    import app as _app

    def fake_fusion(state):
        return {"context": [], "confidence": 0.28, "lexical_rescue": False}

    graph_state = [
        {"role": "user", "content": "something vaguely about my plan tier"},
        {"role": "assistant", "content": _app.CLARIFY_BORDERLINE_MSG},
        {"role": "user", "content": "the other thing i mentioned"},
    ]
    # gate reads confidence + prior clarify text from history
    assert _app.picker_for_test(0.28, graph_state, lexical_rescue=False) == "escalate"
    assert _app.picker_for_test(0.28, graph_state[:1], lexical_rescue=False) == "clarify"


def test_borderline_with_lexical_rescue_answers():
    import app as _app
    msgs = [{"role": "user", "content": "cloudnest-cli"}]
    assert _app.picker_for_test(0.28, msgs, lexical_rescue=True) == "responder"


def test_offtopic_below_floor_never_escalates_via_picker():
    import app as _app
    msgs = [
        {"role": "user", "content": "cat"},
        {"role": "assistant", "content": _app.CLARIFY_OFFTOPIC_MSG},
        {"role": "user", "content": "mouse"},
    ]
    assert _app.picker_for_test(0.10, msgs, lexical_rescue=False) == "clarify"


def test_escalated_result_carries_ticket_and_no_sources():
    result = app.build_app().invoke(
        {"messages": [{"role": "user", "content": "let me talk to a real person"}]},
        {"configurable": {"thread_id": "test-api-ticket"}},
    )
    # what index.py will forward to the client
    assert result["escalate"] is True
    assert result["ticket"] is not None
    sources = [] if (result["clarified"] or result["escalate"]) else \
        app.sources_from_context(result["context"])[:3]
    assert sources == []
