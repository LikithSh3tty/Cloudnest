import app


def test_answer_model_is_used_in_the_api_call():
    """The model id belongs in one named place, not inline in the API call,
    so swapping it is a one-line change with a test that notices. Checks
    ANSWER_MODEL is actually wired into the Anthropic call site (not just
    declared and ignored), which test_answer_model_is_sonnet's equality
    check on the constant alone can't tell apart from a dead constant."""
    import inspect
    source = inspect.getsource(app)
    assert "model=ANSWER_MODEL" in source, "ANSWER_MODEL must be passed to the API call"
    assert source.count('"claude-') <= 1, (
        "the model id should appear as a literal only in the ANSWER_MODEL "
        "constant, not hardcoded again at a call site"
    )


def test_answer_model_is_sonnet():
    """Answers are short and grounded strictly in retrieved sections, so the
    phrasing job does not need the largest model. Pinned so a future edit
    that silently upgrades it fails here first - this call runs on every
    confident answer and the cost difference is real."""
    assert app.ANSWER_MODEL == "claude-sonnet-5"
