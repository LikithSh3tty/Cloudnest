import app


def test_answer_model_is_a_named_constant():
    """The model id belongs in one named place, not inline in the API call,
    so swapping it is a one-line change with a test that notices."""
    assert isinstance(app.ANSWER_MODEL, str)
    assert app.ANSWER_MODEL.startswith("claude-")


def test_answer_model_is_sonnet():
    """Answers are short and grounded strictly in retrieved sections, so the
    phrasing job does not need the largest model. Pinned so a future edit
    that silently upgrades it fails here first - this call runs on every
    confident answer and the cost difference is real."""
    assert app.ANSWER_MODEL == "claude-sonnet-5"
