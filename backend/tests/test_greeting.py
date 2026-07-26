"""A hello should get a hello, not the out-of-scope guard."""
import app


def _clarify(text, confidence):
    state = {"messages": [{"role": "user", "content": text}], "confidence": confidence}
    return app.clarify(state)["messages"][-1]["content"]


def test_plain_greetings_are_recognised():
    for text in ["hi", "Hello", "hey there", "good morning", "hello!",
                 "hi, how are you?", "hello my name is likith"]:
        assert app._is_greeting(text) is True, text


def test_a_greeting_carrying_a_real_question_is_not_just_a_greeting():
    # These must reach the normal path - answering beats saying hello back.
    for text in ["hi, my sync is broken", "hello, how do I cancel my plan",
                 "hey there, the app keeps crashing on upload"]:
        assert app._is_greeting(text) is False, text


def test_off_topic_noise_is_not_a_greeting():
    for text in ["cat mouse banana", "what is the capital of France",
                 "write me a poem"]:
        assert app._is_greeting(text) is False, text


def test_greeting_gets_the_welcome_not_the_out_of_scope_guard():
    reply = _clarify("hello my name is likith", 0.10)
    assert reply == app.GREETING_MSG
    assert reply != app.CLARIFY_OFFTOPIC_MSG


def test_real_off_topic_still_gets_the_out_of_scope_guard():
    assert _clarify("cat mouse banana", 0.10) == app.CLARIFY_OFFTOPIC_MSG


def test_borderline_clarify_is_untouched_by_greeting_handling():
    # Above the floor the borderline prompt still wins, greeting or not.
    assert _clarify("hi", 0.28) == app.CLARIFY_BORDERLINE_MSG


def test_a_greeting_is_not_folded_into_the_next_question():
    """A greeting carries no topic, so folding it in only dilutes the query.

    "what are the plans" scores 0.397 alone but 0.256 as "hello my name is
    likith. what are the plans" - under the off-topic floor, so an in-scope
    question got the out-of-scope guard.
    """
    messages = [
        {"role": "user", "content": "hello my name is likith"},
        {"role": "assistant", "content": app.GREETING_MSG},
        {"role": "user", "content": "what are the plans"},
    ]
    assert app.contextualize_query(messages) == "what are the plans"


def test_a_real_previous_turn_is_still_folded_in():
    messages = [
        {"role": "user", "content": "I want to add two-factor authentication"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "does it cost extra"},
    ]
    folded = app.contextualize_query(messages)
    assert folded.startswith("I want to add two-factor authentication")
    assert folded.endswith("does it cost extra")


def test_it_looks_past_a_greeting_to_the_last_real_turn():
    messages = [
        {"role": "user", "content": "I want to add two-factor authentication"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": app.GREETING_MSG},
        {"role": "user", "content": "does it cost extra"},
    ]
    assert app.contextualize_query(messages).startswith(
        "I want to add two-factor authentication")


def test_question_after_a_greeting_clears_the_threshold():
    messages = [
        {"role": "user", "content": "hello my name is likith"},
        {"role": "assistant", "content": app.GREETING_MSG},
        {"role": "user", "content": "what are the plans"},
    ]
    confidence = app.semantic_retrieve(app.contextualize_query(messages))["confidence"]
    assert confidence >= app.CONFIDENCE_THRESHOLD, confidence


def test_greeting_never_escalates():
    out = app.clarify({"messages": [{"role": "user", "content": "hello"}], "confidence": 0.1})
    assert out["escalate"] is False
    assert out["ticket"] is None
    assert out["clarified"] is True
