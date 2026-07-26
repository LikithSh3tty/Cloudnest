import app
import index
import tickets_store
from fastapi.testclient import TestClient

client = TestClient(index.app)

SAMPLE = [{
    "id": "t1",
    "created_at": "2026-07-26T00:00:00+00:00",
    "question": "get me a person",
    "category": "general",
    "confidence": 0.11,
    "reason": "user_requested",
    "conversation": [{"role": "user", "content": "get me a person"}],
}]


AUTH = {"X-Admin-User": "admin", "X-Admin-Token": "secret"}


def test_tickets_requires_a_token(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    assert client.get("/api/tickets").status_code == 401


def test_tickets_rejects_a_wrong_token(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    r = client.get("/api/tickets", headers={**AUTH, "X-Admin-Token": "nope"})
    assert r.status_code == 401


def test_tickets_rejects_a_wrong_username(monkeypatch):
    # The right password under the wrong name is still not the admin.
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    r = client.get("/api/tickets", headers={**AUTH, "X-Admin-User": "someone"})
    assert r.status_code == 401


def test_tickets_requires_a_username(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    assert client.get("/api/tickets", headers={"X-Admin-Token": "secret"}).status_code == 401


def test_tickets_disabled_when_password_unset(monkeypatch):
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    r = client.get("/api/tickets", headers=AUTH)
    assert r.status_code == 503


def test_tickets_username_defaults_to_admin(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    monkeypatch.delenv("ADMIN_USERNAME", raising=False)
    monkeypatch.setattr(tickets_store, "list_tickets", lambda limit=100: SAMPLE)
    assert client.get("/api/tickets", headers=AUTH).status_code == 200


def test_tickets_honours_a_custom_username(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    monkeypatch.setenv("ADMIN_USERNAME", "root")
    monkeypatch.setattr(tickets_store, "list_tickets", lambda limit=100: SAMPLE)
    assert client.get("/api/tickets", headers={**AUTH, "X-Admin-User": "root"}).status_code == 200
    # the default no longer opens it
    assert client.get("/api/tickets", headers=AUTH).status_code == 401


def test_tickets_returns_list_with_correct_credentials(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    monkeypatch.setattr(tickets_store, "list_tickets", lambda limit=100: SAMPLE)
    r = client.get("/api/tickets", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["tickets"] == SAMPLE


ESCALATING = {
    "message": "no, connect me to a person",
    "history": [
        {"role": "user", "content": "get me a human"},
        {"role": "assistant", "content": "PLACEHOLDER"},
    ],
}


def _escalating_body(**extra):
    history = [dict(m) for m in ESCALATING["history"]]
    history[1]["content"] = app.DEFLECT_MSG
    return {"message": ESCALATING["message"], "history": history, **extra}


def test_chat_attaches_the_signed_in_user_to_the_ticket(monkeypatch):
    saved = {}
    monkeypatch.setattr(tickets_store, "save_ticket", lambda t: saved.update(t) or True)
    r = client.post("/api/chat", json=_escalating_body(
        username="alice", signed_in_at="2026-07-26T09:00:00+00:00"))
    assert r.status_code == 200
    ticket = r.json()["ticket"]
    assert ticket["username"] == "alice"
    assert ticket["signed_in_at"] == "2026-07-26T09:00:00+00:00"
    # and the enriched ticket - not the graph's bare one - is what gets stored
    assert saved["username"] == "alice"
    assert saved["signed_in_at"] == "2026-07-26T09:00:00+00:00"


def test_chat_works_without_a_signed_in_user(monkeypatch):
    # The graph and the store must not require the frontend to send a user.
    saved = {}
    monkeypatch.setattr(tickets_store, "save_ticket", lambda t: saved.update(t) or True)
    r = client.post("/api/chat", json=_escalating_body())
    assert r.status_code == 200
    assert r.json()["ticket"]["username"] is None
    assert saved["signed_in_at"] is None


def test_chat_still_succeeds_when_ticket_write_raises(monkeypatch):
    def boom(ticket):
        raise RuntimeError("db down")
    monkeypatch.setattr(tickets_store, "save_ticket", boom)
    # An explicit request after a prior deflection escalates -> save_ticket runs.
    history = [
        {"role": "user", "content": "get me a human"},
        {"role": "assistant", "content": app.DEFLECT_MSG},
    ]
    r = client.post("/api/chat", json={"message": "no, connect me to a person",
                                       "history": history})
    assert r.status_code == 200
    body = r.json()
    assert body["escalated"] is True
    assert "answer" in body
