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


def test_tickets_requires_a_token(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    assert client.get("/api/tickets").status_code == 401


def test_tickets_rejects_a_wrong_token(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    assert client.get("/api/tickets", headers={"X-Admin-Token": "nope"}).status_code == 401


def test_tickets_disabled_when_password_unset(monkeypatch):
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    r = client.get("/api/tickets", headers={"X-Admin-Token": "anything"})
    assert r.status_code == 503


def test_tickets_returns_list_with_correct_token(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    monkeypatch.setattr(tickets_store, "list_tickets", lambda limit=100: SAMPLE)
    r = client.get("/api/tickets", headers={"X-Admin-Token": "secret"})
    assert r.status_code == 200
    assert r.json()["tickets"] == SAMPLE


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
