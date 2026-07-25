import tickets_store


def _no_db(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)


def test_db_configured_reads_both_env_names(monkeypatch):
    _no_db(monkeypatch)
    assert tickets_store.db_configured() is False
    monkeypatch.setenv("POSTGRES_URL", "postgres://example/db")
    assert tickets_store.db_configured() is True
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgres://example/db")
    assert tickets_store.db_configured() is True


def test_save_ticket_noops_without_db(monkeypatch):
    _no_db(monkeypatch)
    assert tickets_store.save_ticket({"id": "abc"}) is False


def test_list_tickets_empty_without_db(monkeypatch):
    _no_db(monkeypatch)
    assert tickets_store.list_tickets() == []


def test_init_db_noops_without_db(monkeypatch):
    _no_db(monkeypatch)
    tickets_store.init_db()  # must return without raising
