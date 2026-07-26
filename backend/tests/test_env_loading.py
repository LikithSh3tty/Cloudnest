import os

import app


def test_strips_surrounding_quotes(tmp_path, monkeypatch):
    # `vercel env pull` writes KEY="value"; unstripped quotes silently corrupt
    # a connection string into something that can never connect.
    env = tmp_path / ".env"
    env.write_text('DB_TEST_URL="postgres://user:pw@host/db"\n', encoding="utf-8")
    monkeypatch.delenv("DB_TEST_URL", raising=False)
    app._load_env_file(env)
    assert os.environ["DB_TEST_URL"] == "postgres://user:pw@host/db"


def test_skips_comments_and_blank_lines(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# a comment\n\nKEPT=yes\n", encoding="utf-8")
    monkeypatch.delenv("KEPT", raising=False)
    app._load_env_file(env)
    assert os.environ["KEPT"] == "yes"
    assert "# a comment" not in os.environ


def test_does_not_override_the_real_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("ALREADY_SET=from_file\n", encoding="utf-8")
    monkeypatch.setenv("ALREADY_SET", "from_shell")
    app._load_env_file(env)
    assert os.environ["ALREADY_SET"] == "from_shell"


def test_missing_file_is_not_an_error(tmp_path):
    app._load_env_file(tmp_path / "nope.env")  # must return without raising


def test_env_local_is_read_and_wins_over_env():
    # `vercel env pull` targets .env.local, so it must be read at all - and it
    # is the more specific file, so it takes precedence (loading it first wins
    # under setdefault).
    names = [p.name for p in app.ENV_FILES]
    assert ".env.local" in names
    assert ".env" in names
    assert names.index(".env.local") < names.index(".env")
