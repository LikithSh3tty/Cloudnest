import json

import numpy as np

from app import docs_hash, load_chunks
from build_index import build


def test_docs_hash_is_stable_and_content_sensitive(tmp_path, monkeypatch):
    import app

    first = docs_hash()
    assert first == docs_hash()  # deterministic across calls

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# Title\noriginal", encoding="utf-8")
    monkeypatch.setattr(app, "DOCS_DIR", docs)
    before = docs_hash()
    (docs / "a.md").write_text("# Title\nedited", encoding="utf-8")
    assert docs_hash() != before


def test_build_writes_aligned_index(tmp_path):
    from embed import DIM, MODEL_ID

    path = tmp_path / "index.npz"
    count = build(path)
    assert count == len(load_chunks())

    data = np.load(path, allow_pickle=False)
    meta = json.loads(data["meta"].item())
    assert data["vectors"].shape == (count, DIM)
    assert data["vectors"].dtype == np.float32
    assert len(meta) == count
    assert {"doc", "title", "text"} <= meta[0].keys()
    assert data["docs_hash"].item() == docs_hash()
    assert data["model_id"].item() == MODEL_ID


def test_index_rows_align_with_meta(tmp_path):
    """Row i of vectors must be the embedding of meta[i]['text']."""
    from embed import embed

    path = tmp_path / "index.npz"
    build(path)
    data = np.load(path, allow_pickle=False)
    meta = json.loads(data["meta"].item())
    assert np.allclose(embed([meta[3]["text"]])[0], data["vectors"][3], atol=1e-5)
