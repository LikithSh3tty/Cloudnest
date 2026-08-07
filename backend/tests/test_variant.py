import hashlib
from pathlib import Path

import pytest

import variant

BACKEND = Path(__file__).resolve().parent.parent


def test_defaults_to_baseline_when_unset(monkeypatch):
    monkeypatch.delenv("EMBED_VARIANT", raising=False)
    assert variant.active_variant() == "baseline"
    assert variant.model_filename() == "model_quint8_avx2.onnx"
    assert variant.index_filename() == "index.npz"


def test_finetuned_selects_both_artifacts(monkeypatch):
    monkeypatch.setenv("EMBED_VARIANT", "finetuned")
    assert variant.active_variant() == "finetuned"
    assert variant.model_filename() == "model_finetuned_quint8_avx2.onnx"
    assert variant.index_filename() == "index_finetuned.npz"


def test_unknown_variant_falls_back_to_baseline(monkeypatch):
    """A typo in deployment config must degrade to the known-good pair,
    never to a half-configured state that pairs one variant's model with
    the other's index."""
    monkeypatch.setenv("EMBED_VARIANT", "fintuned")
    assert variant.active_variant() == "baseline"
    assert variant.model_filename() == "model_quint8_avx2.onnx"
    assert variant.index_filename() == "index.npz"


def test_variant_is_case_and_whitespace_tolerant(monkeypatch):
    monkeypatch.setenv("EMBED_VARIANT", "  FineTuned ")
    assert variant.active_variant() == "finetuned"


@pytest.mark.parametrize("relative_path", sorted(variant.BASELINE_DIGESTS))
def test_baseline_artifacts_are_never_modified(relative_path):
    """Rollback layer 2: proves the frozen baseline is identical to the
    SHA-256 digests recorded before fine-tuning work started, not merely to
    what happens to be on disk now. If this fails, the rollback story is
    broken and the tuned work must not be trusted."""
    path = BACKEND / relative_path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == variant.BASELINE_DIGESTS[relative_path], (
        f"{relative_path} changed; baseline artifacts are immutable"
    )
