"""Which embedding model and index the app is running on.

One switch picks both halves together. Pairing them here rather than at two
call sites is deliberate: a tuned model scored against baseline vectors
produces plausible-looking cosine scores that are meaningless, and the
failure is silent. Selecting the pair atomically makes that state
unreachable from configuration.

Default is "baseline" - the frozen, calibrated model this project shipped
with. An unset or unrecognized EMBED_VARIANT resolves to it, so a typo in
deployment config degrades to known-good rather than to a broken pair.
"""
import os

BASELINE = "baseline"
FINETUNED = "finetuned"

_ARTIFACTS = {
    BASELINE: ("model_quint8_avx2.onnx", "index.npz"),
    FINETUNED: ("model_finetuned_quint8_avx2.onnx", "index_finetuned.npz"),
}

# SHA-256 of the artifacts that must never change. Guarded by
# tests/test_variant.py so a stray write to the baseline is caught by the
# suite rather than discovered after a regression in production.
BASELINE_DIGESTS = {
    "model/model_quint8_avx2.onnx": "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21",
    "model/tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "index.npz": "4379f182b0e7318a21ab352c026ec08e444c8370765ceb1a17bce51ad0d4857c",
}


def active_variant() -> str:
    """Return the configured variant, or BASELINE if unset/unrecognized."""
    requested = os.environ.get("EMBED_VARIANT", "").strip().lower()
    return requested if requested in _ARTIFACTS else BASELINE


def model_filename(variant: str | None = None) -> str:
    """ONNX filename for the variant, relative to backend/model/."""
    return _ARTIFACTS[variant or active_variant()][0]


def index_filename(variant: str | None = None) -> str:
    """Index filename for the variant, relative to backend/."""
    return _ARTIFACTS[variant or active_variant()][1]
