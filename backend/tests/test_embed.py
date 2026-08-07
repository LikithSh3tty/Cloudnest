import os
import sys
from pathlib import Path

import numpy as np
import pytest

from embed import DIM, embed


def test_rows_are_unit_norm():
    vectors = embed(["pricing and billing", "syncing files"])
    assert vectors.shape == (2, DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_empty_input_returns_empty_matrix():
    assert embed([]).shape == (0, DIM)


def test_paraphrase_scores_above_unrelated():
    charged, pricing, sync = embed([
        "how much am I charged mid cycle",
        "pricing and billing plans",
        "my files will not sync",
    ])
    assert charged @ pricing > charged @ sync
    # zero shared tokens with either; this is the case the lexical scorer misses
    assert charged @ pricing > 0.2


def test_pooling_ignores_padding():
    """A text's embedding must not shift when the batch pads it.

    Regression guard for unmasked mean pooling, which averages padding tokens
    into the vector. Asserted as cosine rather than exact equality because the
    int8 model's attention output varies slightly with sequence length: the
    fp32 export scores 1.0000 here, int8 scores ~0.9888, and broken pooling
    scores ~0.5766.
    """
    alone = embed(["refund policy"])[0]
    padded = embed(["refund policy", "a considerably longer sentence " * 12])[0]
    assert float(alone @ padded) > 0.97


def _embed_under(variant: str, texts: list[str]) -> np.ndarray:
    """Embed texts with a specific variant's model, in a fresh module state.

    embed.py builds its ONNX session at import time from MODEL_ID =
    model_filename(), which reads EMBED_VARIANT once. Getting both variants'
    embeddings inside one test process means clearing the cached
    variant/embed modules between calls (mirrors the pattern
    finetune/eval_retrieval.py's score() uses for the same reason), and
    restoring env/module state afterward so this test cannot leak the
    finetuned model into whatever test runs next in the baseline process.
    """
    had_env = "EMBED_VARIANT" in os.environ
    prior_env = os.environ.get("EMBED_VARIANT")
    saved_modules = {name: sys.modules[name] for name in ("variant", "embed") if name in sys.modules}
    os.environ["EMBED_VARIANT"] = variant
    for name in ("variant", "embed"):
        sys.modules.pop(name, None)
    try:
        from embed import embed as variant_embed
        return variant_embed(texts)
    finally:
        for name in ("variant", "embed"):
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        if had_env:
            os.environ["EMBED_VARIANT"] = prior_env
        else:
            os.environ.pop("EMBED_VARIANT", None)


def test_finetuned_embeddings_differ_meaningfully_from_baseline():
    """Guard against the finetuned ONNX artifact silently being a copy of
    the baseline one.

    Every other assertion in this file (unit norm, paraphrase ranking,
    pooling stability) holds equally for both models, so none of them would
    catch a baseline file accidentally deployed under the finetuned
    filename - a review found exactly that gap. Measured cosine between the
    two variants' embeddings of the same text is 0.92-0.96 in practice, so a
    threshold well below that (but far above what two genuinely different
    models would coincidentally hit) is a cheap, stable tripwire: a copied
    file would score ~1.0, not ~0.94.
    """
    model_dir = Path(__file__).resolve().parent.parent / "model"
    if not (model_dir / "model_finetuned_quint8_avx2.onnx").exists():
        pytest.skip("finetuned model artifact not present")

    texts = [
        "how do I reset my password",
        "pricing and billing plans",
        "my files will not sync",
    ]
    baseline_vecs = _embed_under("baseline", texts)
    finetuned_vecs = _embed_under("finetuned", texts)

    cosines = np.sum(baseline_vecs * finetuned_vecs, axis=1)
    assert np.all(cosines < 0.999), (
        f"finetuned embeddings are near-identical to baseline (cosines={cosines.tolist()}); "
        "the finetuned ONNX artifact may be a copy of the baseline one"
    )
