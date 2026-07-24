import numpy as np

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
