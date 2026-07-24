"""Local sentence embeddings via ONNX Runtime.

No API key and no torch: the int8 all-MiniLM-L6-v2 export plus onnxruntime is
~42 MB, where sentence-transformers would drag in ~800 MB of torch and blow
Vercel's 250 MB function limit.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_ID = "model_quint8_avx2.onnx"
MAX_TOKENS = 256
DIM = 384

# Built once per process, not per call: loading the session costs ~1-3s and a
# warm serverless instance should pay it only on cold start.
_session = ort.InferenceSession(
    str(MODEL_DIR / MODEL_ID), providers=["CPUExecutionProvider"]
)
_input_names = {i.name for i in _session.get_inputs()}
_tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
_tokenizer.enable_truncation(MAX_TOKENS)
_tokenizer.enable_padding()


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts. Returns (len(texts), DIM) float32 with L2-normalized rows.

    Rows are normalized here so every caller can treat a dot product as cosine
    similarity.
    """
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    encoded = _tokenizer.encode_batch(texts)
    ids = np.array([e.ids for e in encoded], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in _input_names:
        feed["token_type_ids"] = np.zeros_like(ids)
    tokens = _session.run(None, feed)[0]  # (n, sequence, DIM)
    # Mask-weighted mean pooling: the export emits per-token vectors, not a
    # sentence vector. Averaging padding in here degrades retrieval silently.
    weights = mask[..., None].astype(np.float32)
    pooled = (tokens * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
    return (pooled / np.linalg.norm(pooled, axis=1, keepdims=True)).astype(np.float32)
