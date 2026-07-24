"""Build the embedding index. Re-run after editing cloudnest_docs/.

    python backend/build_index.py
    python backend/build_index.py --calibrate
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import INDEX_PATH, docs_hash, load_chunks
from embed import MODEL_ID, embed


def build(path: Path = INDEX_PATH) -> int:
    """Embed every chunk and write the index. Returns the chunk count."""
    chunks = load_chunks()
    # one call per chunk, not one batched call: query-time embed() is always a
    # batch of one, and this int8 model's output shifts with batch padding
    vectors = np.array([embed([c["text"]])[0] for c in chunks])
    np.savez(
        path,
        vectors=vectors,
        meta=json.dumps(chunks),
        docs_hash=docs_hash(),
        model_id=MODEL_ID,
    )
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true",
                        help="report the similarity gap and suggest a threshold")
    args = parser.parse_args()
    count = build()
    print(f"indexed {count} chunks -> {INDEX_PATH}")
    if args.calibrate:
        from calibrate import report  # added in Task 4
        report()


if __name__ == "__main__":
    main()
