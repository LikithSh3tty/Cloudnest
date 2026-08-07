"""Generate synthetic (question, section) training pairs from the docs.

    python finetune/make_pairs.py --per-chunk 10

Questions are written by Claude from each section's text, in the register a
real support user would use rather than the section's own vocabulary - the
whole point of fine-tuning here is to close the gap between how the docs are
written and how people ask.

This data is circular by construction: a question written from a chunk,
trained to retrieve that chunk, teaches the generator's phrasing. It is
training data only. The honest measurement is the hand-labeled held-out set
in data/eval_questions.jsonl, and Step 5 asserts no overlap.

Resumable: re-running this command skips any section that already has
rows in data/train_pairs.jsonl and appends only what's missing, so an
interrupted run (API credit, network, etc.) can be topped up without
re-spending on coverage that already exists. Bare title-only chunks
(docs with no intro paragraph before their first subsection) are always
skipped - there is no content in them to ground a question on.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
OUT_PATH = Path(__file__).resolve().parent / "data" / "train_pairs.jsonl"

sys.path.insert(0, str(BACKEND))
from app import load_chunks  # noqa: E402

PROMPT = """Below is one section of the support documentation for CloudNest,
a cloud storage product.

Write {n} distinct questions a customer might send to support that this
section answers.

Rules:
- Write how a frustrated or hurried customer writes, not how documentation
  is written. Lowercase, no punctuation, and typos are all fine.
- At least half must avoid the section's own distinctive nouns entirely.
  Describe the symptom or the goal instead of naming the feature.
- Vary the length. Some three words, some a full sentence.
- No question may mention "documentation", "section", or "the docs".
- Output one question per line. No numbering, no preamble, no blank lines.

Section title: {title}

Section text:
{text}"""


def generate_questions(chunk: dict, n: int = 10) -> list[str]:
    """Ask Claude for n user-style questions answered by this chunk."""
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": PROMPT.format(n=n, title=chunk["title"], text=chunk["text"]),
        }],
    )
    # claude-sonnet-5 can prepend a ThinkingBlock before the TextBlock, so
    # content[0] is not reliably the text - same idiom as backend/app.py's
    # llm_answer: join every block whose type is "text", skip the rest.
    text = "".join(b.text for b in message.content if b.type == "text")
    lines = text.strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def _is_bare_chunk(chunk: dict) -> bool:
    """True if a chunk's text is nothing but its own H1/H2 title line.

    load_chunks() produces these for docs with no intro paragraph before
    their first subsection - there is no content to write a grounded
    question from, so nothing should be generated for them. Forcing
    questions onto them produced mislabeled training data once already
    (see finetune/README.md); the fix is to never generate for them.
    """
    return chunk["text"].count("\n") == 0


def build_pairs(n_per_chunk: int = 10, out_path: Path | None = None) -> list[dict]:
    """Generate pairs for every content-bearing chunk not already covered.

    Resumable by design: if out_path already has rows, any chunk with at
    least one existing row is skipped, and new pairs are appended rather
    than overwriting the file. This lets a run interrupted partway
    (the API credit wall that hit mid-corpus previously) be topped up
    without re-spending on coverage that already exists or risking
    duplicate rows. Bare title-only chunks (see _is_bare_chunk) are always
    skipped regardless of coverage - there is nothing in them to ground a
    question on.

    When out_path is given, each newly-generated chunk's pairs are
    appended and flushed to disk immediately after that chunk finishes,
    so a crash partway through leaves the file valid up to that point
    instead of losing the whole run, and progress is visible to anyone
    tailing the file or the log while the run is still going.

    Returns the full set of pairs: pre-existing rows loaded from out_path
    plus any newly generated in this call.
    """
    chunks = load_chunks()

    existing: list[dict] = []
    covered: set[tuple[str, str]] = set()
    if out_path is not None and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing.append(row)
                covered.add((row["doc"], row["title"]))

    pairs = list(existing)
    handle = out_path.open("a", encoding="utf-8") if out_path is not None else None
    try:
        for i, chunk in enumerate(chunks, 1):
            key = (chunk["doc"], chunk["title"])
            if key in covered:
                print(f"[{i}/{len(chunks)}] {chunk['title']}: already covered, skipping", flush=True)
                continue
            if _is_bare_chunk(chunk):
                print(f"[{i}/{len(chunks)}] {chunk['title']}: bare title chunk, no content, skipping", flush=True)
                continue

            questions = generate_questions(chunk, n_per_chunk)
            print(f"[{i}/{len(chunks)}] {chunk['title']}: {len(questions)} questions", flush=True)
            for question in questions:
                pair = {
                    "question": question,
                    "doc": chunk["doc"],
                    "title": chunk["title"],
                    "text": chunk["text"],
                }
                pairs.append(pair)
                if handle is not None:
                    handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
            if handle is not None:
                handle.flush()
    finally:
        if handle is not None:
            handle.close()
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-chunk", type=int, default=10)
    args = parser.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(args.per_chunk, out_path=OUT_PATH)
    print(f"wrote {len(pairs)} total pairs -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
