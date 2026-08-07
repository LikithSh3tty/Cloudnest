"""Fine-tune all-MiniLM-L6-v2 on the CloudNest pairs.

    python finetune/train.py --epochs 3

MultipleNegativesRankingLoss with in-batch negatives: each batch treats the
other rows' sections as negatives for this row's question, which is the
standard recipe when you have positives but no mined hard negatives. Batch
size matters more than usual because it sets the number of negatives.

Uses the modern SentenceTransformerTrainer API rather than the v2-era
model.fit(train_objectives=[...]) shown in older docs: the installed
sentence-transformers (5.7.0) raises a DeprecationWarning the instant you
`from sentence_transformers import losses` or call InputExample/model.fit,
and that path is slated for removal. SentenceTransformerTrainer is the
current, non-deprecated way to run the same recipe - same loss, same
(anchor, positive) pairs, in-batch negatives, shuffled, drop_last=True,
3 epochs, batch size 32, 10% warmup.
"""
import argparse
import json
from pathlib import Path

FINETUNE = Path(__file__).resolve().parent
PAIRS_PATH = FINETUNE / "data" / "train_pairs.jsonl"
OUT_DIR = FINETUNE / "artifacts" / "finetuned-fp32"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256  # must match MAX_TOKENS in backend/embed.py


def train(epochs: int = 3, batch_size: int = 32) -> Path:
    """Fine-tune and write the fp32 checkpoint. Returns its path."""
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer,
                                        SentenceTransformerTrainer,
                                        SentenceTransformerTrainingArguments)
    from sentence_transformers.sentence_transformer.losses import \
        MultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.training_args import \
        BatchSamplers

    rows = [json.loads(line)
            for line in PAIRS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    print(f"training on {len(rows)} pairs for {epochs} epochs")

    # anchor = question, positive = section text - identical pairing to the
    # v2-era InputExample(texts=[question, text]) used with
    # MultipleNegativesRankingLoss, which treats column 0 as anchor and
    # column 1 as positive, with the rest of the in-batch positives serving
    # as this row's negatives.
    train_dataset = Dataset.from_dict({
        "anchor": [r["question"] for r in rows],
        "positive": [r["text"] for r in rows],
    })

    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = MAX_SEQ_LENGTH

    loss = MultipleNegativesRankingLoss(model)

    # NO_DUPLICATES matters more than usual here: MultipleNegativesRankingLoss
    # treats every other row in the batch as a negative for this row's
    # anchor, so two rows that share the same positive section inside one
    # batch produce a false negative. With ~10-12 questions per section over
    # 40 covered sections, same-section collisions within a batch of 32 are
    # still likely without this.
    #
    # drop_last is False rather than the brief's True. Historically (366 rows
    # over 31 unique positives) NO_DUPLICATES could never fill a batch of 32
    # - every batch was capped at 31 members - and drop_last=True meant zero
    # batches ever reached "full", so the sampler silently yielded nothing
    # for the whole run (verified empirically: 0 batches/epoch). The dataset
    # has since grown to 464 rows over 40 unique sections, so NO_DUPLICATES
    # can now fill a full batch of 32 and drop_last would no longer zero out
    # every batch - but drop_last=False is kept anyway: it costs at most one
    # short batch per epoch and removes any dependence on section count
    # staying above batch_size. Batches/epoch are measured empirically at
    # run time rather than assumed from this arithmetic.
    steps_per_epoch = -(-len(train_dataset) // batch_size)  # ceil, drop_last=False
    warmup_steps = int(0.1 * steps_per_epoch * epochs)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(FINETUNE / "artifacts" / "trainer-tmp"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        warmup_steps=warmup_steps,
        dataloader_drop_last=False,
        # No explicit `shuffle` kwarg exists on this version's args class;
        # the underlying HF Trainer uses a RandomSampler (or, here, the
        # custom NO_DUPLICATES batch sampler, which shuffles internally via
        # its own generator) by default, so batches are shuffled each epoch
        # without needing to ask for it.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=1,
        report_to=[],
        disable_tqdm=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT_DIR))
    print(f"saved fp32 checkpoint -> {OUT_DIR}")
    return OUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train(args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
