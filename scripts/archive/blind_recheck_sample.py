"""
Step 4: blind re-check sample generator.

Per the plan: after writing all 80 QA pairs, blind re-check a random
~10 after a gap to catch labeling errors before they become "ground
truth" -- the same class of error caught organically in sh_014 earlier
(a genuine authoring mistake found via unrelated debugging, not this
planned step, which still hadn't been done until now).

Run with: python -m scripts.blind_recheck_sample
"""

import random

from app.eval.qa_dataset import load_qa_pairs

SAMPLE_SIZE = 10
SEED = 42  # fixed seed so the sample is reproducible, not re-rolled if run again


def main() -> None:
    qa_pairs = load_qa_pairs()
    rng = random.Random(SEED)
    sample = rng.sample(qa_pairs, SAMPLE_SIZE)

    print(f"Blind re-check sample ({SAMPLE_SIZE} pairs, seed={SEED}):\n")
    print("Answer each of these COLD, using only the actual FastAPI docs "
          "(not the stored answer below) -- then compare afterward.\n")

    for i, pair in enumerate(sample, start=1):
        print(f"{i}. [{pair['id']}] [{pair['category']}]")
        print(f"   Q: {pair['question']}\n")

    print("=" * 70)
    print("STORED ANSWERS (do not look until you've answered all above):")
    print("=" * 70)
    for i, pair in enumerate(sample, start=1):
        print(f"\n{i}. [{pair['id']}]")
        print(f"   A: {pair['answer']}")
        print(f"   Source: {pair['source_chunks']}")


if __name__ == "__main__":
    main()