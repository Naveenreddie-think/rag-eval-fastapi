"""
Step 5 pre-check: verify generation + custom faithfulness scoring work
correctly on a small sample BEFORE running the full 80-pair eval
(real API cost/time per pair -- cheaper to catch bugs now).

Includes one deliberately adversarial case: generating an answer against
retrieved context, then scoring faithfulness against WRONG context, to
confirm the scorer actually detects unsupported claims rather than
rubber-stamping everything as faithful.

Run with: python -m scripts.verify_generation_faithfulness
"""

import json

from app.retrieval.bm25 import build_bm25_index
from app.retrieval.dense import dense_search
from app.generation.generator import generate_answer
from app.eval.faithfulness import custom_faithfulness_score
from app.eval.qa_dataset import load_qa_pairs

SAMPLE_IDS = ["sh_001", "sh_005", "mh_002", "na_001"]
TOP_K = 5


def main() -> None:
    chunks = json.loads(open("data/processed/chunks.json", encoding="utf-8").read())
    build_bm25_index(chunks)

    qa_pairs = {p["id"]: p for p in load_qa_pairs()}

    for pid in SAMPLE_IDS:
        pair = qa_pairs[pid]
        print(f"\n{'=' * 70}\n{pid} [{pair['category']}]\nQ: {pair['question']}\n{'=' * 70}")

        retrieved = dense_search(pair["question"], top_k=TOP_K)
        gen = generate_answer(pair["question"], retrieved)
        print(f"\nGenerated answer:\n  {gen['answer']}")

        result = custom_faithfulness_score(gen["answer"], retrieved)
        if result["is_refusal"]:
            print("\nFaithfulness: N/A (model refused / made no claims)")
        else:
            print(f"\nFaithfulness score: {result['score']:.2f} "
                  f"({sum(c['supported'] for c in result['claims'])}/{len(result['claims'])} claims supported)")
            for c in result["claims"]:
                mark = "✓" if c["supported"] else "✗"
                print(f"  [{mark}] {c['claim']}")

    # Adversarial check: take sh_001's generated answer, but score it
    # against completely unrelated context (a no_answer-category topic)
    # to confirm the scorer actually flags unsupported claims.
    print(f"\n{'=' * 70}\nADVERSARIAL CHECK: sh_001's answer scored against WRONG context\n{'=' * 70}")
    sh_001 = qa_pairs["sh_001"]
    retrieved_correct = dense_search(sh_001["question"], top_k=TOP_K)
    gen = generate_answer(sh_001["question"], retrieved_correct)
    print(f"Answer (generated from CORRECT context): {gen['answer']}")

    wrong_context = dense_search("How do I use WebSockets in FastAPI?", top_k=TOP_K)
    result = custom_faithfulness_score(gen["answer"], wrong_context)
    print(f"\nScored against UNRELATED context -- expect a LOW score, not 1.0:")
    if result["is_refusal"]:
        print("  N/A (no claims extracted)")
    else:
        print(f"  Faithfulness score: {result['score']:.2f}")
        for c in result["claims"]:
            mark = "✓" if c["supported"] else "✗"
            print(f"  [{mark}] {c['claim']}")


if __name__ == "__main__":
    main()