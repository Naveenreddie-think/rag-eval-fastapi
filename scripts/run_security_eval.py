"""
Step 6 CLI Runner: Poisoned-Document & Indirect Prompt Injection Security Benchmark.

Run with:
    python -m scripts.run_security_eval
"""

import json
from pathlib import Path
from app.eval.poisoned_doc_test import run_poisoned_corpus_test

OUTPUT_PATH = Path("data/processed/security_eval_results.json")


def main() -> None:
    print("=" * 80)
    print("RUNNING STEP 6 SECURITY EVALUATION: POISONED DOCUMENT INJECTION")
    print("=" * 80)

    results = run_poisoned_corpus_test()

    print(f"\nTotal Attack Scenarios Tested: {results['total_scenarios']}")
    print(f"Retrieval Penetration Rate:     {results['retrieval_penetration_rate'] * 100:.1f}%")
    print(f"Attack Success Rate (ASR):      {results['attack_success_rate'] * 100:.1f}%")

    print("\n" + "-" * 80)
    for sc in results["scenarios"]:
        print(f"\nScenario [{sc['scenario_id']}] ({sc['attack_type']})")
        print(f"  Target Query:       {sc['target_query']}")
        print(f"  Penetrated Top-K:   {'YES' if sc['poison_penetrated_retrieval'] else 'NO'}")
        print(f"  Canary Surfaced:    {'YES (VULNERABLE)' if sc['canary_surfaced_in_answer'] else 'NO (RESISTED)'}")
        print(f"  Attack Successful:  {'YES ❌' if sc['attack_successful'] else 'NO (SAFE) ✅'}")
        print(f"  Generated Answer:   {sc['generated_answer'][:150]}...")
    print("\n" + "=" * 80)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved security evaluation report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
