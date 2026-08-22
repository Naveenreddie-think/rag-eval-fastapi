"""
Instruments RAGAS's internal two-step faithfulness process directly
(statement extraction, then NLI verdict per statement) for specific
disagreement cases, to see concretely WHY RAGAS scored lower than our
custom metric -- rather than treating "RAGAS disagrees" as a black box.

Run with: python -m scripts.inspect_ragas_disagreement
"""

import asyncio
import json

from app.retrieval.dense import dense_search
from app.eval.qa_dataset import load_qa_pairs
from app.eval.ragas_llm import LocalQwenRagasLLM
from ragas.metrics import faithfulness

TOP_K = 5
CASE_IDS = ["sh_014", "sh_019", "mh_019"]


async def inspect_case(pid: str, question: str, answer: str, context_texts: list[str]) -> None:
    print(f"\n{'=' * 70}\n{pid}\nQ: {question}\n{'=' * 70}")
    print(f"\nAnswer:\n  {answer}")

    row = {"question": question, "answer": answer, "contexts": context_texts}

    p_value = faithfulness._create_statements_prompt(row)
    statements_result = await faithfulness.llm.generate(p_value)
    from ragas.metrics._faithfulness import _statements_output_parser
    parsed_statements = await _statements_output_parser.aparse(
        statements_result.generations[0][0].text, p_value, faithfulness.llm, faithfulness.max_retries
    )
    if parsed_statements is None:
        print("\n  RAGAS statement extraction returned None (parse failure)")
        return
    statements = [s for item in parsed_statements.dicts() for s in item["simpler_statements"]]

    print(f"\nRAGAS-extracted statements ({len(statements)}):")
    for s in statements:
        print(f"  - {s}")

    p_value2 = faithfulness._create_nli_prompt(row, statements)
    nli_result = await faithfulness.llm.generate(p_value2)
    from ragas.metrics._faithfulness import _faithfulness_output_parser
    faith_result = await _faithfulness_output_parser.aparse(
        nli_result.generations[0][0].text, p_value2, faithfulness.llm, faithfulness.max_retries
    )
    if faith_result is None:
        print("\n  RAGAS NLI verdict step returned None (parse failure)")
        return

    print("\nRAGAS NLI verdicts:")
    for item in faith_result.dicts():
        mark = "TRUE " if item["verdict"] == 1 else "FALSE"
        print(f"  [{mark}] {item['statement']}")
        print(f"         reason: {item.get('reason', '(none given)')}")


async def main() -> None:
    faithfulness.llm = LocalQwenRagasLLM()

    generation_results = {
        r["id"]: r for r in json.loads(open("data/processed/generation_eval_results.json", encoding="utf-8").read())
    }
    qa_pairs = {p["id"]: p for p in load_qa_pairs()}

    for pid in CASE_IDS:
        question = qa_pairs[pid]["question"]
        answer = generation_results[pid]["answer"]
        retrieved = dense_search(question, top_k=TOP_K)
        context_texts = [c["text"] for c in retrieved]
        await inspect_case(pid, question, answer, context_texts)


if __name__ == "__main__":
    asyncio.run(main())