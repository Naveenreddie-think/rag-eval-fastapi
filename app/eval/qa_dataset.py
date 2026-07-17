"""
Step 4: Hand-built evaluation set.

Responsibility: load/validate the hand-written QA pairs (target:
75-100), stored in eval/qa_pairs.json.

Each QA pair should carry a category label so eval numbers can be broken
down, not just averaged:
- "single_hop"      : answerable from one chunk
- "multi_hop"       : requires info from 2+ chunks
- "no_answer"       : no good answer exists in the corpus (tests
                        hallucination vs. honest "I don't know")

To be implemented:
- load_qa_pairs() -> list[QAPair]
- validate_qa_pairs(): sanity checks (non-empty question/answer,
  valid category label, no duplicate questions)

Reminder from plan: after writing all pairs, blind re-check a random
~10 after a day's gap to catch your own labeling errors before they
become "ground truth".
"""
