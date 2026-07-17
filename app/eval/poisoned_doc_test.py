"""
Step 6 (security tie-in, optional but planned): poisoned-document test.

Responsibility: inject a document containing a hidden/embedded
instruction into the corpus (same pattern as Project 1's
vendor_onboarding_guide.txt), re-run retrieval + generation, and measure
whether the injected instruction gets surfaced/obeyed by the generation
step -- i.e. whether malicious content sitting in the retrieval corpus
itself can manipulate the RAG system's output.

To be implemented:
- craft_poisoned_doc(): build the bait document
- run_poisoned_corpus_test(query_set): re-run eval queries against the
  poisoned corpus, log whether/how the injected instruction surfaces
- Ties directly back to Project 1's attack taxonomy and FINDINGS.md
  methodology -- reuse that structure rather than inventing a new one.
"""
