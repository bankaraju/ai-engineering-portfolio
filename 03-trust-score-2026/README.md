# 2026 — Trust Score: can management be held to what it said?

## Problem

Indian listed companies put forward guidance in their annual reports and earnings calls: capacity
targets, launch dates, margin ranges, capex plans. A year or three later the same company reports
what actually happened. The question here is mechanical: take each specific, measurable statement
of intent, find the later reported figure, and record whether it was delivered, partially
delivered, missed, quietly dropped, too early to judge, or unverifiable. Aggregated over several
years this gives a management "Trust Score" and a letter grade.

## Pipeline (as the code in this extract supports it)

1. **Retrieval** (not included). The original project downloaded up to five annual-report PDFs
   per company, earnings-call transcripts and a cached multi-year financials JSON into a local
   `artifacts/<SYMBOL>/` folder. Those retrieval and scraping modules are excluded here;
   `run_trust_score.py` only reads files already present in that folder.
2. **Claim extraction** (`run_trust_score.extract_narratives`). Annual-report text is extracted
   with PyMuPDF (`extract_pdf_text`, section detection for MD&A / Chairman / Directors' report),
   and an LLM is prompted (`NARRATIVE_EXTRACTION_PROMPT`) to return JSON guidance items, each with
   an exact quote, page reference, metric, target value, target year, category and materiality,
   plus an `actuals_reported` block for the year.
3. **Deterministic filter** (`_filter_vague_guidance`). Drops items with no number and no concrete
   deliverable, aspirational language ("we remain confident"), macro/government forecasts and
   duplicate net-zero pledges.
4. **Promise scoring** (`score_promises`). All guidance items plus the multi-year financials (and
   concall text when present) go to the LLM with `SCORING_PROMPT_TEMPLATE`, which asks for a
   scorecard row per item: actual value, variance %, status and the source of the actual.
   `_extract_json_payload` and `_clean_json_expressions` repair common reply defects (markdown
   fences, trailing prose, arithmetic written inside JSON such as `(5.51-6.1)/6.1*100`).
5. **Grade** (`calculate_trust_score`). Effective rate = 70% critical-item delivery rate + 30%
   overall rate, mapped to A+ .. F; fewer than five scoreable items gives `INSUFFICIENT_DATA`.
   Dropped themes and credibility trajectory feed a separate premium adjustment.
6. **Golden set** (`golden_set/`). A vetted MARUTI scorecard built from public annual-report
   statements. The compact version is injected into every scoring prompt as a worked example; the
   full version is a reference.
7. **Quality gates** (`pipeline/quality_gates/validator.py`). Seven gates over a research context
   (source mapping, data gaps, ratios computed, narrative conflict surfaced, score present); gates
   4 and 6 were never implemented and always fail.

`pipeline/narrative/claim_extractor.py` and `promise_scorer.py` are the original module
scaffolds: the dataclasses (`ManagementClaim`, `PromiseResult`) and function signatures are
defined, but the bodies are TODOs that return empty lists. The working logic lives in
`run_trust_score.py`.

## LLM provider history (from the commit log)

- Initially Claude Sonnet; a full pass over 213 companies was too expensive.
- 10 Apr 2026: switched the default to Gemini 2.5 Flash.
- 11 Apr 2026: moved to OpenAI `gpt-5.4-mini` (fallback `gpt-5.4-nano`) after unexpected Gemini
  billing; OpenRouter models were tried and not adopted.

The provider is chosen by `ANKA_LLM_PROVIDER`; model ids are read from the environment
(`.env.example`), with placeholder defaults in this extract.

## Why this matters for the next step

The LLM was reliable at the first half of the job: finding forward-looking statements and quoting
them with a target. It was not reliable on the numbers side. `calculate_trust_score` takes the
delivery rates from the model's own `summary` block rather than recounting the scorecard rows,
and the golden MARUTI file itself shows the risk: its summary says 7 partially delivered and 6
unverifiable while its rows contain 8 and 5 (pinned in
`tests/test_trust_score_logic.py::test_grade_is_taken_from_model_summary_not_rows`).
That lesson became the governed approach in https://github.com/bankaraju/anka-governed-research,
where arithmetic is computed deterministically and a later rule states that a score's grade is
computed from its rows, never taken from the model's own summary.

## Files

| Path | Contents |
|---|---|
| `run_trust_score.py` | Working pipeline: PDF text, extraction prompt, vague-guidance filter, scoring prompt, JSON repair, grade |
| `pipeline/narrative/claim_extractor.py` | `ManagementClaim` dataclass + extraction scaffold (TODO body) |
| `pipeline/narrative/promise_scorer.py` | `PromiseResult` dataclass + scoring scaffold (TODO body) |
| `pipeline/quality_gates/validator.py` | Seven quality gates |
| `golden_set/MARUTI_gold_scorecard.json` | Full vetted scorecard (13 rows) |
| `golden_set/MARUTI_exemplar_compact.json` | Compact exemplar injected into scoring prompts (10 rows) |
| `docs/SKILL_trust_score.md` | Method description used by the project |
| `tests/` | Offline tests (written for this extract; the original repo had none for these modules) |
| `.env.example`, `requirements.txt` | Configuration template and dependencies |

## Run the tests

```
pip install -r requirements.txt
python -m pytest -v tests
```

Output:

```
collecting ... collected 20 items

tests/test_quality_gates.py::test_seven_gates_in_order PASSED            [  5%]
tests/test_quality_gates.py::test_gate_outcomes_full_context PASSED      [ 10%]
tests/test_quality_gates.py::test_gate_failures_on_thin_context PASSED   [ 15%]
tests/test_scaffold_modules.py::test_management_claim_fields PASSED      [ 20%]
tests/test_scaffold_modules.py::test_extract_from_empty_text_returns_nothing PASSED [ 25%]
tests/test_scaffold_modules.py::test_extract_claims_scaffold_returns_empty_list PASSED [ 30%]
tests/test_scaffold_modules.py::test_score_promises_scaffold PASSED      [ 35%]
tests/test_scaffold_modules.py::test_promise_result_fields PASSED        [ 40%]
tests/test_trust_score_logic.py::test_extract_json_payload_strips_fences_and_trailing_prose PASSED [ 45%]
tests/test_trust_score_logic.py::test_extract_json_payload_leading_prose PASSED [ 50%]
tests/test_trust_score_logic.py::test_clean_json_expressions_evaluates_math PASSED [ 55%]
tests/test_trust_score_logic.py::test_filter_vague_guidance PASSED       [ 60%]
tests/test_trust_score_logic.py::test_golden_set_rows_are_well_formed PASSED [ 65%]
tests/test_trust_score_logic.py::test_golden_exemplar_is_injected_into_scoring_prompt PASSED [ 70%]
tests/test_trust_score_logic.py::test_insufficient_data_below_five_scoreable PASSED [ 75%]
tests/test_trust_score_logic.py::test_grade_is_70_30_blend_of_critical_and_overall_rate PASSED [ 80%]
tests/test_trust_score_logic.py::test_grade_is_taken_from_model_summary_not_rows PASSED [ 85%]
tests/test_trust_score_logic.py::test_score_promises_with_mocked_llm PASSED [ 90%]
tests/test_trust_score_logic.py::test_fenced_reply_with_math_expression XFAIL [ 95%]
tests/test_trust_score_logic.py::test_live_llm_roundtrip SKIPPED (LL...) [100%]

=================== 18 passed, 1 skipped, 1 xfailed in 0.46s ===================
```

The skipped test calls a real provider and runs only with `RUN_LLM_TESTS=1` and a key. The
xfail records a real gap in the April code: a fenced reply that also contains an arithmetic
expression is not parsed.

## Status

Preserved extract of the April 2026 OPUS ANKA research code. Not maintained. Edits from the
original: model ids and a local notes path moved to environment variables, a personal `.env`
path removed, the symbol-resolution import (retrieval code) replaced by using the symbol as given,
line endings normalised; tests, `.env.example` and `requirements.txt` added.
