---
id: "4.3"
epic: 4
title: "Produce Mathematical Reasoning Benchmark Results"
status: "ready"
priority: "medium"
estimate: "M"
assignee: ""
sprint: 4
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "src/evaluation/math_reasoning_evaluator.py"
  - "src/evaluation/benchmark_reporter.py"
---

# Story 4.3: Produce Mathematical Reasoning Benchmark Results

## User Story

As a researcher publishing NeuralMix results,
I want to evaluate NeuralMix on a mathematical reasoning subset,
So that I can report accuracy on the GSM8K benchmark and assess the model's symbolic reasoning capability without Wolfram Alpha active.

## Context

**Epic:** Epic 4 — Produce Research Benchmark Results and Ablation Study
**Brownfield context:** No math reasoning evaluator exists yet. GSM8K evaluation is a v1 baseline (without Wolfram Alpha). Results must clearly state Wolfram is deferred to v1.5 to properly frame the paper's experimental section.

**Primary files:** `src/evaluation/math_reasoning_evaluator.py` (new), `src/evaluation/benchmark_reporter.py` (new)

> ⚠️ **Dependency:** Story 4.1 must be complete. A trained checkpoint required for integration tests.

## Acceptance Criteria

**AC1 — GSM8K evaluation:**
**Given** a trained checkpoint and the GSM8K test subset (500 problems)
**When** `MathReasoningEvaluator.evaluate(checkpoint_path, dataset="gsm8k_subset")` is called
**Then** it returns `exact_match_accuracy` and `partial_credit_accuracy` on the subset
**And** results include a baseline comparison: model accuracy vs. random baseline (PRD Table 6: baseline 30%, target 40–50%)

**AC2 — Wolfram deferred framing:**
**Given** the evaluation completes
**When** results are written to the benchmark report
**Then** the report clearly states: "Wolfram Alpha auxiliary supervision is deferred to v1.5 — these results reflect the base model without Wolfram grounding"
**And** this framing is consistent with the v1.5 scope decision documented in `epics.md`

**AC3 — Compiled benchmark report:**
**Given** the complete benchmark suite (CIFAR-100, VQA, GSM8K) has run
**When** `BenchmarkReporter.compile_results()` is called
**Then** a single `_bmad-output/implementation-artifacts/benchmark-results-{date}.md` is produced containing all benchmark tables formatted for direct inclusion in the paper's experimental section

## Tasks

- [ ] **Task 1:** Create `src/evaluation/math_reasoning_evaluator.py` with `MathReasoningEvaluator.evaluate(checkpoint_path, dataset) -> dict`; implement exact_match and partial_credit scoring
- [ ] **Task 2:** Create `src/evaluation/benchmark_reporter.py` with `BenchmarkReporter.compile_results(cifar_results, vqa_results, math_results, ablation_results) -> str`; write to dated markdown file
- [ ] **Task 3:** Add `MathReasoningEvaluator` and `BenchmarkReporter` to `src/evaluation/__init__.py`
- [ ] **Task 4:** Write unit tests covering AC1–AC3

## Tests Required

- `tests/test_evaluation.py` — verify `MathReasoningEvaluator` returns required keys; verify Wolfram framing in report; verify `BenchmarkReporter` produces markdown with all benchmark tables
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
