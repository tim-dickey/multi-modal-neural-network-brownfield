---
id: "4.1"
epic: 4
title: "Implement CIFAR-100 and VQA Evaluation Framework"
status: "ready"
priority: "high"
estimate: "M"
assignee: ""
sprint: 4
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "src/evaluation/__init__.py"
  - "src/evaluation/cifar100_evaluator.py"
  - "src/evaluation/vqa_evaluator.py"
  - "src/evaluation/ablation_runner.py"
---

# Story 4.1: Implement CIFAR-100 and VQA Evaluation Framework

## User Story

As a researcher publishing NeuralMix results,
I want a reusable evaluation framework for CIFAR-100 and VQA benchmarks,
So that I can produce standardized, reproducible accuracy numbers for the paper.

## Context

**Epic:** Epic 4 — Produce Research Benchmark Results and Ablation Study
**Brownfield context:** `src/evaluation/` is currently empty. This story builds the complete evaluation module — `CIFAR100Evaluator`, `VQAEvaluator`, and `AblationRunner` — all importable from `src/evaluation/__init__.py`. This is the gating prerequisite for paper writing.

**Primary files:** `src/evaluation/` (all new)

> ⚠️ **Dependency:** Epic 1 and Epic 2 must be complete. A trained checkpoint (from Story 3.2) is needed for integration tests.

## Acceptance Criteria

**AC1 — CIFAR-100 evaluator:**
**Given** a trained checkpoint and CIFAR-100 test split
**When** `CIFAR100Evaluator.evaluate(checkpoint_path)` is called
**Then** it returns a dict with `top1_accuracy`, `top5_accuracy`, `eval_samples`, `eval_time_seconds`
**And** evaluation runs without modifying model weights
**And** results are deterministic across runs with the same checkpoint and test split

**AC2 — VQA evaluator:**
**Given** a trained checkpoint and VQA v2 validation split (or OK-VQA subset)
**When** `VQAEvaluator.evaluate(checkpoint_path, split="val")` is called
**Then** it returns `vqa_accuracy` (VQA evaluation metric — answer presence in ground truth list)
**And** the evaluator handles multi-answer ground truth (VQA has 10 annotators per question)

**AC3 — Module importable:**
**Given** `src/evaluation/__init__.py`
**When** it is imported
**Then** `CIFAR100Evaluator`, `VQAEvaluator`, and `AblationRunner` are all importable without error
**And** the module is no longer empty

## Tasks

- [ ] **Task 1:** Create `src/evaluation/cifar100_evaluator.py` with `CIFAR100Evaluator` class implementing `evaluate(checkpoint_path) -> dict`
- [ ] **Task 2:** Create `src/evaluation/vqa_evaluator.py` with `VQAEvaluator` class implementing `evaluate(checkpoint_path, split) -> dict`
- [ ] **Task 3:** Create stub `src/evaluation/ablation_runner.py` with `AblationRunner` class (full implementation in Story 4.2)
- [ ] **Task 4:** Update `src/evaluation/__init__.py` to export all three classes
- [ ] **Task 5:** Write unit tests covering AC1–AC3 with mock checkpoints

## Tests Required

- `tests/test_evaluation.py` — verify all three classes importable; verify `CIFAR100Evaluator.evaluate()` returns required keys; verify deterministic; verify `VQAEvaluator` handles multi-answer GT; mock checkpoint loading
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
