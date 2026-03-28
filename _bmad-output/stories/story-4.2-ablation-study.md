---
id: "4.2"
epic: 4
title: "Run Double-Loop Ablation Study"
status: "ready"
priority: "high"
estimate: "M"
assignee: ""
sprint: 4
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "src/evaluation/ablation_runner.py"
---

# Story 4.2: Run Double-Loop Ablation Study

## User Story

As a researcher publishing NeuralMix results,
I want an automated ablation study comparing training with and without the double-loop controller,
So that I can quantify the meta-learning contribution and report the improvement delta in the paper.

## Context

**Epic:** Epic 4 — Produce Research Benchmark Results and Ablation Study
**Brownfield context:** `AblationRunner` stub created in Story 4.1. This story implements the full two-condition comparison (with/without double-loop) and the report writer.

**Primary files:** `src/evaluation/ablation_runner.py`

> ⚠️ **Dependency:** Story 4.1 must be complete. Epic 2 (double-loop wired) must be complete.

## Acceptance Criteria

**AC1 — Two-condition training runs:**
**Given** `AblationRunner` is initialized with a base config
**When** `AblationRunner.run(conditions=["with_double_loop", "without_double_loop"], epochs=10, seed=42)` is called
**Then** two training runs are executed sequentially: one with `double_loop.enabled: true` and one with `double_loop.enabled: false`
**And** both runs use identical hyperparameters, random seeds, data order, and hardware settings
**And** both runs produce checkpoints in separate output directories

**AC2 — Report generation:**
**Given** both ablation runs complete
**When** `AblationRunner.report()` is called
**Then** it returns a dict containing: CIFAR-100 accuracy for each condition, final validation loss for each condition, accuracy delta (with_double_loop - without_double_loop), and overhead percentage for the double-loop condition
**And** the report is written to `_bmad-output/implementation-artifacts/ablation-results-{date}.md`

**AC3 — PRD target evaluation:**
**Given** the ablation results
**When** `accuracy_delta >= 0.05` (≥5% improvement)
**Then** the report includes `[double-loop] ✓ Meets PRD improvement target of 5–10%`
**When** `accuracy_delta < 0.05`
**Then** the report includes `[double-loop] ⚠ Below PRD improvement target — results recorded for paper`

## Tasks

- [ ] **Task 1:** Implement `AblationRunner.run()` — sequentially execute two Trainer runs with modified config; use `seed=42` for both; save to separate checkpoint dirs
- [ ] **Task 2:** Implement `AblationRunner.report()` — collect CIFAR-100 accuracy from both runs via `CIFAR100Evaluator`; compute delta and overhead; write markdown report
- [ ] **Task 3:** Add PRD target comparison logic and appropriate log/report messages
- [ ] **Task 4:** Write unit tests covering AC1–AC3 with mocked Trainer and CIFAR100Evaluator

## Tests Required

- `tests/test_ablation_runner.py` — mock Trainer; verify two runs with correct config modifications; verify seed applied; verify report dict keys; verify PRD target message logic
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
