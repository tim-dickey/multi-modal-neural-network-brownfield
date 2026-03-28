---
id: "3.3"
epic: 3
title: "Measure Inference Throughput and CIFAR-100 Accuracy"
status: "ready"
priority: "high"
estimate: "M"
assignee: ""
sprint: 3
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "inference.py"
---

# Story 3.3: Measure Inference Throughput and CIFAR-100 Accuracy

## User Story

As a researcher reporting NeuralMix results,
I want to measure inference latency, throughput, and CIFAR-100 classification accuracy from a trained checkpoint,
So that I can report concrete performance numbers in the paper and README.

## Context

**Epic:** Epic 3 — Execute Full Training Run and Produce Results
**Brownfield context:** `inference.py` exists. This story adds `--benchmark` mode and CIFAR-100 evaluation to it, and writes results to a dated artifact file.

**Primary files:** `inference.py`

> ⚠️ **Dependency:** Story 3.2 must be complete (trained checkpoint must exist).

## Acceptance Criteria

**AC1 — Benchmark mode:**
**Given** a trained `best.pt` checkpoint
**When** `python inference.py --checkpoint best.pt --benchmark` is run on the RTX 3060
**Then** the script reports: samples/second throughput at batch size 1, mean latency per sample in ms, and peak VRAM usage during inference
**And** throughput is ≥ 10 samples/second (PRD §5.3 minimum)
**And** latency is ≤ 200ms per sample

**AC2 — CIFAR-100 accuracy reporting:**
**Given** a trained checkpoint and the CIFAR-100 test split
**When** `inference.py` runs classification evaluation
**Then** top-1 accuracy on CIFAR-100 test set is reported as a percentage
**And** results are written to `_bmad-output/implementation-artifacts/training-results-{date}.md` including: epoch count, final train loss, validation loss, CIFAR-100 top-1 accuracy, throughput, VRAM peak

**AC3 — Inference VRAM ceiling:**
**Given** a model loaded for inference
**When** `model.eval()` and `torch.inference_mode()` are active
**Then** peak VRAM during inference is ≤ 8GB

## Tasks

- [ ] **Task 1:** Add `--benchmark` flag to `inference.py`; implement throughput/latency measurement loop (warm-up 10 runs, measure 100 runs)
- [ ] **Task 2:** Add CIFAR-100 evaluation mode to `inference.py`
- [ ] **Task 3:** Write results to `_bmad-output/implementation-artifacts/training-results-{date}.md`
- [ ] **Task 4:** Verify `model.eval()` + `torch.inference_mode()` context used throughout inference
- [ ] **Task 5:** Write unit test verifying output dict contains all required keys; mock CUDA for CI

## Tests Required

- `tests/test_inference.py` — verify benchmark output dict keys; verify results file written; mock CUDA
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
