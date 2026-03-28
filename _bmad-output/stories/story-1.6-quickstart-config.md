---
id: "1.6"
epic: 1
title: "Create Quickstart Configuration for RTX 3060 First Run"
status: "ready"
priority: "medium"
estimate: "S"
assignee: ""
sprint: 1
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "configs/quickstart.yaml"
  - "configs/default.yaml"
---

# Story 1.6: Create Quickstart Configuration for RTX 3060 First Run

## User Story

As an independent AI developer new to NeuralMix,
I want a pre-tuned quickstart configuration file,
So that I can run my first training experiment with safe, validated settings without needing to understand every configuration option.

## Context

**Epic:** Epic 1 — Achieve Consumer GPU Training Target
**Brownfield context:** `configs/default.yaml` exists. `configs/quickstart.yaml` does not exist yet. This story creates it and annotates both files with `# BEGINNER` / `# ADVANCED` comment tiers.

**Primary files:** `configs/quickstart.yaml` (new), `configs/default.yaml` (annotations only)

## Acceptance Criteria

**AC1 — quickstart.yaml content:**
**Given** the NeuralMix repository is cloned
**When** a user opens `configs/quickstart.yaml`
**Then** the file exists and contains a complete, runnable configuration with:
- `mixed_precision: bf16`
- `gradient_checkpointing: true`
- `micro_batch_size: 4`
- `gradient_accumulation: 8`
- `double_loop.enabled: false`
- `wolfram.enabled: false`
- head type set to `classification` with 100 classes (CIFAR-100 compatible)
**And** every configurable section includes a comment tier annotation: `# BEGINNER: safe to change` or `# ADVANCED: change only if you understand the architecture`

**AC2 — quickstart.yaml is runnable:**
**Given** `python train.py --config configs/quickstart.yaml` is executed
**When** the training loop starts
**Then** training runs without error on the quickstart config using the synthetic/toy dataset path
**And** the startup banner confirms all quickstart settings are active

**AC3 — default.yaml annotations:**
**Given** `default.yaml` is opened by an advanced user
**When** they read the file
**Then** sections for `double_loop`, `wolfram`, `hardware.max_memory`, and attention mechanism settings are annotated with `# ADVANCED` comments
**And** `micro_batch_size`, `max_epochs`, and `data.datasets` are annotated with `# BEGINNER` comments

## Tasks

- [ ] **Task 1:** Create `configs/quickstart.yaml` with all required settings and `# BEGINNER` / `# ADVANCED` annotations
- [ ] **Task 2:** Add `# BEGINNER` / `# ADVANCED` annotations to `configs/default.yaml`
- [ ] **Task 3:** Write/update unit test verifying `quickstart.yaml` loads and parses without error and all required keys are present with correct values

## Tests Required

- `tests/test_config.py` (or equivalent) — load `quickstart.yaml`; assert all required keys present; assert `double_loop.enabled == false`; assert `wolfram.enabled == false`; assert `micro_batch_size == 4`
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
