---
id: "5.2"
epic: 5
title: "Build Training Notebook (02_training.ipynb)"
status: "ready"
priority: "medium"
estimate: "M"
assignee: ""
sprint: 5
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "notebooks/02_training.ipynb"
---

# Story 5.2: Build Training Notebook (02_training.ipynb)

## User Story

As an independent AI developer,
I want a training walkthrough notebook,
So that I can understand how to configure and run a training session, observe real VRAM graphs, and interpret training metrics.

## Context

**Epic:** Epic 5 — Build Developer Onboarding and Documentation
**Brownfield context:** `notebooks/02_training.ipynb` exists but is empty. This story implements it with a 3-epoch demo run, VRAM graphs, and config comparison.

**Primary files:** `notebooks/02_training.ipynb`

> ⚠️ **Dependency:** Epic 1 and Epic 2 must be complete. Story 3.2 (checkpoint management) must be complete.

## Acceptance Criteria

**AC1 — 3-epoch demo run with graphs:**
**Given** a developer opens `notebooks/02_training.ipynb`
**When** they run the notebook on an RTX 3060 12GB
**Then** the notebook runs a 3-epoch training loop on a small dataset subset (1000 samples) and displays: live VRAM usage graph (matplotlib), loss curve, accuracy curve, and double-loop `lr_scale` curve (if enabled)

**AC2 — Config comparison:**
**Given** the training section of the notebook
**When** it executes
**Then** it demonstrates both `quickstart.yaml` (beginner) and `default.yaml` (advanced) configurations with inline commentary on the differences
**And** a cell explicitly demonstrates what happens with `double_loop.enabled: false` vs `true` (shows startup banner difference)

**AC3 — Checkpoint structure demonstration:**
**Given** the notebook completes a short training run
**When** checkpoint saving is demonstrated
**Then** the notebook shows the checkpoint directory structure (`best.pt`, `latest.pt`, `checkpoint_NNNN.pt`) and explains each file's purpose

## Tasks

- [ ] **Task 1:** Implement 3-epoch training loop cell with small 1000-sample subset
- [ ] **Task 2:** Add matplotlib VRAM, loss, accuracy, and lr_scale plots
- [ ] **Task 3:** Add config comparison cells for quickstart vs default
- [ ] **Task 4:** Add double-loop enabled/disabled startup banner comparison cell
- [ ] **Task 5:** Add checkpoint directory walkthrough cell
- [ ] **Task 6:** Manual end-to-end run verification

## Tests Required

- Manual end-to-end run on RTX 3060
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
