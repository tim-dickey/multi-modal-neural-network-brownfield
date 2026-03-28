---
id: "5.1"
epic: 5
title: "Build Getting Started Notebook (01_getting_started.ipynb)"
status: "ready"
priority: "high"
estimate: "M"
assignee: ""
sprint: 5
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "notebooks/01_getting_started.ipynb"
---

# Story 5.1: Build Getting Started Notebook (01_getting_started.ipynb)

## User Story

As an independent AI developer new to NeuralMix,
I want a complete getting-started notebook,
So that I can go from `git clone` to a running forward pass in under 15 minutes with no prior knowledge of the codebase.

## Context

**Epic:** Epic 5 — Build Developer Onboarding and Documentation
**Brownfield context:** `notebooks/01_getting_started.ipynb` exists but is empty. This story fully implements it.

**Primary files:** `notebooks/01_getting_started.ipynb`

> ⚠️ **Dependency:** Story 1.5 (startup banner + `--check` mode) and Story 1.6 (quickstart config) must be complete.

## Acceptance Criteria

**AC1 — Runs end-to-end without error:**
**Given** a developer clones the repo and opens `notebooks/01_getting_started.ipynb`
**When** they run all cells top to bottom
**Then** the notebook completes without error on a system with CUDA ≥ 10GB VRAM
**And** the final cell produces a model forward pass with real (not dummy) output tensors and prints their shapes

**AC2 — Required sections in order:**
**Given** the notebook is run
**When** each major section executes
**Then** the sections cover in order: hardware check (VRAM, CUDA version), environment validation (`--check` mode output), loading `quickstart.yaml`, instantiating the model, loading a sample image + text, running a forward pass, and interpreting the output
**And** inline markdown cells explain each step with concrete numbers

**AC3 — Colab compatible:**
**Given** a developer without a local GPU opens the notebook in Google Colab
**When** they run the Colab-specific setup cell (pip installs, mount drive)
**Then** the notebook runs end-to-end using the Colab T4 GPU without modification
**And** a Colab badge link is present at the top of the notebook README section

## Tasks

- [ ] **Task 1:** Implement all notebook sections per AC2 order; add inline markdown explanations
- [ ] **Task 2:** Add Colab setup cell (conditional pip installs, drive mount); add Colab badge at top
- [ ] **Task 3:** Verify final cell prints output tensor shapes from a real forward pass
- [ ] **Task 4:** Manual verification: run notebook top-to-bottom on target hardware

## Tests Required

- Manual end-to-end run on RTX 3060 or Colab T4
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
