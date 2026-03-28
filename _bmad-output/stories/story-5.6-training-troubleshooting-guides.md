---
id: "5.6"
epic: 5
title: "Write Training Guide and Troubleshooting Guide"
status: "ready"
priority: "medium"
estimate: "S"
assignee: ""
sprint: 5
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "docs/TRAINING_GUIDE.md"
  - "docs/TROUBLESHOOTING.md"
---

# Story 5.6: Write Training Guide and Troubleshooting Guide

## User Story

As an independent AI developer training NeuralMix,
I want a training guide and troubleshooting guide in the `docs/` folder,
So that I can resolve common issues (OOM, convergence failure, tokenizer errors) without needing to post on Discord.

## Context

**Epic:** Epic 5 — Build Developer Onboarding and Documentation
**Brownfield context:** `TRAINING_GUIDE.md` exists (21671 bytes — partial content). `docs/TROUBLESHOOTING.md` does not exist yet. Both need to be complete for v1.0 release. Review existing `TRAINING_GUIDE.md` and fill gaps before creating `TROUBLESHOOTING.md`.

**Primary files:** `docs/TRAINING_GUIDE.md` (update), `docs/TROUBLESHOOTING.md` (new)

## Acceptance Criteria

**AC1 — Training guide completeness:**
**Given** `docs/TRAINING_GUIDE.md` exists
**When** a developer reads it
**Then** it covers: recommended hardware, step-by-step first training run, how to adjust batch size and gradient accumulation for different VRAM sizes, how to monitor training with W&B, how to resume from checkpoint, and how to evaluate a trained model

**AC2 — Troubleshooting guide structure:**
**Given** `docs/TROUBLESHOOTING.md` exists
**When** a developer reads it
**Then** it covers at minimum: CUDA OOM (with specific config changes to try), NaN loss (learning rate too high, AMP numerical instability), slow training (GPU utilization < 70% — data loading bottleneck), tokenizer warning (how to confirm `AutoTokenizer` is active), and double-loop inactive warning (how to enable)
**And** each issue has a "Symptom", "Cause", and "Fix" structure

## Tasks

- [ ] **Task 1:** Review existing `TRAINING_GUIDE.md`; add any missing sections per AC1
- [ ] **Task 2:** Create `docs/TROUBLESHOOTING.md` with all 5 required issues in Symptom/Cause/Fix format
- [ ] **Task 3:** Add cross-links between README, TRAINING_GUIDE, and TROUBLESHOOTING

## Tests Required

- Manual review against AC1 and AC2 checklists
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
