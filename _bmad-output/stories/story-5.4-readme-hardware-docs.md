---
id: "5.4"
epic: 5
title: "Write README and Hardware Compatibility Documentation"
status: "ready"
priority: "high"
estimate: "M"
assignee: ""
sprint: 5
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "README.md"
---

# Story 5.4: Write README and Hardware Compatibility Documentation

## User Story

As an independent AI developer discovering NeuralMix,
I want a complete README with a hardware compatibility table at the top,
So that I can immediately determine if my GPU is supported before cloning the repository.

## Context

**Epic:** Epic 5 — Build Developer Onboarding and Documentation
**Brownfield context:** `README.md` exists but is incomplete for public release. This story rewrites it to be release-ready with hardware table above the fold, quickstart commands, architecture overview, and known limitations.

**Primary files:** `README.md`

> ⚠️ **Dependency:** Story 1.5 (VRAM profiling numbers) must be complete for accurate hardware table values.

## Acceptance Criteria

**AC1 — Hardware table above the fold:**
**Given** a developer lands on the NeuralMix GitHub repository page
**When** they read the README
**Then** the first visible content (above the fold) is: project name, one-line value proposition, and a hardware compatibility table listing each tested GPU, measured peak VRAM during training, relative training speed, and support status

**AC2 — Required hardware table entries:**
**Given** the README hardware table
**When** it is read
**Then** it includes at minimum: RTX 3060 12GB (✅ fully tested), RTX 3070 8GB (⚠️ limited), RTX 4070 12GB (✅), RTX 3080 12GB (✅), RX 6700 XT 12GB (✅ ROCm), with measured VRAM numbers from Story 1.5 profiling

**AC3 — Required README body content:**
**Given** the README body
**When** it is read
**Then** it contains: quickstart install instructions, first training run command (`python train.py --config configs/quickstart.yaml`), link to `01_getting_started.ipynb`, link to `02_training.ipynb`, architecture overview ASCII diagram (from architecture doc §1.1), known limitations section listing Wolfram Alpha (v1.5), WebDataset (v1.5), and multi-GPU DDP (v1.5)

## Tasks

- [ ] **Task 1:** Rewrite README with hardware compatibility table as first content section
- [ ] **Task 2:** Add quickstart install + first run commands
- [ ] **Task 3:** Add architecture ASCII diagram from `_bmad-output/implementation-artifacts/architecture-2026-03-03.md §1.1`
- [ ] **Task 4:** Add known limitations section with v1.5 deferrals clearly marked
- [ ] **Task 5:** Add notebook badge links (Colab, GitHub)

## Tests Required

- No automated tests — manual review against AC checklist
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
