---
id: "3.2"
epic: 3
title: "Execute 50-Epoch Training Run with Checkpoint Management"
status: "ready"
priority: "high"
estimate: "L"
assignee: ""
sprint: 3
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "train.py"
  - "src/training/trainer.py"
  - "src/training/checkpoint_manager.py"
---

# Story 3.2: Execute 50-Epoch Training Run with Checkpoint Management

## User Story

As a researcher training NeuralMix,
I want to run a complete 50-epoch training session that saves checkpoints and resumes from interruption,
So that I can train on consumer hardware over multiple sessions without losing progress.

## Context

**Epic:** Epic 3 — Execute Full Training Run and Produce Results
**Brownfield context:** `CheckpointManager` is structurally complete with `latest.pt`, `best.pt`, and numbered checkpoint save logic. Resume logic exists. This story validates the full end-to-end training run on real data and confirms checkpoint rotation and resume correctness.

**Primary files:** `train.py`, `src/training/trainer.py`, `src/training/checkpoint_manager.py`

> ⚠️ **Dependency:** All Epic 1 and Epic 2 stories must be complete. Story 3.1 (data pipeline) must be complete.

## Acceptance Criteria

**AC1 — 50-epoch run completes without OOM:**
**Given** all Epic 1 and Epic 2 stories are complete and a training dataset is configured
**When** `python train.py --config configs/default.yaml` is run for 50 epochs
**Then** training completes without OOM error on RTX 3060 12GB
**And** a checkpoint is saved every 5 epochs as `checkpoint_NNNN.pt` and `checkpoint_NNNN.safetensors`
**And** `best.pt` is updated whenever validation loss improves
**And** `latest.pt` is updated after every epoch

**AC2 — Resume from interruption:**
**Given** training is interrupted (process killed) at epoch N
**When** `python train.py --resume` is run
**Then** training resumes from `latest.pt` at epoch N+1 with the same optimizer state, scheduler state, and training metrics
**And** the resumed run produces identical loss values to an uninterrupted run (given the same data order)

**AC3 — Checkpoint rotation:**
**Given** `checkpoint_manager.max_checkpoints: 5` in config
**When** more than 5 numbered checkpoints exist
**Then** the oldest numbered checkpoint is deleted, keeping only the 5 most recent plus `best.pt` and `latest.pt`

**AC4 — Epoch metrics logged:**
**Given** a training epoch completes
**When** metrics are logged
**Then** train loss, train accuracy, validation loss, validation accuracy, learning rate, and epoch duration are all logged to the metrics file and (if configured) to W&B

## Tasks

- [ ] **Task 1:** Verify checkpoint save every 5 epochs in `CheckpointManager` — both `.pt` and `.safetensors`; update `best.pt` on validation loss improvement; update `latest.pt` each epoch
- [ ] **Task 2:** Verify `--resume` flag loads `latest.pt` and restores optimizer + scheduler + training state
- [ ] **Task 3:** Implement checkpoint rotation: keep max N numbered checkpoints (configurable); delete oldest when exceeded
- [ ] **Task 4:** Verify all epoch metrics are logged (train loss, train acc, val loss, val acc, LR, epoch duration)
- [ ] **Task 5:** Write/update unit tests covering AC2–AC4 (AC1 is hardware integration)

## Tests Required

- `tests/test_checkpoint_manager.py` — verify rotation deletes oldest when max exceeded; verify `best.pt` updated on improvement; verify `latest.pt` updated every epoch
- `tests/test_trainer.py` — verify resume restores optimizer state; verify metrics dict contains all required keys
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
