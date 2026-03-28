---
id: "2.2"
epic: 2
title: "Wire Double-Loop Controller Outputs to Optimizer LR Adaptation"
status: "ready"
priority: "high"
estimate: "M"
assignee: ""
sprint: 2
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "src/training/trainer.py"
  - "src/training/losses.py"
---

# Story 2.2: Wire Double-Loop Controller Outputs to Optimizer LR Adaptation

## User Story

As a researcher studying meta-learning,
I want the double-loop controller's `lr_scale` output to adaptively modify the optimizer learning rate,
So that the outer loop can slow or accelerate learning in response to observed training dynamics.

## Context

**Epic:** Epic 2 — Activate and Validate Meta-Learning Training Loop
**Brownfield context:** `AdaptiveLRController` exists but `update_lr()` is never called from `train_epoch()`. `MetaLoss` exists in `losses.py` but is only called from `train_step()` which is never called from `train_epoch()` — creating a dead code path. `train_step()` must be eliminated as a dead code path and its logic folded into `train_epoch()`.

**Primary files:** `src/training/trainer.py`, `src/training/losses.py`

> ⚠️ **Dependency:** Story 2.1 must be complete (double-loop inputs wired).

## Acceptance Criteria

**AC1 — LR scale applied to optimizer:**
**Given** the double-loop controller is active and `step % update_frequency == 0`
**When** `DoubleLoopController.forward()` returns `meta_info` containing `lr_scale` and `meta_loss`
**Then** `AdaptiveLRController.update_lr(optimizer, meta_info["lr_scale"])` is called immediately after the controller forward pass
**And** the optimizer's learning rate for all parameter groups is scaled by `lr_scale.mean().item()` (clamped to [0.1, 2.0] to prevent degenerate values)
**And** the base learning rate is restored to `inner_lr` at the start of each epoch (LR scaling is per-step, not cumulative)

**AC2 — meta_loss combined with task_loss:**
**Given** `meta_info["meta_loss"]` is returned by the controller
**When** total loss is computed
**Then** `total_loss = task_loss + meta_loss_weight * meta_info["meta_loss"]` where `meta_loss_weight` defaults to 0.1 (configurable via `double_loop.meta_loss_weight`)
**And** `MetaLoss` in `losses.py` is used for this combination
**And** `train_step()` is eliminated as a dead code path and replaced by the wired logic in `train_epoch()`

**AC3 — arch_adaptation deferred log:**
**Given** the `arch_adaptation` output from the controller
**When** v1 training runs
**Then** a log message is emitted: `[double-loop] arch_adaptation output produced but AdaptiveLayerNorm wiring deferred to v1.5`
**And** `arch_adaptation` is detached and not used in the computation graph (no gradient flows through it in v1)

## Tasks

- [ ] **Task 1:** After controller forward pass in `train_epoch()`, call `adaptive_lr_controller.update_lr(optimizer, meta_info["lr_scale"])` with clamping to [0.1, 2.0]
- [ ] **Task 2:** At start of each epoch, restore base LR to `inner_lr` for all param groups
- [ ] **Task 3:** Compute `total_loss = task_loss + meta_loss_weight * meta_info["meta_loss"]` using `MetaLoss`; remove dead `train_step()` code path
- [ ] **Task 4:** Detach `arch_adaptation`; emit deferred-to-v1.5 log message
- [ ] **Task 5:** Make `meta_loss_weight` configurable via `double_loop.meta_loss_weight` (default 0.1)
- [ ] **Task 6:** Write/update unit tests covering AC1–AC3

## Tests Required

- `tests/test_trainer.py` — verify `update_lr()` called with clamped lr_scale; verify base LR restored each epoch; verify `total_loss` includes meta_loss contribution; verify `arch_adaptation` detached
- `tests/test_losses.py` — verify MetaLoss combines task and meta loss with correct weight
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
