---
id: "2.1"
epic: 2
title: "Wire Double-Loop Controller Inputs to Training Loop"
status: "ready"
priority: "high"
estimate: "M"
assignee: ""
sprint: 2
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "src/training/trainer.py"
  - "src/models/multi_modal_model.py"
---

# Story 2.1: Wire Double-Loop Controller Inputs to Training Loop

## User Story

As a researcher studying meta-learning,
I want the double-loop controller to receive live training metrics (loss, accuracy, gradient norm) during training,
So that the LSTM meta-controller can observe training history and produce adaptation signals.

## Context

**Epic:** Epic 2 — Activate and Validate Meta-Learning Training Loop
**Brownfield context:** `DoubleLoopController` is structurally complete in `src/models/double_loop_controller.py`. `MultiModalModel.forward()` accepts `current_loss`, `current_accuracy`, `gradient_norm` optional params. `train_epoch()` does NOT yet pass these. `WOLFRAM_API_KEY` env var pattern already follows NFR6 — this story verifies it.

**Primary files:** `src/training/trainer.py`

> ⚠️ **Dependency:** Epic 1 (Stories 1.1–1.5) must be complete. `prev_grad_norm` requires `GradientClipper` to return the grad norm after `unscale_`, which is set up in Story 1.1's scaler integration.

## Acceptance Criteria

**AC1 — Live metrics passed to forward():**
**Given** `model.double_loop.enabled: true` in config and `Trainer.train_epoch()` is executing
**When** a training step completes
**Then** `prev_loss`, `prev_accuracy`, and `prev_grad_norm` are tracked as scalar tensors across steps
**And** the model's `forward()` call passes these as `current_loss=prev_loss`, `current_accuracy=prev_accuracy`, `gradient_norm=prev_grad_norm`
**And** `prev_loss` is updated from `task_loss.detach()` after each backward pass
**And** `prev_accuracy` is computed as `(logits.argmax(-1) == labels).float().mean().detach()`
**And** `prev_grad_norm` is the L2 gradient norm returned by `GradientClipper` after `unscale_`

**AC2 — Zero-initialization on first step:**
**Given** the first step of the first epoch (no previous step exists)
**When** `forward()` is called with double-loop inputs
**Then** `prev_loss`, `prev_accuracy`, and `prev_grad_norm` are initialized to zero tensors of the correct shape
**And** the controller handles zero-initialized inputs without NaN or error

**AC3 — No overhead when disabled:**
**Given** `model.double_loop.enabled: false` in config
**When** `train_epoch()` runs
**Then** none of the double-loop inputs are passed to `forward()` and no overhead is incurred
**And** the startup banner shows `[double-loop] ⚠ Inactive — set double_loop.enabled: true to activate`

**AC4 — Wolfram env var banner:**
**Given** the `WOLFRAM_API_KEY` environment variable is not set
**When** the startup banner is printed
**Then** the banner shows `[wolfram] ⚠ Inactive — deferred to v1.5` without error

## Tasks

- [ ] **Task 1:** In `Trainer.train_epoch()` initialize `prev_loss`, `prev_accuracy`, `prev_grad_norm` as zero tensors before the batch loop
- [ ] **Task 2:** After each backward+clip step, update `prev_loss`, `prev_accuracy`, `prev_grad_norm` from current step values
- [ ] **Task 3:** Pass these three tensors to `model.forward()` when `double_loop.enabled: true`
- [ ] **Task 4:** Skip double-loop inputs entirely when `double_loop.enabled: false`
- [ ] **Task 5:** Write/update unit tests covering AC1–AC4

## Tests Required

- `tests/test_trainer.py` — mock controller; verify `forward()` called with `current_loss`/`current_accuracy`/`gradient_norm` when enabled; verify zero init on step 0; verify NOT passed when disabled
- `tests/test_double_loop_controller.py` — verify zero-initialized inputs don't produce NaN outputs
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
