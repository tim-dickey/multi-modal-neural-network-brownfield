---
id: "1.1"
epic: 1
title: "Apply BF16 Automatic Mixed Precision to Training Loop"
status: "ready"
priority: "high"
estimate: "S"
assignee: ""
sprint: 1
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "src/training/trainer.py"
---

# Story 1.1: Apply BF16 Automatic Mixed Precision to Training Loop

## User Story

As an independent AI developer,
I want the training loop to use BF16 automatic mixed precision,
So that GPU memory usage is reduced by ~40% and I can train the 250M model without exhausting VRAM.

## Context

**Epic:** Epic 1 — Achieve Consumer GPU Training Target
**Brownfield context:** `self.scaler` (GradScaler) is already initialized in `Trainer.__init__()`. `training.mixed_precision` config key already exists. AMP is configured but NOT applied in `train_epoch()`. This story wires the existing initialization to the actual training loop.

**Primary file:** `src/training/trainer.py`
**Config key:** `training.mixed_precision: bf16`

## Acceptance Criteria

**AC1 — AMP forward/backward pass:**
**Given** `training.mixed_precision: bf16` is set in `configs/default.yaml`
**When** `train_epoch()` executes a forward + backward pass
**Then** the forward pass runs inside `torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)`
**And** `self.scaler.scale(loss).backward()` is called instead of `loss.backward()`
**And** `self.scaler.step(optimizer)` and `self.scaler.update()` replace the bare `optimizer.step()` call
**And** the existing `self.scaler` (GradScaler) initialized in `Trainer.__init__()` is used — no new scaler is created

**AC2 — Non-AMP fallback:**
**Given** `training.mixed_precision` is set to anything other than `bf16` or `fp16`
**When** `train_epoch()` runs
**Then** the autocast context is skipped and training proceeds in full precision without error

**AC3 — Validation AMP:**
**Given** the training loop runs with AMP active
**When** a validation step executes in `validate()`
**Then** validation also runs inside `torch.amp.autocast` with the same dtype
**And** `self.scaler` is NOT used during validation (no backward pass in validation)

**AC4 — Loss logging:**
**Given** a training step completes with AMP
**When** `logging_manager.log_metrics()` is called
**Then** the logged loss value is a Python float (not a scaled tensor)

## Tasks

- [ ] **Task 1:** Wrap `train_epoch()` forward pass in `torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)` conditioned on `mixed_precision` config
- [ ] **Task 2:** Replace `loss.backward()` with `self.scaler.scale(loss).backward()`
- [ ] **Task 3:** Replace `optimizer.step()` with `self.scaler.step(optimizer)` + `self.scaler.update()`
- [ ] **Task 4:** Add `torch.amp.autocast` context to `validate()` (no scaler)
- [ ] **Task 5:** Ensure logged loss is `.item()` float
- [ ] **Task 6:** Write/update unit tests covering AC1–AC4

## Tests Required

- `tests/test_trainer.py` — mock autocast context manager; verify scaler methods called; verify fallback with non-AMP config; verify validate() uses autocast but not scaler
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
