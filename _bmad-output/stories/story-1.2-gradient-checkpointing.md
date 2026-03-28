---
id: "1.2"
epic: 1
title: "Apply Gradient Checkpointing to Encoder Forward Passes"
status: "ready"
priority: "high"
estimate: "S"
assignee: ""
sprint: 1
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "src/models/vision_encoder.py"
  - "src/models/text_encoder.py"
  - "src/training/trainer.py"
---

# Story 1.2: Apply Gradient Checkpointing to Encoder Forward Passes

## User Story

As an independent AI developer,
I want gradient checkpointing applied to the transformer blocks in the vision and text encoders,
So that activation memory is reduced by 30–40% during backpropagation, enabling larger effective batch sizes.

## Context

**Epic:** Epic 1 — Achieve Consumer GPU Training Target
**Brownfield context:** `training.gradient_checkpointing` config flag already exists. `model.gradient_checkpointing_enable()` or equivalent pattern needs to be called in `Trainer.__init__()`. The actual `torch.utils.checkpoint.checkpoint` call is NOT yet applied in the encoder `forward()` methods.

**Primary files:** `src/models/vision_encoder.py`, `src/models/text_encoder.py`
**Config key:** `training.gradient_checkpointing: true`

## Acceptance Criteria

**AC1 — Checkpointing applied to encoders:**
**Given** `training.gradient_checkpointing: true` is set in config
**When** `VisionEncoder.forward()` processes transformer blocks
**Then** `torch.utils.checkpoint.checkpoint` is applied to every 2nd transformer block (blocks at index 1, 3, 5, 7, 9, 11)
**And** the same checkpoint pattern is applied to every 2nd block in `TextEncoder.forward()`
**And** the model's `self.gradient_checkpointing` flag is set to `True` via `model.gradient_checkpointing_enable()` or equivalent in `Trainer.__init__()`

**AC2 — Disabled when config is false:**
**Given** `training.gradient_checkpointing: false` is set in config
**When** the encoders process transformer blocks
**Then** no checkpointing is applied and forward passes execute normally

**AC3 — VRAM reduction verified:**
**Given** gradient checkpointing is active during training
**When** a backward pass completes
**Then** peak VRAM usage is measurably lower than without checkpointing (verified by `torch.cuda.max_memory_allocated()` comparison in test)
**And** training loss convergence is identical to the non-checkpointed run (same random seed, same data, same hyperparameters)

**AC4 — Disabled during eval:**
**Given** gradient checkpointing is enabled
**When** the model is set to `eval()` mode for validation
**Then** checkpointing is automatically disabled during validation (no recomputation overhead)

## Tasks

- [ ] **Task 1:** Add `self.gradient_checkpointing = False` attribute to `VisionEncoder` and `TextEncoder` `__init__`
- [ ] **Task 2:** Add `gradient_checkpointing_enable()` / `gradient_checkpointing_disable()` methods to both encoders
- [ ] **Task 3:** In `VisionEncoder.forward()`, wrap every 2nd transformer block (indices 1,3,5,7,9,11) with `torch.utils.checkpoint.checkpoint` when `self.gradient_checkpointing and self.training`
- [ ] **Task 4:** Apply the same pattern in `TextEncoder.forward()`
- [ ] **Task 5:** Call `model.gradient_checkpointing_enable()` in `Trainer.__init__()` when config flag is true
- [ ] **Task 6:** Write/update unit tests covering AC1–AC4

## Tests Required

- `tests/test_vision_encoder.py` — verify checkpointing flag; verify VRAM difference (CUDA only, skip on CPU)
- `tests/test_text_encoder.py` — same pattern
- `tests/test_trainer.py` — verify `gradient_checkpointing_enable()` called on init when config true
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
