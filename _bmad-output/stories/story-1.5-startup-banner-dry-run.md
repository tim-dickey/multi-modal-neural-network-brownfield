---
id: "1.5"
epic: 1
title: "Add Startup Banner and Validate Consumer GPU Training in Dry Run"
status: "ready"
priority: "high"
estimate: "M"
assignee: ""
sprint: 1
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "train.py"
  - "src/training/trainer.py"
---

# Story 1.5: Add Startup Banner and Validate Consumer GPU Training in Dry Run

## User Story

As an independent AI developer,
I want a startup banner printed when training begins and a `--check` dry-run mode,
So that I immediately know whether my hardware, VRAM, and configuration are compatible before committing to a 100+ hour training run.

## Context

**Epic:** Epic 1 — Achieve Consumer GPU Training Target
**Brownfield context:** `train.py` exists but does not print a startup banner. No `--check` CLI flag exists. This story depends on Stories 1.1–1.4 being complete (AMP, checkpointing, Flash Attention, and tokenizer must all be active for a meaningful dry run).

**Primary files:** `train.py`, `src/training/trainer.py`

> ⚠️ **Dependency:** Stories 1.1, 1.2, 1.3, 1.4 must be complete before this story is verified end-to-end.

## Acceptance Criteria

**AC1 — Startup banner on normal launch:**
**Given** `python train.py` is executed
**When** `Trainer.__init__()` completes initialization
**Then** a startup banner is printed to stdout containing:
- Detected device type (CUDA/MPS/CPU)
- Available VRAM (if CUDA)
- Selected precision mode (bf16/fp32)
- Active configuration file path
- Gradient checkpointing status (enabled/disabled)
- Wolfram Alpha status (`[wolfram] ⚠ Inactive — deferred to v1.5`)
- Double-loop status: `[double-loop] ✓ Active — update_frequency=N` or `[double-loop] ⚠ Inactive`

**AC2 — `--check` dry-run mode:**
**Given** `python train.py --check` is executed
**When** the dry-run mode runs
**Then** the system validates: CUDA availability and VRAM ≥ 10GB, config file loads without error, model forward pass completes on a single synthetic batch without OOM, and tokenizer loads successfully
**And** a `✓ System check passed — ready to train` or `✗ System check failed: [reason]` message is printed
**And** the process exits with code 0 on success or 1 on failure without starting the training loop

**AC3 — Low VRAM warning:**
**Given** `train.py` is run on a system with less than 10GB available VRAM
**When** the startup banner is printed
**Then** a `⚠ WARNING: Available VRAM (N GB) is below recommended 10GB. Consider reducing micro_batch_size or enabling gradient_checkpointing.` warning is included

**AC4 — End-to-end epoch validation:**
**Given** a full training epoch runs on RTX 3060 12GB with AMP + gradient checkpointing + Flash Attention active
**When** the epoch completes
**Then** peak VRAM usage reported by `torch.cuda.max_memory_allocated()` is ≤ 11.5GB
**And** training speed is ≥ 5 samples/second (minimum threshold per PRD Table 4)
**And** the training loss is a finite number (not NaN or Inf)

## Tasks

- [ ] **Task 1:** Add `--check` argument to `train.py` argument parser
- [ ] **Task 2:** Implement `_print_startup_banner()` method in `Trainer` (or as a standalone function in `train.py`) — prints device, VRAM, precision, config path, checkpointing status, wolfram status, double-loop status
- [ ] **Task 3:** Implement dry-run validation logic for `--check` mode: load config, instantiate model, run one synthetic forward pass, load tokenizer; print pass/fail and exit with appropriate code
- [ ] **Task 4:** Add low-VRAM warning branch when `torch.cuda.get_device_properties().total_memory < 10GB`
- [ ] **Task 5:** Write/update unit tests covering AC1–AC3 (AC4 is hardware-dependent integration test)

## Tests Required

- `tests/test_train.py` (or `tests/test_trainer.py`) — mock CUDA; verify banner contains required fields; verify `--check` exits 0 on valid config; verify `--check` exits 1 on invalid config; verify low-VRAM warning emitted
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
