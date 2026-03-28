---
id: "1.4"
epic: 1
title: "Replace SimpleTokenizer with BERT Tokenizer and Add Checkpoint Validation"
status: "ready"
priority: "high"
estimate: "S"
assignee: ""
sprint: 1
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "src/data/dataset.py"
  - "src/models/text_encoder.py"
  - "src/training/trainer.py"
---

# Story 1.4: Replace SimpleTokenizer with BERT Tokenizer and Add Checkpoint Validation

## User Story

As an independent AI developer,
I want the text encoder to use a real BPE tokenizer,
So that text inputs are properly tokenized and accuracy benchmarks reflect genuine model capability rather than character-level encoding artifacts.

## Context

**Epic:** Epic 1 — Achieve Consumer GPU Training Target
**Brownfield context:** `AutoTokenizer.from_pretrained("bert-base-uncased")` is already conditionally loaded in `src/data/dataset.py`. `SimpleTokenizer` (character-level placeholder) still exists in `src/models/text_encoder.py`. The tokenizer needs to be fully activated and `SimpleTokenizer` guarded with a `UserWarning`. Two defensive fixes are also bundled here: empty validation loader ZeroDivisionError guard and `safe_load_checkpoint` external path guard.

**Primary files:** `src/data/dataset.py`, `src/models/text_encoder.py`, `src/training/trainer.py`

## Acceptance Criteria

**AC1 — BERT tokenizer in dataset:**
**Given** `AutoTokenizer.from_pretrained("bert-base-uncased")` is called during dataset initialization in `src/data/dataset.py`
**When** `MultiModalDataset.__getitem__()` processes a text sample
**Then** tokenization uses the BERT WordPiece tokenizer (30522 vocab), not `SimpleTokenizer`
**And** the tokenizer is loaded once at dataset construction and reused for all samples (not re-instantiated per sample)

**AC2 — SimpleTokenizer UserWarning:**
**Given** `SimpleTokenizer` is still present in `src/models/text_encoder.py` as a fallback
**When** `SimpleTokenizer` is instantiated anywhere in the codebase
**Then** a `UserWarning` is emitted: `"SimpleTokenizer is a character-level research placeholder. Replace with AutoTokenizer for any accuracy benchmarking."`

**AC3 — Checkpoint external path guard:**
**Given** a training checkpoint was saved with safetensors format
**When** `safe_load_checkpoint()` is called with `allow_external=False` (default)
**Then** the checkpoint is loaded successfully from an internal project path
**And** a checkpoint saved at an external absolute path raises `ValueError` when `allow_external=False`

**AC4 — Empty validation loader guard:**
**Given** `validate()` in `trainer.py` is called with an empty validation loader
**When** the validation loop runs zero batches
**Then** the function returns `{"loss": 0.0, "accuracy": 0.0}` without a ZeroDivisionError
**And** a warning log is emitted: `"Validation loader is empty — skipping validation"`

## Tasks

- [ ] **Task 1:** In `src/data/dataset.py` ensure `AutoTokenizer.from_pretrained("bert-base-uncased")` is the active tokenizer in `MultiModalDataset.__init__()`; confirm it is not re-instantiated per sample
- [ ] **Task 2:** In `src/models/text_encoder.py` add `warnings.warn(...)` to `SimpleTokenizer.__init__()` with the prescribed message
- [ ] **Task 3:** In `src/training/trainer.py` (or wherever `safe_load_checkpoint` lives) add path validation that raises `ValueError` for external paths when `allow_external=False`
- [ ] **Task 4:** In `trainer.py` `validate()` guard against `len(val_loader) == 0`; return early with zeros and log warning
- [ ] **Task 5:** Write/update unit tests covering AC1–AC4

## Tests Required

- `tests/test_dataset.py` — verify `AutoTokenizer` used; verify single instantiation; verify token ids in BERT vocab range [0, 30521]
- `tests/test_text_encoder.py` — verify `SimpleTokenizer` emits `UserWarning`
- `tests/test_trainer.py` — verify `ValueError` on external path with `allow_external=False`; verify empty loader returns zeros without exception
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
