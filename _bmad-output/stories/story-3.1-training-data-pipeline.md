---
id: "3.1"
epic: 3
title: "Assemble and Validate Training Data Pipeline"
status: "ready"
priority: "high"
estimate: "M"
assignee: ""
sprint: 3
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "src/data/dataset.py"
  - "src/data/selector.py"
  - "src/data/coco_dataset.py"
---

# Story 3.1: Assemble and Validate Training Data Pipeline

## User Story

As a researcher training NeuralMix,
I want a validated multi-dataset training pipeline assembled from COCO captions and ImageNet-subset,
So that the model has sufficient multimodal training data to learn cross-modal representations.

## Context

**Epic:** Epic 3 — Execute Full Training Run and Produce Results
**Brownfield context:** `COCOCaptionsDataset`, `ImageNetDataset`, and `selector.py` are implemented. Key bug: `collate_fn` uses key `"image"` while `Trainer._normalize_batch()` expects `"images"` — this latent key mismatch must be resolved at the dataset level. `ConcatDataset` assembly needs split-ratio validation.

**Primary files:** `src/data/dataset.py`, `src/data/selector.py`

> ⚠️ **Dependency:** Epic 1 (Stories 1.1–1.5) must be complete.

## Acceptance Criteria

**AC1 — COCO dataset loads correctly:**
**Given** COCO annotations and images are present at the configured path
**When** `build_dataloaders(config)` is called with `selector.py`
**Then** `COCOCaptionsDataset` loads successfully and `len(train_loader.dataset)` returns the expected sample count
**And** each batch contains `images` (B, 3, 224, 224), `input_ids` (B, seq_len), `attention_mask` (B, seq_len), and `labels` (B,) tensors
**And** the `collate_fn` key mismatch (`"image"` vs `"images"`) is resolved — all datasets produce `"images"` keyed outputs, removing the need for `_normalize_batch()` renaming

**AC2 — Multi-dataset ConcatDataset:**
**Given** `selector.py` config lists multiple datasets with `enabled: true`
**When** `build_dataloaders()` assembles datasets
**Then** a `ConcatDataset` is returned for the train split containing samples from all enabled datasets
**And** dataset split ratios are validated to sum to 1.0 with a clear `ValueError` if they do not

**AC3 — Batch content validation:**
**Given** a data loader is constructed
**When** 3 consecutive batches are drawn
**Then** no `None` values appear in any tensor field
**And** image tensors are normalized to ImageNet stats (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
**And** text tokens are within the BERT vocabulary range [0, 30521]

**AC4 — Missing path error:**
**Given** a dataset path does not exist at the configured location
**When** `build_dataloaders()` is called
**Then** a clear `FileNotFoundError` with the missing path is raised (not a silent fallback to dummy data during training)

## Tasks

- [ ] **Task 1:** Fix key mismatch — update all dataset `__getitem__` methods to return `"images"` key (not `"image"`); remove `_normalize_batch()` renaming logic from `Trainer`
- [ ] **Task 2:** Add split-ratio sum validation in `selector.py` `build_dataloaders()` — raise `ValueError` with clear message if ratios don't sum to 1.0
- [ ] **Task 3:** Add `FileNotFoundError` for missing dataset paths (replace any silent fallback)
- [ ] **Task 4:** Write/update unit tests covering AC1–AC4

## Tests Required

- `tests/test_dataset.py` — verify `"images"` key in all dataset outputs; verify ImageNet normalization stats; verify token range [0, 30521]
- `tests/test_selector.py` — verify `ValueError` on bad split ratios; verify `FileNotFoundError` on missing path; verify `ConcatDataset` from multiple enabled datasets
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
