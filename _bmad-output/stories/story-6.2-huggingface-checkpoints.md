---
id: "6.2"
epic: 6
title: "Upload Pre-Trained Checkpoints to Hugging Face"
status: "ready"
priority: "high"
estimate: "M"
assignee: ""
sprint: 6
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "scripts/upload_to_huggingface.py"
---

# Story 6.2: Upload Pre-Trained Checkpoints to Hugging Face

## User Story

As an independent AI developer who cannot run a 100-hour training run,
I want to download pre-trained NeuralMix checkpoints from Hugging Face,
So that I can experiment with the model immediately without training from scratch.

## Context

**Epic:** Epic 6 — Public Release and Community Launch
**Brownfield context:** No HF upload script exists yet. Trained checkpoints must be available from Story 3.2. Model cards need to be authored.

**Primary files:** `scripts/upload_to_huggingface.py` (new)

> ⚠️ **Dependency:** Story 3.2 (trained checkpoints) and Story 4.1 (benchmark results for model card) must be complete.

## Acceptance Criteria

**AC1 — HF model page content:**
**Given** the Hugging Face organization page for NeuralMix
**When** a developer visits it
**Then** at minimum the 100M and 250M parameter checkpoints are available as safetensors files
**And** each checkpoint has a model card containing: model architecture summary, training hardware (RTX 3060 12GB), training data description, benchmark results table (CIFAR-100, VQA accuracy), known limitations, and usage example with code snippet

**AC2 — Checkpoint loadable from HF:**
**Given** a developer runs `from huggingface_hub import hf_hub_download` to fetch a checkpoint
**When** they load it with `safe_load_checkpoint(path)`
**Then** the model loads without error and produces valid inference outputs
**And** the model card usage example code runs without modification

**AC3 — INT8 quantization stretch target:**
**Given** INT8 quantization is applied to the 250M checkpoint
**When** the quantized model is loaded for inference
**Then** peak VRAM is ≤ 4GB and throughput is ≥ 15 samples/second
**And** CIFAR-100 accuracy degradation vs. fp32 is ≤ 3%

## Tasks

- [ ] **Task 1:** Create `scripts/upload_to_huggingface.py` — uploads safetensors checkpoints and model card to HF Hub
- [ ] **Task 2:** Author model cards for 100M and 250M variants per AC1 requirements
- [ ] **Task 3:** Write and verify usage example code snippet in model card
- [ ] **Task 4:** (Stretch) Apply INT8 quantization via `torch.quantization` and measure accuracy/VRAM delta

## Tests Required

- Manual: download uploaded checkpoint via `hf_hub_download`; run usage example; verify valid outputs
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
