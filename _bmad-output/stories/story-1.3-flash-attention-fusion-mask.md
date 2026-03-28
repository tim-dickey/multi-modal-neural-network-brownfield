---
id: "1.3"
epic: 1
title: "Replace Standard Attention with Flash Attention 2 and Fix Fusion Mask"
status: "ready"
priority: "high"
estimate: "M"
assignee: ""
sprint: 1
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "src/models/vision_encoder.py"
  - "src/models/text_encoder.py"
  - "src/models/fusion_layer.py"
---

# Story 1.3: Replace Standard Attention with Flash Attention 2 and Fix Fusion Mask

## User Story

As an independent AI developer,
I want attention computation to use PyTorch's scaled dot-product attention (Flash Attention 2 backend),
So that attention memory is reduced from O(N²) to O(N), making sequences of length 196+512 feasible within 11.5GB VRAM.

## Context

**Epic:** Epic 1 — Achieve Consumer GPU Training Target
**Brownfield context:** All three attention modules currently use manual `q @ k.transpose(-2, -1) * scale` + softmax + `@ v`. This must be replaced with `F.scaled_dot_product_attention`. Additionally, `EarlyFusionLayer` has the combined attention mask implementation commented out at lines 204–209 — this must be uncommented and implemented.

**Primary files:** `src/models/vision_encoder.py`, `src/models/text_encoder.py`, `src/models/fusion_layer.py`

> ⚠️ **Critical:** Without Flash Attention 2, estimated peak VRAM is 14–16GB — hard OOM on RTX 3060. This story is a hard prerequisite for Story 1.5 (end-to-end validation).

## Acceptance Criteria

**AC1 — Vision encoder attention replacement:**
**Given** the `MultiHeadAttention` module in `src/models/vision_encoder.py`
**When** `forward()` computes attention
**Then** the manual `(q @ k.transpose(-2, -1)) * scale` + softmax + `@ v` is replaced with `F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p if self.training else 0.0, is_causal=False)`
**And** `torch.nn.functional` is imported at the top of the file

**AC2 — Text encoder attention replacement:**
**Given** the `TextMultiHeadAttention` module in `src/models/text_encoder.py`
**When** `forward()` computes attention
**Then** the same `F.scaled_dot_product_attention` replacement is applied
**And** the `attention_mask` from the text encoder is correctly converted to an additive bias mask (0 for real tokens, -inf for padding) before being passed to `scaled_dot_product_attention`

**AC3 — Fusion layer mask fix:**
**Given** `EarlyFusionLayer.forward()` with a batch containing variable-length text inputs
**When** cross-modal attention runs in a `FusionTransformerBlock`
**Then** the combined attention mask (lines 204–209) is uncommented and implemented: vision tokens have no padding mask, text tokens use their `text_mask`; the combined mask prevents text padding positions from being attended to during cross-modal attention
**And** the mask is correctly shaped and broadcast for the multi-head attention computation

**AC4 — Flash backend dispatched on CUDA:**
**Given** a CUDA device with PyTorch 2.0+ is available
**When** `F.scaled_dot_product_attention` is called
**Then** PyTorch automatically dispatches to the Flash Attention kernel (verified by `torch.backends.cuda.flash_sdp_enabled()` returning `True` in unit test)

**AC5 — CPU fallback:**
**Given** standard CPU or non-Flash-capable hardware
**When** `F.scaled_dot_product_attention` is called
**Then** PyTorch falls back to the math attention kernel without error

## Tasks

- [ ] **Task 1:** In `MultiHeadAttention.forward()` (`vision_encoder.py`) replace manual attention with `F.scaled_dot_product_attention`; add `import torch.nn.functional as F` if not present
- [ ] **Task 2:** In `TextMultiHeadAttention.forward()` (`text_encoder.py`) apply same replacement; convert `attention_mask` (B, seq_len) bool/int mask to additive bias (0/-inf) before passing
- [ ] **Task 3:** In `EarlyFusionLayer.forward()` (`fusion_layer.py`) uncomment and implement combined attention mask at lines 204–209; construct combined mask: `[ones(B, n_vision), text_mask]` → convert to additive bias
- [ ] **Task 4:** Write/update unit tests covering AC1–AC5

## Tests Required

- `tests/test_vision_encoder.py` — verify output shape unchanged; verify no manual matmul; CUDA test verifies flash_sdp_enabled
- `tests/test_text_encoder.py` — verify masked positions get -inf bias; verify output shape
- `tests/test_fusion_layer.py` — verify combined mask shape; verify padding tokens do not contribute to cross-attention output
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
