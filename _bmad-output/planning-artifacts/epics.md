---
stepsCompleted: ["validate-prerequisites", "design-epics", "create-stories"]
inputDocuments:
  - "Open-source multi-modal small neural network v1.md"
  - "_bmad-output/planning-artifacts/product-brief.md"
  - "_bmad-output/implementation-artifacts/architecture-2026-03-03.md"
  - "_bmad-output/implementation-artifacts/ux-assessment-2026-03-03.md"
  - "_bmad-output/implementation-artifacts/codebase-review-2026-03-03.md"
  - "_bmad-output/planning-artifacts/implementation-readiness-report-2026-03-03.md"
---

# NeuralMix (multi-modal-neural-network-brownfield) - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for NeuralMix, decomposing the requirements from the PRD, UX Design, and Architecture into implementable stories.

**IMPORTANT CONTEXT — Brownfield Project:**
This is a brownfield project. The codebase is partially through Phase 5 (Phases 1–4 of the 9-phase PRD plan are implemented and Phase 5 is partially configured). Epics in this document cover the remaining Phase 5 work and **Phases 6–9** — completing the functional wiring gaps, executing the training run, building the evaluation framework, producing documentation/UX, and releasing publicly. No epic re-implements work that already exists.

**Existing codebase foundation (do not re-implement):**
- `src/models/` — VisionEncoder (ViT-S), TextEncoder (BERT-Small), FusionLayer (early fusion), DoubleLoopController (LSTM), task Heads — all structurally complete
- `src/training/` — Trainer, optimizer, losses, checkpoint manager, training state — all structurally complete
- `src/data/` — MultiModalDataset, COCOCaptionsDataset, ImageNetDataset, selector — implemented
- `src/integrations/` — WolframAlphaIntegration, WolframKnowledgeInjector — implemented but not wired to training loop
- `tests/` — 20 test files, comprehensive coverage

---

## Requirements Inventory

### Functional Requirements

FR1: The system shall support vision (image) and text (natural language) modalities as inputs.
FR2: The system shall implement an Early Fusion (Type-C) architecture with a shared transformer-based encoder processing both modalities.
FR3: The total model parameter count shall not exceed 500M parameters, with a target of 250M.
FR4: The system shall implement a Double-Loop Learning mechanism: inner loop (standard gradient descent) and outer loop (meta-learning controller adjusting learning strategies, attention patterns, and architectural choices).
FR5: The Double-Loop Controller shall process performance metrics (loss, accuracy) aggregated over N batches via LSTM and output adjustment signals for cross-modal attention weights, layer-wise LR multipliers, regularization strength, and loss function weighting.
FR6: The system shall integrate with the Wolfram Alpha API for fact verification during training, mathematical computation augmentation, symbolic knowledge injection, and evaluation benchmarking.
FR7: The Wolfram Alpha integration shall include local caching (SQLite, 30-day TTL) and the system must operate without Wolfram Alpha if the API is unavailable.
FR8: The system shall train end-to-end on a single consumer GPU (NVIDIA RTX 3060 12GB VRAM) without requiring cloud infrastructure.
FR9: The system shall implement BF16/FP16 Mixed Precision Training (AMP), Gradient Checkpointing, and Gradient Accumulation (4–8 steps) as memory optimization strategies.
FR10: The system shall implement Flash Attention 2 (via `torch.nn.functional.scaled_dot_product_attention`) to meet the 11.5GB peak VRAM target.
FR11: The system shall support YAML-based configuration covering all training and model parameters.
FR12: The system shall implement a streaming data pipeline supporting WebDataset or TFRecord format.
FR13: The system shall implement all modular training components: encoders, fusion, controller, heads, trainer, optimizer, losses, checkpointing.
FR14: The system shall support multi-modal training datasets including COCO, ImageNet-1k subset, Wikipedia/OpenWebText, NQ/TriviaQA/SciQ, GSM8K/MATH subset.
FR15: The system shall evaluate against standard benchmarks: VQA, OK-VQA, CIFAR-100, GSM8K test set, and Wolfram-verified factual accuracy.
FR16: The system shall achieve accuracy targets: CIFAR-100 75–80%, VQA 50–55%, Text Classification 82–85%, Mathematical Reasoning 40–50%, Factual Accuracy vs Wolfram 70–75%.
FR17: Peak VRAM usage during training shall not exceed 11.5GB on the RTX 3060 12GB.
FR18: The system shall achieve inference latency of 50–100ms per sample and throughput of 10–20 samples/second on the target GPU.
FR19: The codebase shall achieve 80%+ code coverage with unit, integration, and performance tests.
FR20: The system shall release pre-trained checkpoints (100M, 250M variants) on Hugging Face Model Hub with model cards.
FR21: The repository shall include a Dockerfile for reproducible environment setup.
FR22: The system shall include 3 Jupyter notebook tutorials, Sphinx API documentation, training guide, and troubleshooting guide.
FR23: The double-loop controller update must add less than 15% computational overhead per epoch; controller forward pass limited to 10ms on target hardware.

### NonFunctional Requirements

NFR1: Training shall achieve 10–20 samples/second on the RTX 3060 12GB; each epoch (10k samples) completes in 30–45 minutes (max 90 min). GPU utilization 80–95%.
NFR2: Total training time shall be 100–200 hours on a single RTX 3060 12GB. Model shall reach 90% of target accuracy within 50% of total training time.
NFR3: Peak system RAM usage shall not exceed 15GB during training.
NFR4: Checkpoint size shall be 1–2GB per checkpoint (model + optimizer state).
NFR5: The system shall implement checkpoint save/load with both PyTorch .pt and safetensors formats. Checkpointing overhead shall be less than 5% of total training time.
NFR6: The Wolfram Alpha API key shall never be hardcoded; must be provided via `${WOLFRAM_API_KEY}` environment variable.
NFR7: Checkpoint loading shall validate paths and guard against external/unsafe checkpoint sources (`allow_external=False` default).
NFR8: The system shall provide comprehensive documentation: architecture diagrams, API docs, training guide, hardware requirements, Wolfram integration guide, troubleshooting guide, and 3 Jupyter notebook tutorials.
NFR9: The system shall run on Linux (Ubuntu 22.04+), Windows 11, and macOS 12+. Stack requires Python 3.10+, CUDA 12.1+, and/or ROCm 5.7+.
NFR10: The project shall be licensed under Apache 2.0.
NFR11: The system shall use YAML-based configuration with environment variable resolution, structured logging, and modular architecture enabling component-level experimentation.
NFR12: For multi-GPU setups, the system shall support ZeRO-2 optimization via DeepSpeed (optional, not required for single-GPU).
NFR13: Wolfram Alpha integration shall not exceed 2,000 queries/day (free tier). Similar queries shall be batched; SQLite cache shall minimize API calls.
NFR14: The project shall target 25/60/100 GitHub stars at 30/60/90 days post-release.

### Additional Requirements

**From Architecture Document (2026-03-03):**
- Flash Attention 2 must be implemented by replacing manual `q @ k.T` with `F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=...)` in both `MultiHeadAttention` (vision encoder) and `TextMultiHeadAttention` (text encoder) — this is not optional, it is required to meet the 11.5GB VRAM ceiling
- Combined attention mask in `EarlyFusionLayer` is commented out (lines 204–209) — must be implemented to prevent text padding tokens from corrupting vision cross-attention
- Double-loop wiring: `train_epoch()` must pass `prev_loss`, `prev_accuracy`, `prev_grad_norm` to `model.forward()`; `AdaptiveLRController.update_lr()` must be called after controller output
- AMP (`torch.amp.autocast`) and `GradScaler` are initialized but not applied in `train_epoch()` — must be wired
- Gradient checkpointing flag exists in config but `torch.utils.checkpoint.checkpoint` is not applied in encoder `forward()` methods
- `SimpleTokenizer` (character-level placeholder) must be replaced with `AutoTokenizer.from_pretrained("bert-base-uncased")` before any accuracy benchmarking
- `evaluation/` module is empty — full benchmark suite required for paper
- `SequenceGenerationHead.forward()` raises `NotImplementedError` for inference — not needed for Phase 1 research targets (classification + VQA)

**From UX Assessment (2026-03-03) — P0 UX Items:**
- Startup banner on `train.py` launch: must print detected device, available VRAM, selected precision mode, active config file path, Wolfram Alpha active/inactive status
- Feature-status logging: each major feature (double-loop, AMP, Wolfram) must emit a clear `[feature] ✓ Active` or `[feature] ⚠ Inactive` log line on startup
- `python train.py --check` dry-run mode: validates hardware and config without starting training
- Runtime `UserWarning` for `SimpleTokenizer`: warn users before any benchmarking run
- `configs/quickstart.yaml`: pre-tuned RTX 3060 config — no Wolfram, no double-loop, basic classification, micro_batch=4, gradient_accumulation=8, bf16

**From Codebase Review (2026-03-03) — Brownfield Integration Context:**
- `train_step()` and `train_epoch()` are disconnected — `MetaLoss` is only called from `train_step()` which is never called from `train_epoch()`
- `validate()` divides by `len(self.val_loader)` without guarding for empty loader
- `collate_fn` uses key `"image"` while `Trainer._normalize_batch()` expects `"images"` — normalized at trainer level but latent confusion source
- `AdaptiveLayerNorm` exists as a standalone class not integrated into encoders/fusion — document decision to skip in v1 or wire it

### FR Coverage Map

FR1: Epic 1 — vision/text modality support already implemented; activated by memory optimization work
FR2: Epic 1 — Early Fusion architecture already implemented; attention mask fix in Story 1.3
FR3: Epic 1 — parameter budget already met; confirmed during VRAM profiling in Story 1.5
FR4: Epic 2 — double-loop wiring to `train_epoch()` in Story 2.1
FR5: Epic 2 — controller outputs (lr_scale, arch_adaptation, meta_loss) wired in Story 2.1–2.2
FR6: **v1.5 scope** — Wolfram Alpha wiring deferred; integration code exists but not activated in v1
FR7: **v1.5 scope** — SQLite caching and graceful fallback deferred to v1.5
FR8: Epic 1 — consumer GPU training target; validated by end-to-end run in Story 1.5
FR9: Epic 1 — AMP wiring (Story 1.1), gradient checkpointing (Story 1.2), gradient accumulation (already configured)
FR10: Epic 1 — Flash Attention 2 replacement in Story 1.3
FR11: Epic 1 — YAML config already implemented; quickstart config in Story 1.6 (UX)
FR12: **v1.5 scope** — WebDataset streaming deferred; JSON/COCO/ImageNet loaders sufficient for v1 training run
FR13: Epic 1 + Epic 2 — all modules structurally complete; wiring completes functional coverage
FR14: Epic 3 — training data pipeline assembly in Story 3.1–3.2; NQ/TriviaQA/GSM8K loaders for Wolfram tasks deferred to v1.5
FR15: Epic 4 — evaluation benchmark suite (VQA, CIFAR-100, GSM8K) in Story 4.1–4.3
FR16: Epic 3 + Epic 4 — accuracy targets measured in Stories 3.3 and 4.2–4.4
FR17: Epic 1 — VRAM ceiling validated in Story 1.5
FR18: Epic 3 — inference benchmarking in Story 3.3
FR19: Epic 1 + Epic 2 — existing 80%+ coverage maintained; regression tests added per story
FR20: Epic 6 — Hugging Face checkpoint upload in Story 6.2
FR21: Epic 6 — Dockerfile in Story 6.3
FR22: Epic 5 — notebooks and documentation in Stories 5.1–5.5
FR23: Epic 2 — overhead profiling and controller timing in Story 2.3

NFR1: Epic 1 — throughput and epoch timing validated in Story 1.5
NFR2: Epic 3 — total training time monitored across Stories 3.1–3.3
NFR3: Epic 1 — system RAM profiled in Story 1.5
NFR4: Epic 1 — checkpoint size verified in Story 1.4
NFR5: Epic 1 — checkpoint save/load already implemented; dual .pt + safetensors verified in Story 1.4
NFR6: Epic 2 — env var pattern already implemented; validated in Story 2.1
NFR7: Epic 1 — safe_load_checkpoint already implemented; verified in Story 1.4
NFR8: Epic 5 — full documentation suite in Stories 5.1–5.6
NFR9: Epic 1 — cross-platform compatibility tested in Story 1.5
NFR10: Apache 2.0 already set; confirmed in Epic 6 repo polish Story 6.1
NFR11: Epic 1 — config system already implemented; quickstart.yaml in Story 1.6
NFR12: v1.5 scope — DDP/DeepSpeed config flag exists, not activated
NFR13: v1.5 scope — Wolfram rate limiting deferred with Wolfram wiring
NFR14: Epic 6 — community launch actions in Stories 6.4–6.5

---

## Epic List

### Epic 1: Achieve Consumer GPU Training Target
A developer with an RTX 3060 12GB can run `python train.py` and complete a full training epoch without OOM errors, with all memory optimizations active and confirmed, and a clear startup signal confirming the environment is ready.
**FRs covered:** FR1, FR2, FR3, FR8, FR9, FR10, FR11, FR13, FR17, FR19
**NFRs covered:** NFR1, NFR3, NFR4, NFR5, NFR7, NFR9, NFR11

### Epic 2: Activate and Validate Meta-Learning Training Loop
A researcher can run a training session where the double-loop meta-learning controller is functionally active — producing measurable lr_scale adjustments and meta_loss contributions — and can confirm the controller is operating by reading structured log output.
**FRs covered:** FR4, FR5, FR23
**NFRs covered:** NFR6

### Epic 3: Execute Full Training Run and Produce Results
A researcher can execute a complete 50-epoch training run on the NeuralMix 250M model, producing converged checkpoints with measured accuracy results on CIFAR-100 classification, and confirmed inference throughput on target hardware.
**FRs covered:** FR14, FR16, FR17, FR18
**NFRs covered:** NFR1, NFR2, NFR3

### Epic 4: Produce Research Benchmark Results and Ablation Study
A researcher can produce publication-quality benchmark results across VQA, CIFAR-100, and mathematical reasoning tasks, and an ablation study comparing training with vs. without the double-loop controller — sufficient to support a NeurIPS 2026 paper submission.
**FRs covered:** FR15, FR16, FR23
**NFRs covered:** NFR1

### Epic 5: Build Developer Onboarding and Documentation
An independent developer can clone NeuralMix, follow `01_getting_started.ipynb`, and run a forward pass end-to-end in under 15 minutes — with clear hardware feedback, a quickstart config, and all major documentation complete.
**FRs covered:** FR22
**NFRs covered:** NFR8, NFR11

### Epic 6: Public Release and Community Launch
NeuralMix v1.0 is publicly available on GitHub and Hugging Face with pre-trained checkpoints, a Dockerfile, model cards, and an active Discord community — and the launch day sequence (GitHub → HF → Reddit → demo video → Discord → X thread) has been executed.
**FRs covered:** FR20, FR21
**NFRs covered:** NFR10, NFR14

---

## Epic 1: Achieve Consumer GPU Training Target

A developer with an RTX 3060 12GB can run `python train.py` and complete a full training epoch without OOM errors, with all memory optimizations active and confirmed, and a clear startup signal confirming the environment is ready.

**FRs covered:** FR1, FR2, FR3, FR8, FR9, FR10, FR11, FR13, FR17, FR19
**NFRs covered:** NFR1, NFR3, NFR4, NFR5, NFR7, NFR9, NFR11

### Story 1.1: Apply BF16 Automatic Mixed Precision to Training Loop

As an independent AI developer,
I want the training loop to use BF16 automatic mixed precision,
So that GPU memory usage is reduced by ~40% and I can train the 250M model without exhausting VRAM.

**Acceptance Criteria:**

**Given** `training.mixed_precision: bf16` is set in `configs/default.yaml`
**When** `train_epoch()` executes a forward + backward pass
**Then** the forward pass runs inside `torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)`
**And** `self.scaler.scale(loss).backward()` is called instead of `loss.backward()`
**And** `self.scaler.step(optimizer)` and `self.scaler.update()` replace the bare `optimizer.step()` call
**And** the existing `self.scaler` (GradScaler) initialized in `Trainer.__init__()` is used — no new scaler is created

**Given** `training.mixed_precision` is set to anything other than `bf16` or `fp16`
**When** `train_epoch()` runs
**Then** the autocast context is skipped and training proceeds in full precision without error

**Given** the training loop runs with AMP active
**When** a validation step executes in `validate()`
**Then** validation also runs inside `torch.amp.autocast` with the same dtype
**And** `self.scaler` is NOT used during validation (no backward pass in validation)

**Given** a training step completes with AMP
**When** `logging_manager.log_metrics()` is called
**Then** the logged loss value is a Python float (not a scaled tensor)

---

### Story 1.2: Apply Gradient Checkpointing to Encoder Forward Passes

As an independent AI developer,
I want gradient checkpointing applied to the transformer blocks in the vision and text encoders,
So that activation memory is reduced by 30–40% during backpropagation, enabling larger effective batch sizes.

**Acceptance Criteria:**

**Given** `training.gradient_checkpointing: true` is set in config
**When** `VisionEncoder.forward()` processes transformer blocks
**Then** `torch.utils.checkpoint.checkpoint` is applied to every 2nd transformer block (blocks at index 1, 3, 5, 7, 9, 11)
**And** the same checkpoint pattern is applied to every 2nd block in `TextEncoder.forward()`
**And** the model's `self.gradient_checkpointing` flag is set to `True` via `model.gradient_checkpointing_enable()` or equivalent in `Trainer.__init__()`

**Given** `training.gradient_checkpointing: false` is set in config
**When** the encoders process transformer blocks
**Then** no checkpointing is applied and forward passes execute normally

**Given** gradient checkpointing is active during training
**When** a backward pass completes
**Then** peak VRAM usage is measurably lower than without checkpointing (verified by `torch.cuda.max_memory_allocated()` comparison in test)
**And** training loss convergence is identical to the non-checkpointed run (same random seed, same data, same hyperparameters)

**Given** gradient checkpointing is enabled
**When** the model is set to `eval()` mode for validation
**Then** checkpointing is automatically disabled during validation (no recomputation overhead)

---

### Story 1.3: Replace Standard Attention with Flash Attention 2 and Fix Fusion Mask

As an independent AI developer,
I want attention computation to use PyTorch's scaled dot-product attention (Flash Attention 2 backend),
So that attention memory is reduced from O(N²) to O(N), making sequences of length 196+512 feasible within 11.5GB VRAM.

**Acceptance Criteria:**

**Given** the `MultiHeadAttention` module in `src/models/vision_encoder.py`
**When** `forward()` computes attention
**Then** the manual `(q @ k.transpose(-2, -1)) * scale` + softmax + `@ v` is replaced with `F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p if self.training else 0.0, is_causal=False)`
**And** `torch.nn.functional` is imported at the top of the file

**Given** the `TextMultiHeadAttention` module in `src/models/text_encoder.py`
**When** `forward()` computes attention
**Then** the same `F.scaled_dot_product_attention` replacement is applied
**And** the `attention_mask` from the text encoder is correctly converted to an additive bias mask (0 for real tokens, -inf for padding) before being passed to `scaled_dot_product_attention`

**Given** `EarlyFusionLayer.forward()` with a batch containing variable-length text inputs
**When** cross-modal attention runs in a `FusionTransformerBlock`
**Then** the combined attention mask (lines 204–209) is uncommented and implemented: vision tokens have no padding mask, text tokens use their `text_mask`; the combined mask prevents text padding positions from being attended to during cross-modal attention
**And** the mask is correctly shaped and broadcast for the multi-head attention computation

**Given** a CUDA device with PyTorch 2.0+ is available
**When** `F.scaled_dot_product_attention` is called
**Then** PyTorch automatically dispatches to the Flash Attention kernel (verified by `torch.backends.cuda.flash_sdp_enabled()` returning `True` in unit test)

**Given** standard CPU or non-Flash-capable hardware
**When** `F.scaled_dot_product_attention` is called
**Then** PyTorch falls back to the math attention kernel without error

---

### Story 1.4: Replace SimpleTokenizer with BERT Tokenizer and Add Checkpoint Validation

As an independent AI developer,
I want the text encoder to use a real BPE tokenizer,
So that text inputs are properly tokenized and accuracy benchmarks reflect genuine model capability rather than character-level encoding artifacts.

**Acceptance Criteria:**

**Given** `AutoTokenizer.from_pretrained("bert-base-uncased")` is called during dataset initialization in `src/data/dataset.py`
**When** `MultiModalDataset.__getitem__()` processes a text sample
**Then** tokenization uses the BERT WordPiece tokenizer (30522 vocab), not `SimpleTokenizer`
**And** the tokenizer is loaded once at dataset construction and reused for all samples (not re-instantiated per sample)

**Given** `SimpleTokenizer` is still present in `src/models/text_encoder.py` as a fallback
**When** `SimpleTokenizer` is instantiated anywhere in the codebase
**Then** a `UserWarning` is emitted: `"SimpleTokenizer is a character-level research placeholder. Replace with AutoTokenizer for any accuracy benchmarking."`

**Given** a training checkpoint was saved with safetensors format
**When** `safe_load_checkpoint()` is called with `allow_external=False` (default)
**Then** the checkpoint is loaded successfully from an internal project path
**And** a checkpoint saved at an external absolute path raises `ValueError` when `allow_external=False`

**Given** `validate()` in `trainer.py` is called with an empty validation loader
**When** the validation loop runs zero batches
**Then** the function returns `{"loss": 0.0, "accuracy": 0.0}` without a ZeroDivisionError
**And** a warning log is emitted: `"Validation loader is empty — skipping validation"`

---

### Story 1.5: Add Startup Banner and Validate Consumer GPU Training in Dry Run

As an independent AI developer,
I want a startup banner printed when training begins and a `--check` dry-run mode,
So that I immediately know whether my hardware, VRAM, and configuration are compatible before committing to a 100+ hour training run.

**Acceptance Criteria:**

**Given** `python train.py` is executed
**When** `Trainer.__init__()` completes initialization
**Then** a startup banner is printed to stdout containing: detected device type (CUDA/MPS/CPU), available VRAM (if CUDA), selected precision mode (bf16/fp32), active configuration file path, gradient checkpointing status (enabled/disabled), and Wolfram Alpha status (`⚠ Inactive — deferred to v1.5`)
**And** double-loop status is printed as `✓ Active — update_frequency=N` when enabled, or `⚠ Inactive` when disabled

**Given** `python train.py --check` is executed
**When** the dry-run mode runs
**Then** the system validates: CUDA availability and VRAM ≥ 10GB, config file loads without error, model forward pass completes on a single synthetic batch without OOM, and tokenizer loads successfully
**And** a `✓ System check passed — ready to train` or `✗ System check failed: [reason]` message is printed
**And** the process exits with code 0 on success or 1 on failure without starting the training loop

**Given** `train.py` is run on a system with less than 10GB available VRAM
**When** the startup banner is printed
**Then** a `⚠ WARNING: Available VRAM (N GB) is below recommended 10GB. Consider reducing micro_batch_size or enabling gradient_checkpointing.` warning is included

**Given** a full training epoch runs on RTX 3060 12GB with AMP + gradient checkpointing + Flash Attention active
**When** the epoch completes
**Then** peak VRAM usage reported by `torch.cuda.max_memory_allocated()` is ≤ 11.5GB
**And** training speed is ≥ 5 samples/second (minimum threshold per PRD Table 4)
**And** the training loss is a finite number (not NaN or Inf)

---

### Story 1.6: Create Quickstart Configuration for RTX 3060 First Run

As an independent AI developer new to NeuralMix,
I want a pre-tuned quickstart configuration file,
So that I can run my first training experiment with safe, validated settings without needing to understand every configuration option.

**Acceptance Criteria:**

**Given** the NeuralMix repository is cloned
**When** a user opens `configs/quickstart.yaml`
**Then** the file exists and contains a complete, runnable configuration with: `mixed_precision: bf16`, `gradient_checkpointing: true`, `micro_batch_size: 4`, `gradient_accumulation: 8`, `double_loop.enabled: false`, `wolfram.enabled: false`, head type set to `classification` with 100 classes (CIFAR-100 compatible)
**And** every configurable section includes a comment tier annotation: `# BEGINNER: safe to change` or `# ADVANCED: change only if you understand the architecture`

**Given** `python train.py --config configs/quickstart.yaml` is executed
**When** the training loop starts
**Then** training runs without error on the quickstart config using the synthetic/toy dataset path
**And** the startup banner confirms all quickstart settings are active

**Given** `default.yaml` is opened by an advanced user
**When** they read the file
**Then** sections for `double_loop`, `wolfram`, `hardware.max_memory`, and attention mechanism settings are annotated with `# ADVANCED` comments
**And** `micro_batch_size`, `max_epochs`, and `data.datasets` are annotated with `# BEGINNER` comments

---

## Epic 2: Activate and Validate Meta-Learning Training Loop

A researcher can run a training session where the double-loop meta-learning controller is functionally active — producing measurable `lr_scale` adjustments and `meta_loss` contributions — and can confirm the controller is operating by reading structured log output.

**FRs covered:** FR4, FR5, FR23
**NFRs covered:** NFR6

### Story 2.1: Wire Double-Loop Controller Inputs to Training Loop

As a researcher studying meta-learning,
I want the double-loop controller to receive live training metrics (loss, accuracy, gradient norm) during training,
So that the LSTM meta-controller can observe training history and produce adaptation signals.

**Acceptance Criteria:**

**Given** `model.double_loop.enabled: true` in config and `Trainer.train_epoch()` is executing
**When** a training step completes
**Then** `prev_loss`, `prev_accuracy`, and `prev_grad_norm` are tracked as scalar tensors across steps
**And** the model's `forward()` call passes these as `current_loss=prev_loss`, `current_accuracy=prev_accuracy`, `gradient_norm=prev_grad_norm`
**And** `prev_loss` is updated from `task_loss.detach()` after each backward pass
**And** `prev_accuracy` is computed as `(logits.argmax(-1) == labels).float().mean().detach()`
**And** `prev_grad_norm` is the L2 gradient norm returned by `GradientClipper` after `unscale_`

**Given** the first step of the first epoch (no previous step exists)
**When** `forward()` is called with double-loop inputs
**Then** `prev_loss`, `prev_accuracy`, and `prev_grad_norm` are initialized to zero tensors of the correct shape
**And** the controller handles zero-initialized inputs without NaN or error

**Given** `model.double_loop.enabled: false` in config
**When** `train_epoch()` runs
**Then** none of the double-loop inputs are passed to `forward()` and no overhead is incurred
**And** the startup banner shows `[double-loop] ⚠ Inactive — set double_loop.enabled: true to activate`

**Given** the `WOLFRAM_API_KEY` environment variable is not set
**When** the startup banner is printed
**Then** the banner shows `[wolfram] ⚠ Inactive — deferred to v1.5` without error

---

### Story 2.2: Wire Double-Loop Controller Outputs to Optimizer LR Adaptation

As a researcher studying meta-learning,
I want the double-loop controller's `lr_scale` output to adaptively modify the optimizer learning rate,
So that the outer loop can slow or accelerate learning in response to observed training dynamics.

**Acceptance Criteria:**

**Given** the double-loop controller is active and `step % update_frequency == 0`
**When** `DoubleLoopController.forward()` returns `meta_info` containing `lr_scale` and `meta_loss`
**Then** `AdaptiveLRController.update_lr(optimizer, meta_info["lr_scale"])` is called immediately after the controller forward pass
**And** the optimizer's learning rate for all parameter groups is scaled by `lr_scale.mean().item()` (clamped to [0.1, 2.0] to prevent degenerate values)
**And** the base learning rate is restored to `inner_lr` at the start of each epoch (LR scaling is per-step, not cumulative)

**Given** `meta_info["meta_loss"]` is returned by the controller
**When** total loss is computed
**Then** `total_loss = task_loss + meta_loss_weight * meta_info["meta_loss"]` where `meta_loss_weight` defaults to 0.1 (configurable via `double_loop.meta_loss_weight`)
**And** `MetaLoss` in `losses.py` is used for this combination — `train_step()` is eliminated as a dead code path and replaced by the wired logic in `train_epoch()`

**Given** the `arch_adaptation` output from the controller
**When** v1 training runs
**Then** a log message is emitted: `[double-loop] arch_adaptation output produced but AdaptiveLayerNorm wiring deferred to v1.5`
**And** `arch_adaptation` is detached and not used in the computation graph (no gradient flows through it in v1)

---

### Story 2.3: Add Double-Loop Feature Status Logging and Overhead Profiling

As a researcher studying meta-learning,
I want structured log output confirming the double-loop controller is active and measuring its overhead,
So that I can verify the controller is functioning and confirm overhead is within the 15% budget specified in the PRD.

**Acceptance Criteria:**

**Given** double-loop is active during training
**When** the controller executes at `step % update_frequency == 0`
**Then** a structured log entry is written at `DEBUG` level containing: step number, `lr_scale` value, `meta_loss` value, controller forward pass duration in ms
**And** every 500 steps a `INFO` level summary is logged: `[double-loop] Steps since last update: N | avg lr_scale: X.XX | avg meta_loss: X.XXXX | controller overhead: X.Xms avg`

**Given** a complete training epoch with double-loop active
**When** the epoch ends
**Then** the epoch summary log includes `double_loop_overhead_pct: X.X%` representing the fraction of total epoch time spent in the controller
**And** if overhead exceeds 15%, a `WARNING` log is emitted: `[double-loop] Controller overhead (X.X%) exceeds 15% budget. Consider increasing update_frequency.`

**Given** a unit test runs the controller forward pass 1000 times on synthetic inputs
**When** timing is measured
**Then** the mean controller forward pass time on CPU is ≤ 50ms (relaxed from PRD's 10ms GPU target to allow CPU CI)
**And** on a CUDA device, mean forward pass time is ≤ 10ms (PRD §2.2.3 constraint)

**Given** the training run completes an epoch with double-loop overhead measured
**When** overhead is ≤ 15%
**Then** no warning is emitted and the epoch summary notes `[double-loop] ✓ Overhead within budget`

---

## Epic 3: Execute Full Training Run and Produce Results

A researcher can execute a complete 50-epoch training run on the NeuralMix 250M model, producing converged checkpoints with measured accuracy on CIFAR-100 classification and confirmed inference throughput on the target hardware.

**FRs covered:** FR14, FR16, FR17, FR18
**NFRs covered:** NFR1, NFR2, NFR3

### Story 3.1: Assemble and Validate Training Data Pipeline

As a researcher training NeuralMix,
I want a validated multi-dataset training pipeline assembled from COCO captions and ImageNet-subset,
So that the model has sufficient multimodal training data to learn cross-modal representations.

**Acceptance Criteria:**

**Given** COCO annotations and images are present at the configured path
**When** `build_dataloaders(config)` is called with `selector.py`
**Then** `COCOCaptionsDataset` loads successfully and `len(train_loader.dataset)` returns the expected sample count
**And** each batch contains `images` (B, 3, 224, 224), `input_ids` (B, seq_len), `attention_mask` (B, seq_len), and `labels` (B,) tensors
**And** the `collate_fn` key mismatch (`"image"` vs `"images"`) is resolved — all datasets produce `"images"` keyed outputs, removing the need for `_normalize_batch()` renaming

**Given** `selector.py` config lists multiple datasets with `enabled: true`
**When** `build_dataloaders()` assembles datasets
**Then** a `ConcatDataset` is returned for the train split containing samples from all enabled datasets
**And** dataset split ratios are validated to sum to 1.0 with a clear `ValueError` if they do not

**Given** a data loader is constructed
**When** 3 consecutive batches are drawn
**Then** no `None` values appear in any tensor field
**And** image tensors are normalized to ImageNet stats (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
**And** text tokens are within the BERT vocabulary range [0, 30521]

**Given** a dataset path does not exist at the configured location
**When** `build_dataloaders()` is called
**Then** a clear `FileNotFoundError` with the missing path is raised (not a silent fallback to dummy data during training)

---

### Story 3.2: Execute 50-Epoch Training Run with Checkpoint Management

As a researcher training NeuralMix,
I want to run a complete 50-epoch training session that saves checkpoints and resumes from interruption,
So that I can train on consumer hardware over multiple sessions without losing progress.

**Acceptance Criteria:**

**Given** all Epic 1 and Epic 2 stories are complete and a training dataset is configured
**When** `python train.py --config configs/default.yaml` is run for 50 epochs
**Then** training completes without OOM error on RTX 3060 12GB
**And** a checkpoint is saved every 5 epochs as `checkpoint_NNNN.pt` and `checkpoint_NNNN.safetensors`
**And** `best.pt` is updated whenever validation loss improves
**And** `latest.pt` is updated after every epoch

**Given** training is interrupted (process killed) at epoch N
**When** `python train.py --resume` is run
**Then** training resumes from `latest.pt` at epoch N+1 with the same optimizer state, scheduler state, and training metrics
**And** the resumed run produces identical loss values to an uninterrupted run (given the same data order)

**Given** `checkpoint_manager.max_checkpoints: 5` in config
**When** more than 5 numbered checkpoints exist
**Then** the oldest numbered checkpoint is deleted, keeping only the 5 most recent plus `best.pt` and `latest.pt`

**Given** a training epoch completes
**When** metrics are logged
**Then** train loss, train accuracy, validation loss, validation accuracy, learning rate, and epoch duration are all logged to the metrics file and (if configured) to W&B

---

### Story 3.3: Measure Inference Throughput and CIFAR-100 Accuracy

As a researcher reporting NeuralMix results,
I want to measure inference latency, throughput, and CIFAR-100 classification accuracy from a trained checkpoint,
So that I can report concrete performance numbers in the paper and README.

**Acceptance Criteria:**

**Given** a trained `best.pt` checkpoint
**When** `python inference.py --checkpoint best.pt --benchmark` is run on the RTX 3060
**Then** the script reports: samples/second throughput at batch size 1, mean latency per sample in ms, and peak VRAM usage during inference
**And** throughput is ≥ 10 samples/second (PRD §5.3 minimum)
**And** latency is ≤ 200ms per sample (relaxed from 100ms if throughput target is met)

**Given** a trained checkpoint and the CIFAR-100 test split
**When** `inference.py` runs classification evaluation
**Then** top-1 accuracy on CIFAR-100 test set is reported as a percentage
**And** results are written to `_bmad-output/implementation-artifacts/training-results-{date}.md` including: epoch count, final train loss, validation loss, CIFAR-100 top-1 accuracy, throughput, VRAM peak

**Given** a model loaded for inference
**When** `model.eval()` and `torch.inference_mode()` are active
**Then** peak VRAM during inference is ≤ 8GB (PRD §5.3 — 6–8GB target)

---

## Epic 4: Produce Research Benchmark Results and Ablation Study

A researcher can produce publication-quality benchmark results across VQA, CIFAR-100, and mathematical reasoning tasks, and an ablation study comparing training with vs. without the double-loop controller — sufficient to support a NeurIPS 2026 paper submission.

**FRs covered:** FR15, FR16, FR23
**NFRs covered:** NFR1

### Story 4.1: Implement CIFAR-100 and VQA Evaluation Framework

As a researcher publishing NeuralMix results,
I want a reusable evaluation framework for CIFAR-100 and VQA benchmarks,
So that I can produce standardized, reproducible accuracy numbers for the paper.

**Acceptance Criteria:**

**Given** a trained checkpoint and CIFAR-100 test split
**When** `CIFAR100Evaluator.evaluate(checkpoint_path)` is called from `src/evaluation/`
**Then** it returns a dict with `top1_accuracy`, `top5_accuracy`, `eval_samples`, `eval_time_seconds`
**And** evaluation runs without modifying model weights
**And** results are deterministic across runs with the same checkpoint and test split

**Given** a trained checkpoint and VQA v2 validation split (or OK-VQA subset)
**When** `VQAEvaluator.evaluate(checkpoint_path, split="val")` is called
**Then** it returns `vqa_accuracy` (VQA evaluation metric — answer presence in ground truth list)
**And** the evaluator handles multi-answer ground truth (VQA has 10 annotators per question)

**Given** `src/evaluation/__init__.py`
**When** it is imported
**Then** `CIFAR100Evaluator`, `VQAEvaluator`, and `AblationRunner` are all importable without error
**And** the module is no longer empty

---

### Story 4.2: Run Double-Loop Ablation Study

As a researcher publishing NeuralMix results,
I want an automated ablation study comparing training with and without the double-loop controller,
So that I can quantify the meta-learning contribution and report the improvement delta in the paper.

**Acceptance Criteria:**

**Given** `AblationRunner` is initialized with a base config
**When** `AblationRunner.run(conditions=["with_double_loop", "without_double_loop"], epochs=10, seed=42)` is called
**Then** two training runs are executed sequentially: one with `double_loop.enabled: true` and one with `double_loop.enabled: false`
**And** both runs use identical hyperparameters, random seeds, data order, and hardware settings
**And** both runs produce checkpoints in separate output directories

**Given** both ablation runs complete
**When** `AblationRunner.report()` is called
**Then** it returns a dict containing: CIFAR-100 accuracy for each condition, final validation loss for each condition, accuracy delta (with_double_loop - without_double_loop), and overhead percentage for the double-loop condition
**And** the report is written to `_bmad-output/implementation-artifacts/ablation-results-{date}.md`

**Given** the ablation results
**When** `accuracy_delta >= 0.05` (≥5% improvement, PRD §9.1 target)
**Then** the report includes `[double-loop] ✓ Meets PRD improvement target of 5–10%`
**When** `accuracy_delta < 0.05`
**Then** the report includes `[double-loop] ⚠ Below PRD improvement target — results recorded for paper`

---

### Story 4.3: Produce Mathematical Reasoning Benchmark Results

As a researcher publishing NeuralMix results,
I want to evaluate NeuralMix on a mathematical reasoning subset,
So that I can report accuracy on the GSM8K benchmark and assess the model's symbolic reasoning capability without Wolfram Alpha active.

**Acceptance Criteria:**

**Given** a trained checkpoint and the GSM8K test subset (500 problems)
**When** `MathReasoningEvaluator.evaluate(checkpoint_path, dataset="gsm8k_subset")` is called
**Then** it returns `exact_match_accuracy` and `partial_credit_accuracy` on the subset
**And** results include a baseline comparison: model accuracy vs. random baseline (PRD Table 6: baseline 30%, target 40–50%)

**Given** the evaluation completes
**When** results are written to the benchmark report
**Then** the report clearly states: "Wolfram Alpha auxiliary supervision is deferred to v1.5 — these results reflect the base model without Wolfram grounding"
**And** this framing is consistent with the v1.5 scope decision documented in `epics.md`

**Given** the complete benchmark suite (CIFAR-100, VQA, GSM8K) has run
**When** `BenchmarkReporter.compile_results()` is called
**Then** a single `_bmad-output/implementation-artifacts/benchmark-results-{date}.md` is produced containing all benchmark tables formatted for direct inclusion in the paper's experimental section

---

## Epic 5: Build Developer Onboarding and Documentation

An independent developer can clone NeuralMix, follow `01_getting_started.ipynb`, and run a forward pass end-to-end in under 15 minutes — with clear hardware feedback, a quickstart config, and all major documentation complete.

**FRs covered:** FR22
**NFRs covered:** NFR8, NFR11

### Story 5.1: Build Getting Started Notebook (01_getting_started.ipynb)

As an independent AI developer new to NeuralMix,
I want a complete getting-started notebook,
So that I can go from `git clone` to a running forward pass in under 15 minutes with no prior knowledge of the codebase.

**Acceptance Criteria:**

**Given** a developer clones the repo and opens `notebooks/01_getting_started.ipynb`
**When** they run all cells top to bottom
**Then** the notebook completes without error on a system with CUDA ≥ 10GB VRAM
**And** the final cell produces a model forward pass with real (not dummy) output tensors and prints their shapes

**Given** the notebook is run
**When** each major section executes
**Then** the sections cover in order: hardware check (VRAM, CUDA version), environment validation (`--check` mode output), loading `quickstart.yaml`, instantiating the model, loading a sample image + text, running a forward pass, and interpreting the output
**And** inline markdown cells explain each step with concrete numbers (e.g., "This model has ~230M parameters — here's how to count them")

**Given** a developer without a local GPU opens the notebook in Google Colab
**When** they run the Colab-specific setup cell (pip installs, mount drive)
**Then** the notebook runs end-to-end using the Colab T4 GPU without modification
**And** a Colab badge link is present at the top of the notebook README section

---

### Story 5.2: Build Training Notebook (02_training.ipynb)

As an independent AI developer,
I want a training walkthrough notebook,
So that I can understand how to configure and run a training session, observe real VRAM graphs, and interpret training metrics.

**Acceptance Criteria:**

**Given** a developer opens `notebooks/02_training.ipynb`
**When** they run the notebook on an RTX 3060 12GB
**Then** the notebook runs a 3-epoch training loop on a small dataset subset (1000 samples) and displays: live VRAM usage graph (matplotlib), loss curve, accuracy curve, and double-loop `lr_scale` curve (if enabled)

**Given** the training section of the notebook
**When** it executes
**Then** it demonstrates both `quickstart.yaml` (beginner) and `default.yaml` (advanced) configurations with inline commentary on the differences
**And** a cell explicitly demonstrates what happens with `double_loop.enabled: false` vs `true` (shows startup banner difference)

**Given** the notebook completes a short training run
**When** checkpoint saving is demonstrated
**Then** the notebook shows the checkpoint directory structure (`best.pt`, `latest.pt`, `checkpoint_NNNN.pt`) and explains each file's purpose

---

### Story 5.3: Build Evaluation Notebook (03_evaluation.ipynb)

As a researcher using NeuralMix,
I want an evaluation walkthrough notebook,
So that I can load a trained checkpoint and reproduce benchmark results interactively.

**Acceptance Criteria:**

**Given** a researcher opens `notebooks/03_evaluation.ipynb` with a trained checkpoint available
**When** they run all cells
**Then** the notebook demonstrates: loading a checkpoint, running CIFAR-100 evaluation, running the double-loop ablation comparison (abbreviated, 3 epochs), and interpreting the ablation delta

**Given** the ablation section of the notebook
**When** it runs
**Then** results are displayed as a formatted table: condition, accuracy, delta, overhead %
**And** an inline markdown cell explains how this table maps to the paper's experimental section

---

### Story 5.4: Write README and Hardware Compatibility Documentation

As an independent AI developer discovering NeuralMix,
I want a complete README with a hardware compatibility table at the top,
So that I can immediately determine if my GPU is supported before cloning the repository.

**Acceptance Criteria:**

**Given** a developer lands on the NeuralMix GitHub repository page
**When** they read the README
**Then** the first visible content (above the fold) is: project name, one-line value proposition, and a hardware compatibility table listing each tested GPU, measured peak VRAM during training, relative training speed, and support status

**Given** the README hardware table
**When** it is read
**Then** it includes at minimum: RTX 3060 12GB (✅ fully tested), RTX 3070 8GB (⚠️ limited — 8GB may OOM at full batch size), RTX 4070 12GB (✅), RTX 3080 12GB (✅), RX 6700 XT 12GB (✅ ROCm), with measured VRAM numbers from Story 1.5 profiling

**Given** the README body
**When** it is read
**Then** it contains: quickstart install instructions (`pip install -r requirements.txt`), first training run command (`python train.py --config configs/quickstart.yaml`), link to `01_getting_started.ipynb`, link to `02_training.ipynb`, architecture overview diagram (from architecture doc §1.1 ASCII diagram), known limitations section listing: double-loop wired (v1), Wolfram Alpha (v1.5), WebDataset (v1.5)

---

### Story 5.5: Generate Sphinx API Documentation

As a developer integrating NeuralMix components,
I want generated API documentation for all public modules,
So that I can understand module interfaces without reading source code.

**Acceptance Criteria:**

**Given** `docs/` directory contains a Sphinx configuration (`conf.py`, `index.rst`)
**When** `make html` is run in the `docs/` directory
**Then** HTML documentation is generated without errors covering all public classes and functions in: `src/models/`, `src/training/`, `src/data/`, `src/integrations/`, `src/evaluation/`, `src/utils/`

**Given** the generated docs
**When** the `MultiModalModel`, `Trainer`, `DoubleLoopController`, and `FusionLayer` pages are viewed
**Then** each page shows: class docstring, `__init__` parameters with types, `forward()` signature with input/output shapes, and at least one usage example

**Given** a new public function is added to any `src/` module
**When** it has a docstring
**Then** it appears automatically in the generated docs without manual RST updates (autodoc configuration)

---

### Story 5.6: Write Training Guide and Troubleshooting Guide

As an independent AI developer training NeuralMix,
I want a training guide and troubleshooting guide in the `docs/` folder,
So that I can resolve common issues (OOM, convergence failure, tokenizer errors) without needing to post on Discord.

**Acceptance Criteria:**

**Given** `docs/TRAINING_GUIDE.md` exists
**When** a developer reads it
**Then** it covers: recommended hardware, step-by-step first training run, how to adjust batch size and gradient accumulation for different VRAM sizes, how to monitor training with W&B, how to resume from checkpoint, and how to evaluate a trained model

**Given** `docs/TROUBLESHOOTING.md` exists
**When** a developer reads it
**Then** it covers at minimum: CUDA OOM (with specific config changes to try), NaN loss (learning rate too high, AMP numerical instability), slow training (GPU utilization < 70% — data loading bottleneck), tokenizer warning (how to confirm `AutoTokenizer` is active), and double-loop inactive warning (how to enable)
**And** each issue has a "Symptom", "Cause", and "Fix" structure

---

## Epic 6: Public Release and Community Launch

NeuralMix v1.0 is publicly available on GitHub and Hugging Face with pre-trained checkpoints, a Dockerfile, model cards, and an active Discord community — and the launch day sequence (GitHub → HF → Reddit → demo video → Discord → X thread) has been executed.

**FRs covered:** FR20, FR21
**NFRs covered:** NFR10, NFR14

### Story 6.1: Repository Polish and v1.0 Release Tagging

As the NeuralMix project maintainer,
I want the repository to be fully polished and tagged as v1.0.0,
So that the public release is professional, discoverable, and sets the right first impression.

**Acceptance Criteria:**

**Given** the repository at release time
**When** a developer visits the GitHub repo
**Then** `README.md` is complete (Story 5.4), `LICENSE` is Apache 2.0, `CONTRIBUTING.md` exists with contribution guidelines and code of conduct reference, `.github/ISSUE_TEMPLATE/` contains at minimum a bug report and feature request template, and `CHANGELOG.md` has a v1.0.0 entry with all major features listed

**Given** the v1.0.0 git tag is created
**When** `git tag -a v1.0.0` is run
**Then** the tagged release on GitHub includes: full release notes (features, known limitations, hardware compatibility), links to Hugging Face model hub, link to Discord server, and the demo video URL

**Given** the `known-limitations` section of the README
**When** it is read
**Then** it clearly states: "Wolfram Alpha auxiliary supervision is implemented (v1.5)", "WebDataset streaming is not yet implemented (v1.5)", "Auto-regressive text generation is not yet implemented (v1.5)", "Multi-GPU DDP training is not activated (v1.5)"

---

### Story 6.2: Upload Pre-Trained Checkpoints to Hugging Face

As an independent AI developer who cannot run a 100-hour training run,
I want to download pre-trained NeuralMix checkpoints from Hugging Face,
So that I can experiment with the model immediately without training from scratch.

**Acceptance Criteria:**

**Given** the Hugging Face organization page for NeuralMix
**When** a developer visits it
**Then** at minimum the 100M and 250M parameter checkpoints are available for download as safetensors files
**And** each checkpoint has a model card containing: model architecture summary, training hardware (RTX 3060 12GB), training data description, benchmark results table (CIFAR-100, VQA accuracy), known limitations, and usage example with code snippet

**Given** a developer runs `from huggingface_hub import hf_hub_download` to fetch a checkpoint
**When** they load it with `safe_load_checkpoint(path)`
**Then** the model loads without error and produces valid inference outputs
**And** the model card usage example code runs without modification

**Given** INT8 quantization is applied to the 250M checkpoint
**When** the quantized model is loaded for inference
**Then** peak VRAM is ≤ 4GB and throughput is ≥ 15 samples/second
**And** CIFAR-100 accuracy degradation vs. fp32 is ≤ 3% (acceptable quantization loss)

---

### Story 6.3: Create Dockerfile for Reproducible Environment

As an independent AI developer,
I want a Dockerfile that sets up the complete NeuralMix training environment,
So that I can reproduce any published results exactly without manually managing CUDA, PyTorch, and dependency versions.

**Acceptance Criteria:**

**Given** the NeuralMix repository root
**When** `docker build -t neuralmix:v1.0 .` is run
**Then** the Docker image builds successfully using NVIDIA NGC PyTorch base image (`nvcr.io/nvidia/pytorch:24.01-py3` or equivalent)
**And** all `requirements.txt` dependencies are installed in the image

**Given** the built Docker image
**When** `docker run --gpus all neuralmix:v1.0 python train.py --check` is run
**Then** the `--check` dry run completes successfully and reports CUDA available
**And** the container exits with code 0

**Given** `docker-compose.yml` exists
**When** `docker compose up training` is run
**Then** the training service starts with the GPU device mounted, the project directory bind-mounted, and `configs/default.yaml` as the active config

---

### Story 6.4: Execute Community Launch Sequence

As the NeuralMix project maintainer,
I want to execute the launch day sequence across all channels,
So that the initial community response and GitHub star count establish momentum for ongoing adoption.

**Acceptance Criteria:**

**Given** the repository is public and checkpoints are on Hugging Face
**When** the launch day sequence executes
**Then** the following actions are completed in order within a 4-hour window: GitHub repo goes public with v1.0.0 tag → Hugging Face model page is published → Reddit posts submitted to r/LocalLLaMA and r/MachineLearning with the approved messaging ("Train a full multimodal model on your RTX 3060. No AWS bill.") → demo video published to YouTube/X showing end-to-end training on RTX 3060 → Discord server opens with invite link in all posts → X/Twitter thread posted

**Given** the Discord server opens
**When** a new member joins
**Then** they are greeted by a pinned #welcome message containing: quickstart guide link, `01_getting_started.ipynb` link, #help channel direction, and RTX 3060 setup link
**And** #start-here channel has a pinned message: "New here? → [Quickstart] | Have a GPU? → [Training Guide] | No GPU? → [Colab Notebook] | Questions? → #help"

---

### Story 6.5: Monitor and Respond to Launch Week Community Activity

As the NeuralMix project maintainer,
I want to monitor and respond to all GitHub issues and Reddit comments in the first week post-launch,
So that early adopters get fast responses and the community sees the project is actively maintained.

**Acceptance Criteria:**

**Given** the repository is public and launch posts are live
**When** a GitHub Issue is opened
**Then** it receives a response within 24 hours (acknowledgment if not an immediate fix)
**And** issues are triaged with labels: `bug`, `question`, `feature-request`, `good-first-issue`

**Given** launch Reddit posts are live
**When** a comment asks a question answerable from the documentation
**Then** the response links directly to the relevant doc section (`TRAINING_GUIDE.md`, `TROUBLESHOOTING.md`, or notebook)

**Given** the first 7 days post-launch
**When** GitHub star count and clone count are measured
**Then** progress vs. PRD §9.2 targets (25 stars / 100 clones at 30 days) is recorded in a `_bmad-output/implementation-artifacts/launch-metrics-{date}.md` tracking document
**And** the tracking document notes any issues or friction patterns reported by early adopters for v1.1 prioritization
