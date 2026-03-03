# NeuralMix — Architecture Document
**Prepared by:** Winston (🏗️ Architect, BMAD Method v6)
**Date:** 2026-03-03
**Version:** 1.0
**Status:** Approved — Tim_D (2026-03-03)
**Source Documents:**
- PRD v1.0 — `Open-source multi-modal small neural network v1.md`
- Product Brief v1.0 — `_bmad-output/planning-artifacts/product-brief.md`
- Codebase Review — `_bmad-output/implementation-artifacts/codebase-review-2026-03-03.md`

---

## 1. Architecture Overview

NeuralMix is a **250M parameter multimodal neural network** combining a Vision Transformer (ViT) encoder, a BERT-style text encoder, an early fusion layer, a double-loop meta-learning controller, and configurable task heads. The system is designed to train end-to-end on a single consumer GPU (12GB VRAM) using BF16 AMP, gradient checkpointing, and Flash Attention 2.

### 1.1 High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NeuralMix Forward Pass                       │
│                                                                     │
│  Image Input (B, 3, 224, 224)     Text Input (B, seq_len)          │
│         │                                  │                        │
│         ▼                                  ▼                        │
│  ┌─────────────┐                  ┌──────────────┐                 │
│  │   Vision    │                  │    Text      │                 │
│  │   Encoder   │                  │   Encoder    │                 │
│  │  (ViT-S)    │                  │  (BERT-S)    │                 │
│  │  ~50M param │                  │  ~50M param  │                 │
│  └──────┬──────┘                  └──────┬───────┘                 │
│         │ (B, 196, 512)                  │ (B, seq_len, 512)       │
│         │ + CLS (B, 512)                 │ + CLS (B, 512)          │
│         └──────────────┬─────────────────┘                         │
│                        ▼                                            │
│              ┌──────────────────┐                                   │
│              │  Early Fusion    │                                   │
│              │     Layer        │                                   │
│              │  (6 xformer blks)│                                   │
│              │  ~50M param      │                                   │
│              └────────┬─────────┘                                   │
│                       │ (B, 196+seq_len, 512) → mean pool           │
│                       │ (B, 512)                                    │
│                       ▼                                             │
│              ┌──────────────────┐    ┌────────────────────────┐    │
│              │   Task Head(s)   │    │  Double-Loop           │    │
│              │  Classification  │    │  Controller (LSTM)     │    │
│              │  Contrastive     │    │  ~25M param            │    │
│              │  Multi-task      │◄───│  lr_scale, arch_adapt  │    │
│              └────────┬─────────┘    └────────────────────────┘    │
│                       │                                             │
│                  logits / outputs                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Parameter Budget

| Component | Target | Maximum | Current Implementation |
|-----------|--------|---------|------------------------|
| Vision Encoder | 50M | 100M | ~50M (12L × 8H × 512D) |
| Text Encoder | 50M | 100M | ~50M (12L × 8H × 512D) |
| Fusion Layer | 50M | 100M | ~50M (6L × 8H × 512D) |
| Task Heads | 25M | 50M | ~2–5M (classification default) |
| Double-Loop Controller | 25M | 50M | ~10–15M (LSTM 2L × 256H) |
| **Total** | **200–250M** | **500M** | **~180–230M** |

---

## 2. Component Architecture

### 2.1 Vision Encoder (`src/models/vision_encoder.py`)

**Architecture:** Vision Transformer Small (ViT-S)

```
Input: (B, 3, 224, 224)
  │
  ▼ PatchEmbedding (Conv2d 16×16 stride 16)
  │ → (B, 196, 512)   [196 = (224/16)²]
  │
  ▼ Prepend CLS token → (B, 197, 512)
  │
  ▼ Add learnable position embeddings (197 × 512)
  │
  ▼ Dropout
  │
  ▼ 12× TransformerBlock
  │   ├─ LayerNorm (pre-norm)
  │   ├─ MultiHeadAttention (8 heads, head_dim=64)
  │   ├─ Residual
  │   ├─ LayerNorm (pre-norm)
  │   ├─ MLP (512 → 2048 → 512, GELU)
  │   └─ Residual
  │
  ▼ LayerNorm
  │
  ▼ Split: CLS token (B, 512) + patch tokens (B, 196, 512)
```

**Key design decisions:**
- Pre-norm style (LayerNorm before attention) — more stable training than post-norm
- CLS token for classification tasks; patch tokens for cross-modal attention
- Truncated-normal initialization (std=0.02) — standard ViT init

**Outstanding work:**
- Replace standard `q @ k.T` attention with **Flash Attention 2** (`torch.nn.functional.scaled_dot_product_attention` with `is_causal=False`) — required for 11.5GB VRAM target
- Implement `get_attention_maps()` for interpretability figures

---

### 2.2 Text Encoder (`src/models/text_encoder.py`)

**Architecture:** BERT-Small

```
Input: input_ids (B, seq_len), attention_mask (B, seq_len), token_type_ids (B, seq_len)
  │
  ▼ TextEmbedding
  │   ├─ Token embeddings (30522 vocab × 512)
  │   ├─ Position embeddings (512 positions × 512)
  │   ├─ Segment embeddings (2 × 512)
  │   └─ LayerNorm + Dropout
  │   → (B, seq_len, 512)
  │
  ▼ 12× TextTransformerBlock
  │   ├─ LayerNorm (pre-norm)
  │   ├─ TextMultiHeadAttention (8 heads, separate Q/K/V projections)
  │   ├─ Residual
  │   ├─ LayerNorm (pre-norm)
  │   ├─ MLP (512 → 2048 → 512, GELU)
  │   └─ Residual
  │
  ▼ LayerNorm
  │
  ▼ CLS pooler: Linear(512, 512) + Tanh → CLS token (B, 512)
  ▼ Full sequence: (B, seq_len, 512)
```

**Key design decisions:**
- BERT vocabulary size (30522) for compatibility with `bert-base-uncased` tokenizer
- Segment embeddings (token_type_ids) for sentence-pair tasks
- Separate Q/K/V projections vs. fused QKV in VisionEncoder — both are correct; different implementation styles

**Outstanding work:**
- Replace `SimpleTokenizer` (character-level, research placeholder) with `AutoTokenizer.from_pretrained("bert-base-uncased")` — already conditionally loaded in `dataset.py`

---

### 2.3 Fusion Layer (`src/models/fusion_layer.py`)

**Architecture:** Early Fusion with Alternating Cross-Modal Attention

```
Inputs:
  vision_features: (B, 196, 512)
  text_features:   (B, seq_len, 512)
  text_mask:       (B, seq_len)

  │
  ▼ Linear projection: vision_proj (512→512), text_proj (512→512)
  │
  ▼ Add modality-type embeddings (learnable, 2 × 512)
  │   vision: + modality_embed[0]
  │   text:   + modality_embed[1]
  │
  ▼ Concatenate: (B, 196+seq_len, 512)
  │
  ▼ 6× FusionTransformerBlock (alternating):
  │   Even blocks (i=0,2,4): vision attends to text
  │   │   ├─ Self-attention on vision portion
  │   │   ├─ Cross-attention: vision queries text
  │   │   └─ MLP
  │   Odd blocks (i=1,3,5): text attends to vision
  │   │   ├─ Self-attention on text portion
  │   │   ├─ Cross-attention: text queries vision
  │   │   └─ MLP
  │
  ▼ LayerNorm
  │
  ▼ Mean pool over full sequence → (B, 512)
```

**Key design decisions:**
- Early fusion chosen over late fusion: lower parameter overhead, stronger cross-modal interaction from early layers
- Alternating attention strategy: each modality progressively integrates information from the other
- Modality-type embeddings distinguish vision tokens from text tokens post-concatenation

**Outstanding work:**
- Implement combined attention mask for early fusion — vision tokens are always real (no padding), text tokens have padding; combined mask prevents attended-to padding from polluting vision's cross-attention
- Consider Flash Attention 2 here as well — this layer operates on sequences of length 196+seq_len which can be memory-intensive

---

### 2.4 Double-Loop Controller (`src/models/double_loop_controller.py`)

**Architecture:** LSTM Meta-Controller

```
Inputs (every N steps, where N=update_frequency):
  model_features:  (B, 512)  — pooled fusion output, detached from computation graph
  loss:            (B, 1)    — current training loss
  accuracy:        (B, 1)    — current batch accuracy
  gradient_norm:   (B, 1)    — L2 norm of all model gradients

  │
  ▼ Concatenate → (B, 515)
  │
  ▼ Unsqueeze seq dim → (B, 1, 515)
  │
  ▼ 2-layer LSTM (hidden_dim=256)
  │   Maintains rolling hidden state across training steps
  │   → lstm_out: (B, 1, 256)
  │
  ▼ Squeeze → (B, 256)
  │
  ├─▶ lr_modulator: Linear(256,64) → ReLU → Linear(64,1) → Sigmoid
  │       → lr_scale: (B, 1) ∈ [0, 1]
  │
  ├─▶ arch_predictor: Linear(256,128) → ReLU → Linear(128,64) → Tanh
  │       → arch_adaptation: (B, 64)
  │
  └─▶ meta_loss_predictor: Linear(256,64) → ReLU → Linear(64,1)
          → meta_loss: (B, 1)
```

**Inner loop (standard gradient descent):**
- AdamW optimizer on task loss (CrossEntropy default)
- Learning rate: `inner_lr` (default 3e-4)
- Update every batch

**Outer loop (meta-learning):**
- LSTM controller processes training history every `update_frequency` steps (default 100)
- Outputs `lr_scale` → scales optimizer learning rate via `AdaptiveLRController`
- Outputs `arch_adaptation` → intended for `AdaptiveLayerNorm` parameter modulation (not yet connected)
- Outputs `meta_loss` → added to task loss via `MetaLoss` (weight: 0.1 default)

**How meta-feedback flows:**

```
Batch N:
  forward(images, text) → logits → task_loss
  backward(task_loss) → compute gradient_norm
  ↓
  if step % update_frequency == 0:
    controller(pooled_features.detach(), task_loss, accuracy, gradient_norm)
    → lr_scale, arch_adaptation, meta_loss
    → adaptive_lr_controller.update_lr(optimizer, lr_scale)
    → total_loss = task_loss + 0.1 * meta_loss
    → backward(total_loss)
```

**Outstanding work:**
- Wire double-loop inputs into `Trainer.train_epoch()` — currently not called
- Connect `arch_adaptation` signal to `AdaptiveLayerNorm` instances in encoders (or document decision to not use it in v1)
- Connect `AdaptiveLRController.update_lr()` call after controller forward pass

---

### 2.5 Task Heads (`src/models/heads.py`)

Five head types, configured via YAML:

| Head Type | Config key | Primary use |
|-----------|-----------|-------------|
| `ClassificationHead` | `classification` | CIFAR-100, ImageNet |
| `RegressionHead` | `regression` | Continuous output tasks |
| `MultiLabelHead` | `multilabel` | Multi-label classification (COCO) |
| `ContrastiveHead` | `contrastive` | Image-text matching (CLIP-style) |
| `SequenceGenerationHead` | `generation` | Captioning (Phase 2+) |
| `MultiTaskHead` | `multitask` | Combines multiple heads |

**Default config:** Classification with 1000 classes (ImageNet-like).

**For research targets:**
- Phase 1 evaluation: use `ClassificationHead` (CIFAR-100 target: 75–80%)
- Contrastive pre-training: use `ContrastiveHead` (image-text matching)
- VQA evaluation: use `ClassificationHead` with VQA answer vocabulary (VQA target: 50–55%)

---

### 2.6 Wolfram Alpha Integration (`src/integrations/`)

```
Training Data Batch
  │
  │  [if math/factual content detected]
  ▼
WolframKnowledgeInjector.inject_knowledge()
  │
  ▼ _extract_math_expressions() — regex-based pattern detection
  │
  ▼ WolframAlphaIntegration.query() — with daily limit check + caching
  │
  ▼ extract_mathematical_result() → ground truth string
  │
  ▼ Validation signal → auxiliary loss term (weight: 0.15)
  │
  └─▶ total_loss = task_loss + 0.15 * wolfram_validation_loss
```

**Status:** Implemented but not connected to training loop. See §4.2 for wiring plan.

---

## 3. Training System Architecture

### 3.1 Component Interaction

```
Trainer
  │
  ├── DeviceManager          — CUDA/NPU/MPS/CPU detection & configuration
  ├── LoggingManager         — File logger + MetricsLogger + WandbLogger
  ├── TrainingState          — epoch, step, best_val_loss state
  ├── TrainingComponentsFactory
  │     ├── create_criterion()          → CrossEntropyLoss (default)
  │     ├── create_meta_criterion()     → MetaLoss (if double_loop enabled)
  │     ├── create_optimizer()          → AdamW with param group separation
  │     ├── create_scheduler()          → CosineAnnealingLR with linear warmup
  │     ├── create_gradient_clipper()   → GradientClipper (max_norm=1.0)
  │     └── create_adaptive_lr_controller() → AdaptiveLRController
  │
  └── CheckpointManager      — latest.pt + best.pt + checkpoint_NNNN.pt + safetensors
```

### 3.2 Training Loop (Target State — Phase 6)

```python
for epoch in range(max_epochs):
    for batch in train_loader:
        optimizer.zero_grad()

        # AMP forward pass
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(
                images=batch['images'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                current_loss=prev_loss,         # double-loop input
                current_accuracy=prev_accuracy, # double-loop input
                gradient_norm=prev_grad_norm,   # double-loop input
            )
            logits = outputs['logits']
            task_loss = criterion(logits, batch['labels'])
            meta_info = outputs.get('meta_info')
            total_loss = meta_criterion(task_loss, meta_info)

        # AMP backward
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = grad_clipper(model.parameters())
        scaler.step(optimizer)
        scaler.update()

        # Update double-loop LR adaptation
        if meta_info and adaptive_lr:
            adaptive_lr.update_lr(optimizer, meta_info['lr_scale'])

        # Track for next step's double-loop inputs
        prev_loss = task_loss.detach()
        prev_accuracy = (logits.argmax(-1) == batch['labels']).float().mean().detach()
        prev_grad_norm = torch.tensor(grad_norm)

    scheduler.step()  # epoch-level update for plateau; step-level for cosine
```

### 3.3 Memory Budget (12GB VRAM target)

| Component | VRAM Estimate |
|-----------|---------------|
| Model parameters (BF16) | ~460MB |
| Optimizer states (FP32) | ~920MB |
| Activations (micro_batch=4, grad_checkpoint) | ~4–6GB |
| Gradients (BF16) | ~460MB |
| Data buffers | ~500MB |
| Flash Attention overhead | ~200MB |
| **Total (estimated)** | **~7–8.5GB** |
| **Peak (with accumulation)** | **~10–11GB** |
| **VRAM ceiling** | **11.5GB** |

Memory optimizations required:
1. BF16 AMP — halves activation and gradient memory
2. Gradient checkpointing — recomputes activations backward; reduces activation memory ~30–40%
3. Flash Attention 2 — reduces attention memory from O(N²) to O(N) — essential at seq_len 196+512
4. Micro-batch size 4 with gradient accumulation 8 — effective batch 32

---

## 4. Architecture Decision Records (ADRs)

### ADR-001: Early Fusion (Type-C) over Late Fusion

**Decision:** Use early fusion with alternating cross-modal attention blocks.

**Rationale:**
- Late fusion (fuse after independent encoding) requires two full forward passes before interaction — less parameter-efficient
- Early fusion allows each modality to attend to the other from the fusion layer onward, richer cross-modal representation
- Parameter count is comparable: early fusion's 6-layer transformer uses ~50M params (same as a late-fusion projection network)
- Competitive with CLIP's cross-attention at lower total parameter cost

**Trade-off accepted:** Early fusion is less interpretable per-modality (features are entangled earlier). For Phase 1 research purposes, interpretability is secondary to performance.

---

### ADR-002: Custom ViT + BERT over Pre-trained Backbones

**Decision:** Implement ViT-S and BERT-Small from scratch rather than loading pre-trained HuggingFace weights.

**Rationale:**
- PRD goal is training from scratch on consumer hardware — loading CLIP/BERT weights would defeat the research purpose
- Custom implementation allows architectural modifications (e.g., adding `AdaptiveLayerNorm` for double-loop)
- Demonstrates full trainability without external dependencies on model hubs

**Trade-off accepted:** Training from scratch requires more compute and data than fine-tuning. 100–200 hours on RTX 3060 is acceptable per PRD.

---

### ADR-003: LSTM-based Meta-Controller over Transformer-based

**Decision:** Use a 2-layer LSTM for the double-loop meta-controller.

**Rationale:**
- LSTM maintains a rolling hidden state naturally — meta-learning requires memory of training history across many steps
- Transformers on long sequences of training metrics would be computationally expensive
- LSTM at 256 hidden units is ~10–15M parameters — fits within the 25M controller budget
- Update frequency of 100 steps means the LSTM processes one input every 100 inner loop steps — low overhead

**Trade-off accepted:** LSTM cannot attend back to arbitrary points in training history (only recent hidden state). This is acceptable for v1.

---

### ADR-004: Apache 2.0 License

**Decision:** Apache 2.0.

**Rationale:** Patent grant protects community contributors from patent claims. More permissive for commercial-adjacent use than GPL/LGPL. Preferred by enterprise adopters.

---

### ADR-005: Wolfram Alpha as Optional Auxiliary Signal

**Decision:** Wolfram Alpha validation loss weighted at 0.15 (15% of total), with graceful fallback when API unavailable.

**Rationale:**
- Over-weighting Wolfram signal risks teaching the model to optimise for Wolfram queries rather than multimodal understanding
- 15% is sufficient to provide a measurable accuracy lift on factual/mathematical tasks
- SQLite caching (30-day TTL) makes API unavailability a minor issue during long training runs

---

### ADR-006: Single-GPU Training (No DDP in v1)

**Decision:** Target single RTX 3060 12GB. DDP config flag exists but disabled.

**Rationale:**
- Target user profile is independent developer with one GPU
- DDP adds implementation complexity disproportionate to research value at 250M params
- v1.5 can introduce DDP for multi-GPU fine-tuning

---

## 5. Module Interface Contracts

These define the expected input/output contracts for each module — required for story acceptance criteria.

### 5.1 `VisionEncoder.forward(x)`

```
Input:  x: torch.Tensor  shape=(B, 3, H, W)  dtype=float32|bfloat16
Output: Tuple[
    cls_token: torch.Tensor  shape=(B, hidden_dim),
    patch_tokens: torch.Tensor  shape=(B, n_patches, hidden_dim)
]
```

### 5.2 `TextEncoder.forward(input_ids, attention_mask, token_type_ids)`

```
Input:
    input_ids:       torch.Tensor  shape=(B, seq_len)  dtype=int64
    attention_mask:  torch.Tensor  shape=(B, seq_len)  dtype=int64  (1=real, 0=pad)
    token_type_ids:  Optional[torch.Tensor]  shape=(B, seq_len)
Output: Tuple[
    cls_token: torch.Tensor  shape=(B, hidden_dim),
    sequence_output: torch.Tensor  shape=(B, seq_len, hidden_dim)
]
```

### 5.3 `FusionLayer.forward(...)`

```
Input:
    vision_features: torch.Tensor  shape=(B, n_patches, hidden_dim)
    text_features:   torch.Tensor  shape=(B, seq_len, hidden_dim)
    text_mask:       Optional[torch.Tensor]  shape=(B, seq_len)
Output: Tuple[
    fused_features: torch.Tensor  shape=(B, n_patches+seq_len, hidden_dim)  [early]
                                   OR  shape=(B, hidden_dim)  [late]
    vision_seq_len: Optional[int]
]
```

### 5.4 `DoubleLoopController.forward(...)`

```
Input:
    model_features:  torch.Tensor  shape=(B, model_hidden_dim)  [detached]
    loss:            torch.Tensor  shape=(B,) or scalar
    accuracy:        torch.Tensor  shape=(B,) or scalar
    gradient_norm:   torch.Tensor  shape=(B,) or scalar
Output: Dict[
    "lr_scale":           torch.Tensor  shape=(B, 1)  ∈ [0, 1]
    "arch_adaptation":    torch.Tensor  shape=(B, 64)
    "meta_loss":          torch.Tensor  shape=(B, 1)
    "should_update_meta": bool
]
```

### 5.5 `MultiModalModel.forward(...)`

```
Input:
    images:           Optional[torch.Tensor]  shape=(B, 3, H, W)
    input_ids:        Optional[torch.Tensor]  shape=(B, seq_len)
    attention_mask:   Optional[torch.Tensor]  shape=(B, seq_len)
    token_type_ids:   Optional[torch.Tensor]  shape=(B, seq_len)
    current_loss:     Optional[torch.Tensor]  — for double-loop
    current_accuracy: Optional[torch.Tensor]  — for double-loop
    gradient_norm:    Optional[torch.Tensor]  — for double-loop
    return_features:  bool  (default False)
    task_name:        Optional[str]  — for MultiTaskHead routing
Output: Dict[
    "logits":    torch.Tensor  — task predictions
    "meta_info": Optional[Dict]  — double-loop controller outputs
    "features":  Optional[Dict]  — if return_features=True
]
```

### 5.6 `Trainer.train_epoch(epoch)`

```
Input:   epoch: int
Output:  Dict["loss": float, "accuracy": float]
Side effects: optimizer step, scheduler step, checkpoint save
```

---

## 6. Data Architecture

### 6.1 Dataset Registry

| Type key | Class | Use |
|----------|-------|-----|
| `multimodal` | `MultiModalDataset` | Generic JSON-annotated vision+text |
| `coco_captions` | `COCOCaptionsDataset` | MS-COCO caption pairs |
| `imagenet` | `ImageNetDataset` | ImageNet-style class directories |

### 6.2 Multi-Dataset Assembly (`selector.py`)

```yaml
data:
  datasets:
    - name: multimodal_core
      type: multimodal
      data_dir: ./data/multimodal
      splits: {train: 0.8, val: 0.1, test: 0.1}
      enabled: true
    - name: captions_aux
      type: coco_captions
      root: ./data/coco/images
      ann_file: ./data/coco/annotations/captions_train2017.json
      splits: {train: 1.0}
      use_in: [train]
      enabled: true
```

`build_dataloaders()` assembles `ConcatDataset` per split from all enabled, applicable entries.

### 6.3 Target Training Data (PRD §4.1)

| Dataset | Size | Split | Status |
|---------|------|-------|--------|
| Conceptual Captions / COCO | 100k–500k multimodal pairs | Primary train | Loader implemented (COCO) |
| ImageNet-1k subset | 50k–100k images | Vision pretraining | Loader implemented |
| Wikipedia / OpenWebText subset | 50k–100k texts | Text pretraining | Not implemented |
| Natural Questions / TriviaQA / SciQ | 10k–25k | Wolfram validation | Not implemented |
| GSM8K / MATH subset | 5k–15k | Math reasoning | Not implemented |

---

## 7. Configuration Reference

All configuration via `configs/default.yaml`. Key sections:

```yaml
model:
  vision_encoder:
    hidden_dim: 512    # Shared hidden dimension across all components
    num_layers: 12     # Depth of ViT
    num_heads: 8       # Attention heads
  text_encoder:
    vocab_size: 30522  # BERT vocabulary
    max_seq_length: 512
  fusion:
    type: early        # early | late
    num_layers: 6      # Fusion transformer depth
  double_loop:
    controller_type: lstm
    update_frequency: 100  # Inner loop steps between outer loop updates
    meta_lr: 1e-5          # Outer loop LR (slower than inner)

training:
  inner_lr: 3e-4         # Inner loop LR
  max_epochs: 50
  mixed_precision: bf16  # BF16 AMP
  gradient_checkpointing: true
  micro_batch_size: 4
  gradient_accumulation: 8  # Effective batch = 32

hardware:
  device: auto           # cuda | npu | mps | cpu | auto
  max_memory: "11GB"
```

---

## 8. Security and Safety

| Concern | Implementation |
|---------|---------------|
| Checkpoint loading | `safe_load_checkpoint()` with path validation; `allow_external=False` default |
| Safetensors | Dual-save alongside `.pt`; preferred for loading |
| Wolfram API key | `${WOLFRAM_API_KEY}` env var — never hardcoded |
| External paths | `allow_external` config gate in `Trainer.load_checkpoint()` |

---

## 9. Phase-by-Phase Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Environment setup, base architecture, tests | ✅ Complete |
| 2 | Vision encoder, text encoder, fusion layer | ✅ Complete |
| 3 | Double-loop controller (structural) | ✅ Structural complete |
| 3b | Double-loop wired to training loop | ❌ Not done |
| 4 | Wolfram Alpha integration (structural) | ✅ Structural complete |
| 4b | Wolfram wired to training loss | ❌ Not done |
| 5 | BF16 AMP (configured) | ⚠️ Config exists, not applied |
| 5b | Flash Attention 2 | ❌ Not done |
| 5c | Gradient checkpointing (flag exists) | ⚠️ Flag exists, not applied |
| 6 | Full training run | 🔲 Not started |
| 7 | Evaluation / benchmarks | ❌ Module empty |
| 8 | Documentation + tutorials | 🔲 Not started |
| 9 | Public release | 🔲 Not started |

---

## 10. Immediate Implementation Priorities

Listed in order for Phase 6 readiness:

1. **Apply AMP in training loop** — wrap forward pass in `torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)`, wrap backward with `scaler.scale(loss).backward()`, call `scaler.step()` and `scaler.update()`
2. **Apply gradient checkpointing** — in `VisionEncoder` and `TextEncoder`, apply `torch.utils.checkpoint.checkpoint` every 2nd transformer block when `self.gradient_checkpointing=True`
3. **Wire double-loop to `train_epoch()`** — track `prev_loss`, `prev_accuracy`, `prev_grad_norm`; pass to model forward; call `adaptive_lr.update_lr()` after controller output
4. **Replace Flash Attention** — in `MultiHeadAttention` and `TextMultiHeadAttention`, replace manual `q @ k.T` with `F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.1 if training else 0.0)`
5. **Wire Wolfram to loss** — instantiate `WolframKnowledgeInjector` in `Trainer.__init__()` (gated on `wolfram.api_key` presence); add validation loss to `total_loss` in `train_step()`
6. **Implement `evaluation/` module** — `VQAEvaluator`, `CIFAR100Evaluator`, `AblationRunner`

---

*Prepared by Winston (🏗️ Architect, BMAD Method v6.0.3) | 2026-03-03*
