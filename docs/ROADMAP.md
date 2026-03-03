# NeuralMix — Version Roadmap

**Date:** 2026-03-03
**Project:** NeuralMix (multi-modal-neural-network-brownfield)

NeuralMix follows a deliberate three-stage progression aligned to the edge IoT deployment use case that inspired it: *"What if we built the model where it needs to live?"*

## Table of Contents

- [v1 — Experimental Research Platform](#v1--experimental-research-platform-current)
- [v1.5 — Production-Ready Progression](#v15--production-ready-progression)
- [v2 — Edge / IoT Production Target](#v2--edge--iot-production-target)
- [Version Progression Gates](#version-progression-gates)
- [Success Metrics](#success-metrics)

---

## v1 — Experimental Research Platform (current)

**Goal:** Prove the architecture works. Establish the research community.

| Dimension | Target |
|-----------|--------|
| Parameters | 250M (500M max) |
| Hardware | RTX 3060 12GB (single GPU) |
| Primary purpose | Experimentation, ablation, research publication |
| Distribution | GitHub + Hugging Face (open weights) |
| Wolfram Alpha | Optional auxiliary signal (not wired to training loss in v1) |
| Audience | Independent developers, researchers |

### v1 Scope

**In scope:**

- Vision (image) + text modality support with early cross-modal fusion
- Double-loop meta-learning controller (inner loop: AdamW gradient descent; outer loop: LSTM policy adaptation)
- BF16 AMP + Flash Attention 2 + gradient checkpointing to fit 12GB VRAM
- YAML-based configuration with environment variable resolution
- COCO captions + ImageNet-1k training datasets
- CIFAR-100 and VQA evaluation benchmarks
- Ablation study: with vs. without double-loop controller
- Research paper (arXiv preprint + NeurIPS 2026 submission target)
- Apache 2.0 license

**Out of scope for v1:**

- Production inference API or deployment-ready service
- Fine-tuning wrapper for existing large models
- Audio / video modalities
- Wolfram Alpha wiring to training loss (deferred to v1.5)
- WebDataset streaming pipeline
- Dockerfile
- Jupyter notebook content (shells exist; content is Epic 5)

### v1 Accuracy Targets

| Benchmark | Target |
|-----------|--------|
| CIFAR-100 image classification | 75–80% |
| VQA (VQA v2 / OK-VQA) | 50–55% |
| Text classification | 82–85% |
| Mathematical reasoning (GSM8K subset) | 40–50% |
| Factual accuracy vs. Wolfram Alpha | 70–75% |

### v1 Success Criteria

- Academic preprint posted to arXiv as a preprint or 50+ citations/views within 12 months
- 5+ independent result reproductions within 6 months
- Double-loop ablation validated: ≥5% accuracy improvement documented
- 100+ GitHub stars in 90 days post-release
- Stable training run completing on RTX 3060 12GB without OOM

### v1 Epic Plan

| Epic | Goal | Status |
|------|------|--------|
| Epic 1 | Achieve Consumer GPU Training Target (AMP, Flash Attention 2, gradient checkpointing) | 🔲 Pending |
| Epic 2 | Activate and Validate Meta-Learning Training Loop | 🔲 Pending |
| Epic 3 | Execute Full Training Run and Produce Results | 🔲 Pending |
| Epic 4 | Produce Research Benchmark Results and Ablation Study | 🔲 Pending |
| Epic 5 | Build Developer Onboarding and Documentation | 🔲 Pending |
| Epic 6 | Public Release and Community Launch | 🔲 Pending |

---

## v1.5 — Production-Ready Progression

**Goal:** Harden v1 into a reliable base suitable for developer tools and prototypes.

| Dimension | Target |
|-----------|--------|
| Architecture | v1 architecture + stability improvements from community feedback |
| Inference optimization | INT8 quantization, ONNX export, TorchScript |
| API surface | Stable Python API, HuggingFace-compatible interface |
| Documentation | Production-grade API docs, deployment guides |
| Community | Active contributor base; v1.5 built on community PRs |
| Audience | Small orgs, startup prototyping, developer tools |

### v1.5 Key Features

- **Wolfram Alpha training wiring** — wire `WolframKnowledgeInjector` into `train_epoch()` with auxiliary loss weight 0.15; implement SQLite TTL cache (30-day) to replace in-memory daily counter
- **INT8 quantization** — post-training quantization pipeline for inference deployment
- **ONNX export** — `torch.onnx.export` pipeline with dynamic axes and opset 18; validated on ONNX Runtime with CPU, OpenVINO, and DirectML execution providers
- **TorchScript** — `torch.jit.script` export for production serving
- **HuggingFace-compatible interface** — `from_pretrained()` / `push_to_hub()` pattern
- **DDP support** — multi-GPU training via `torch.nn.parallel.DistributedDataParallel` for users with multiple GPUs
- **WebDataset streaming pipeline** — efficient large-scale training beyond 1M samples
- **Startup UX improvements** — startup banner with device/VRAM/config info; feature-status logging; `--check` dry-run mode

### v1.5 Success Criteria

- 5+ community-built applications using NeuralMix as a base
- 500+ Hugging Face model downloads
- Stable INT8 inference validated on ONNX Runtime (CPU + GPU)
- Paper submitted to NeurIPS 2026 (deadline ~May 2026)

---

## v2 — Edge / IoT Production Target

**Goal:** Purpose-built for the deployment environment that inspired the project.

| Dimension | Target |
|-----------|--------|
| Architecture | v1.5 + architectural search for edge-optimal variants |
| Target hardware | ARM Cortex-M, NVIDIA Jetson, Raspberry Pi 5, ESP32-S3 class |
| Size variants | 10M, 50M, 100M parameter tiers |
| Double-loop | Full online adaptation for distribution shift at device level |
| Modalities | Vision + text + optional audio (sensor fusion) |
| Deployment | ONNX Runtime, TFLite, EdgeTPU compatible |
| Audience | Edge AI practitioners, IoT platform developers |

### v2 Key Features

- **Tiered parameter variants** — 10M (microcontroller), 50M (edge device), 100M (Jetson class)
- **Full online adaptation** — double-loop controller running on-device to adapt to sensor drift and environmental changes without cloud retraining
- **Audio modality** — sensor fusion: vision + text + optional audio for IoT applications
- **EdgeTPU / TFLite export** — deploy on Google Coral Edge TPU (4 TOPS) and similar inference accelerators
- **Structured pruning** — reduce v1 250M model to target tiers without full retraining

### v2 Success Criteria

- Documented deployment on ≥2 edge hardware platforms (e.g., Jetson Nano + Raspberry Pi 5)
- 3+ enterprise pilot evaluations for IoT use cases
- Online adaptation demonstrated on real-world distribution shift scenario

---

## Version Progression Gates

| Gate | Criteria for Advancing |
|------|------------------------|
| v1 → v1.5 | Paper submitted + 10+ community contributors + stable training on 3+ GPU models |
| v1.5 → v2 | 500+ model downloads + 5+ apps built on v1.5 + documented INT8 inference on Jetson |

---

## Success Metrics

### Research Success (Primary — v1)

| Metric | Target |
|--------|--------|
| Preprint / paper submission | 1 paper |
| Paper citations / views | 50+ |
| Independent result reproductions | 5+ |
| Double-loop ablation validated | ≥5% accuracy improvement documented |

### Community Success (v1 targets)

| Metric | 30 Days | 60 Days | 90 Days |
|--------|---------|---------|---------|
| GitHub Stars | 25 | 60 | 100 |
| Discord Members | 50 | 150 | 300 |
| Community PRs Merged | 1 | 5 | 10 |
| HF Model Downloads | 100 | 300 | 500 |

---

## Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| License | **Apache 2.0** | Patent grant protects community from contributor patent claims; preferred for commercial-adjacent open-source |
| Publication venue | **arXiv preprint + NeurIPS 2026** | arXiv at v1 release starts citation clock immediately; NeurIPS 2026 deadline ~May 2026 aligns with ~Week 23 release |
| Wolfram Alpha API tier | **Free tier** (2,000 req/day) | Sufficient for v1 experimental use with SQLite caching planned for v1.5 |
| Wolfram wiring scope | **v1.5** | Core v1 research claim is double-loop meta-learning; Wolfram is an enhancement, not the primary contribution |
| v1.5 advancement trigger | **Milestone-based** | Advances when: paper submitted + 10+ community contributors + stable training on 3+ GPU models |
