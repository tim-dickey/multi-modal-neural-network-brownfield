---
title: "NeuralMix — Product Brief"
version: "1.0"
date: "2026-02-28"
author: "Mary (Business Analyst, BMAD)"
status: "Draft — Ready for PM Review"
stepsCompleted: ["discovery", "problem-framing", "user-definition", "scope", "version-roadmap", "success-criteria"]
---

# NeuralMix — Product Brief

**Project Codename:** NeuralMix  
**Full Name:** Open-Source Multi-Modal Small Neural Network with Double-Loop Learning  
**Version:** 1.0  
**Date:** 2026-02-28  
**Prepared by:** Mary, Business Analyst (BMAD)  
**Prepared for:** Tim_D  

---

## 1. The Origin Story

While advising a client deploying edge IoT devices, a recurring and costly problem became visible: every time hardware capabilities shifted or sensor configurations changed, teams were forced back into expensive cloud-based retraining pipelines. The conventional wisdom — retrain an existing large model on new data — created a dependency on infrastructure that edge environments fundamentally cannot support.

The insight: **if you could build a capable multimodal model from scratch on the same class of hardware the edge device runs near, you eliminate the retraining-on-cloud dependency at root.** Not fine-tuning. Not distillation. Actual training — from scratch — on a consumer GPU that anyone can own.

NeuralMix is the answer to that question: *"What if we built the model where it needs to live?"*

---

## 2. Problem Statement

### 2.1 The Core Problem

Multimodal AI model development is structurally inaccessible to independent developers, edge AI practitioners, and researchers without institutional compute budgets. Every mature open-source multimodal model — LLaVA (7B), BLIP-2 (3.9B), InstructBLIP (8B+) — requires 24–40GB+ VRAM for training, locking out the RTX 3060/3070/6700 XT class of consumer GPU that costs under $400.

The consequence: developers targeting edge IoT deployments are forced to train in the cloud and deploy to the edge — a workflow that is expensive, slow, and architecturally decoupled from the reality of their target environment.

### 2.2 Why Retraining Isn't the Answer

Retraining adapts existing model weights to new distributions. It preserves the original model's architectural assumptions, which were designed for high-memory, high-compute environments. When the target environment is an edge device with constrained memory and inference budgets, you are perpetually fighting the model's architecture rather than working with it.

Building from scratch on the target hardware class forces architectural discipline from day one: every design decision — parameter count, attention mechanism, fusion strategy — must be justifiable at 12GB VRAM. The result is a model that is *native* to its operating environment, not ported to it.

### 2.3 The Meta-Learning Angle

Edge IoT deployments face a unique challenge: **distribution shift at the device level**. Sensor drift, environmental changes, and hardware variation mean models need to adapt structurally — not just update weights. Double-loop learning (inner loop: parameter updates; outer loop: policy/strategy adaptation) is the architectural mechanism that enables this. NeuralMix embeds it as a first-class feature, not an afterthought.

---

## 3. Target Users

### 3.1 Primary User: Independent AI Developer / Edge AI Practitioner

**Profile:**
- Owns a consumer GPU (RTX 3060, 3070 Ti, 4060 Ti 16GB, RX 6700 XT — 12GB VRAM class)
- Interest in multimodal AI, edge deployment, or research-grade experimentation
- No institutional cloud budget; works independently or at a small org
- Motivated by building and understanding, not just using pre-trained black boxes

**Current workflow (without NeuralMix):**
1. Download a large pre-trained model (LLaVA, BLIP-2)
2. Discover they cannot retrain it locally — VRAM wall
3. Either pay for cloud compute ($500–2,000/month) or restrict to inference-only use
4. Cannot modify architecture, study training dynamics, or run controlled ablations

**With NeuralMix:**
1. Clone repo, configure `default.yaml`
2. Run `train.py` on local RTX 3060
3. Observe full training dynamics, modify architecture components, run ablations
4. Publish results, contribute to community

**Jobs to be Done:**
- *"I want to understand how multimodal models actually work by training one myself"*
- *"I want to experiment with meta-learning on a model I can actually afford to run"*
- *"I want a research-grade platform for my edge AI work that doesn't require AWS"*

### 3.2 Secondary Users

| User Type | Use Case | What They Get from NeuralMix |
|-----------|----------|------------------------------|
| Academic researchers | Meta-learning / multimodal research | First OSS implementation of double-loop learning in a multimodal model |
| Graduate students | Coursework, thesis projects | Reproducible, trainable reference architecture with documentation |
| Educators | Teaching advanced ML concepts | A model students can actually run and modify |
| Small orgs / startups | Prototyping edge AI applications | Production progression path (v1 → v1.5 → v2) |
| Edge IoT practitioners | Deploy-at-the-edge pipelines | v2 roadmap target: prod-ready edge/IoT optimized model |

---

## 4. Solution Overview

### 4.1 What NeuralMix Is

NeuralMix is a **250M parameter multimodal neural network** (vision + text) designed to be trained end-to-end on a single consumer GPU with 12GB VRAM. It incorporates:

- **Early fusion architecture** (Type-C): lower parameter overhead than late/deep fusion
- **Double-loop learning controller**: inner loop (weight updates) + outer loop (policy adaptation) — the key architectural differentiator
- **Optional Wolfram Alpha integration**: auxiliary supervision signal for factual/mathematical tasks; graceful fallback when unavailable
- **Consumer-hardware-first design**: BF16 AMP, gradient checkpointing, Flash Attention 2 — all required to hit the 12GB VRAM ceiling, not optional extras

### 4.2 What NeuralMix Is Not (v1 Scope)

- ❌ A production-inference API or deployment-ready service (that is v1.5/v2)
- ❌ A fine-tuning wrapper for existing large models
- ❌ An audio/video model (vision + text only in v1)
- ❌ A replacement for GPT-4, Gemini, or any frontier model
- ❌ Cloud-dependent (by design and constraint)

### 4.3 Key Value Proposition

> *"The first open-source multimodal model you can actually train at home — 250M parameters, consumer GPU ready, with built-in meta-learning. No cloud account required."*

---

## 5. Version Roadmap

NeuralMix follows a deliberate three-stage progression aligned to the IoT + edge deployment use case that inspired it.

### v1 — Experimental Research Platform (Current Scope)
**Goal:** Prove the architecture works. Establish the research community.

| Dimension | Target |
|-----------|--------|
| Parameters | 250M (500M max) |
| Hardware | RTX 3060 12GB (single GPU) |
| Training time | 100–200 hours |
| Primary purpose | Experimentation, ablation, research publication |
| Distribution | GitHub + Hugging Face (open weights) |
| Wolfram Alpha | Optional auxiliary signal |
| Audience | Independent developers, researchers |

**Success signal:** Academic paper accepted or preprint with 50+ citations/views; 100+ GitHub stars in 90 days.

---

### v1.5 — Production-Ready Progression
**Goal:** Harden v1 into a reliable base suitable for developer tools and prototypes.

| Dimension | Target |
|-----------|--------|
| Architecture | v1 architecture + stability improvements from community feedback |
| Inference optimization | INT8 quantization, ONNX export, TorchScript |
| API surface | Stable Python API, HuggingFace-compatible interface |
| Documentation | Production-grade API docs, deployment guides |
| Community | Active contributor base, v1.5 built on community PRs |
| Audience | Small orgs, startup prototyping, developer tools |

**Success signal:** 5+ community-built applications using NeuralMix as a base; 500+ model downloads.

---

### v2 — Edge / IoT Production Target
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

**Success signal:** Documented deployment on ≥2 edge hardware platforms; 3+ enterprise pilot evaluations.

---

## 6. Competitive Positioning

### 6.1 The VRAM Wall — Why No Existing Model Solves This

| Model | Params | Min VRAM to Train | Consumer GPU Trainable | Double-Loop | Open Source |
|-------|--------|-------------------|------------------------|-------------|-------------|
| LLaVA-7B | 7B | 40GB+ | ❌ | ❌ | ✅ |
| BLIP-2 (OPT-6.7B) | 3.9B | 24GB+ | ❌ | ❌ | ✅ |
| InstructBLIP | 8B+ | 40GB+ | ❌ | ❌ | ✅ |
| CLIP + TinyLLaMA | ~1.5B | 16–24GB | ⚠️ Limited | ❌ | ✅ |
| Phi-3-mini (text only) | 3.8B | 16GB | ⚠️ Inference only | ❌ | ✅ |
| MobileViT (vision only) | 5–30M | 4GB | ✅ | ❌ | ✅ |
| **NeuralMix v1** | **250M** | **12GB** | **✅** | **✅** | **✅** |

### 6.2 Research Differentiation

NeuralMix is the **only open-source multimodal model** at any parameter scale that implements double-loop (meta-learning) as a first-class architectural feature integrated into the training pipeline. MAML and Reptile implementations exist as research code but are not embedded in a production-grade multimodal training framework. This is the primary research contribution.

### 6.3 Positioning Statement

NeuralMix occupies a unique position: **small enough to train locally, sophisticated enough to publish about.** It is not competing with frontier models on benchmark leaderboards — it is competing with the absence of a trainable, research-grade multimodal platform for independent developers.

---

## 7. Success Metrics

### 7.1 Research Success (Primary — v1)

| Metric | Target | Timeframe |
|--------|--------|-----------|
| Preprint / paper submission | 1 paper | Within 6 months of v1 release |
| Paper citations / views | 50+ | Within 12 months |
| Independent result reproductions | 5+ | Within 6 months |
| Double-loop ablation validated | ≥5% accuracy improvement documented | At v1 release |

### 7.2 Community Success (Bonus — v1)

| Metric | 30 Days | 60 Days | 90 Days |
|--------|---------|---------|---------|
| GitHub Stars | 25 | 60 | 100 |
| Discord Members | 50 | 150 | 300 |
| Community PRs Merged | 1 | 5 | 10 |
| HF Model Downloads | 100 | 300 | 500 |

### 7.3 Version Progression Gates

| Gate | Criteria for Advancing |
|------|------------------------|
| v1 → v1.5 | Paper submitted + 10+ community contributors + stable training on 3+ GPU models |
| v1.5 → v2 | 500+ downloads + 5+ apps built on v1.5 + documented INT8 inference on Jetson |

---

## 8. Risks and Open Questions

### 8.1 Key Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Double-loop controller adds >15% training overhead | Medium | High | Ablation studies in Phase 3; controller size capped at 25M params |
| Community adoption slower than projected | Medium | Medium | Early access briefings for ML YouTubers; Colab fallback for non-GPU users |
| Wolfram Alpha API costs exceed budget | Low | Low | Positioned as optional; free-tier sufficient for validation tasks |
| Paper not accepted to target venue | Medium | Medium | Preprint on arXiv ensures public record regardless of venue acceptance |
| v2 edge targets require architectural rethink | Medium | High | v1.5 stability phase allows architectural research before v2 commitment |

### 8.2 Open Decisions (Require Resolution Before Development Kickoff)

| Decision | Options | Recommendation | Owner |
|----------|---------|----------------|-------|
| License | Apache 2.0 vs MIT | Apache 2.0 (patent protection for commercial-adjacent community) | Tim_D |
| Target publication venue | NeurIPS, ICML, ICLR, arXiv preprint | arXiv preprint first (fastest to community) + conference submission | Tim_D |
| Wolfram Alpha tier | Free (2,000 req/day) vs Paid (10,000 req/day) | Free for v1 development; upgrade if validation tasks demand it | Tim_D |
| v1.5 timeline trigger | Calendar-based vs milestone-based | Milestone-based (10+ contributors + paper submitted) | Tim_D |

---

## 9. Next Steps

| Action | Owner | Timing |
|--------|-------|--------|
| Resolve open license decision (§8.2) | Tim_D | Before Week 1 |
| Confirm paper venue target | Tim_D | Before Week 6 |
| Complete PRD stakeholder sign-off | Technical Lead, ML Researcher, Systems Engineer | Before Week 1 |
| Create architecture document | Winston (Architect, BMAD) | Week 1–2 |
| Set up GitHub repository and Hugging Face org | Tim_D | Week 1 |
| Create epics and stories from PRD | Bob (Scrum Master, BMAD) | Week 2 |

---

## 10. Reference

- **PRD:** [Open-source multi-modal small neural network v1.md](../../Open-source%20multi-modal%20small%20neural%20network%20v1.md)
- **PRD Assessment:** [PRD_Assessment_Summary.md](../PRD_Assessment_Summary.md)
- **BMAD Config:** [_bmad/bmm/config.yaml](../../_bmad/bmm/config.yaml)

---

*Prepared by Mary (📊 Business Analyst, BMAD Method v6.0.3) | 2026-02-28*
