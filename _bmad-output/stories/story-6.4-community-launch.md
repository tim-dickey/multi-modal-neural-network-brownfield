---
id: "6.4"
epic: 6
title: "Execute Community Launch Sequence"
status: "ready"
priority: "high"
estimate: "M"
assignee: ""
sprint: 6
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files: []
---

# Story 6.4: Execute Community Launch Sequence

## User Story

As the NeuralMix project maintainer,
I want to execute the launch day sequence across all channels,
So that the initial community response and GitHub star count establish momentum for ongoing adoption.

## Context

**Epic:** Epic 6 — Public Release and Community Launch
**Brownfield context:** This is an operational story — no code changes. All prior stories (1.1–6.3) must be complete before launch day. This story is the execution checklist for launch day itself.

> ⚠️ **Dependency:** ALL prior stories must be complete. Story 6.2 (HF checkpoints) and Story 6.3 (Dockerfile) must be done.

## Acceptance Criteria

**AC1 — Launch day sequence completed within 4 hours:**
**Given** the repository is public and checkpoints are on Hugging Face
**When** the launch day sequence executes
**Then** the following actions are completed in order within a 4-hour window:
1. GitHub repo goes public with v1.0.0 tag
2. Hugging Face model page is published
3. Reddit posts submitted to r/LocalLLaMA and r/MachineLearning with approved messaging: *"Train a full multimodal model on your RTX 3060. No AWS bill."*
4. Demo video published to YouTube/X showing end-to-end training on RTX 3060
5. Discord server opens with invite link in all posts
6. X/Twitter thread posted

**AC2 — Discord welcome structure:**
**Given** the Discord server opens
**When** a new member joins
**Then** they are greeted by a pinned #welcome message containing: quickstart guide link, `01_getting_started.ipynb` link, #help channel direction, and RTX 3060 setup link
**And** #start-here channel has a pinned message: "New here? → [Quickstart] | Have a GPU? → [Training Guide] | No GPU? → [Colab Notebook] | Questions? → #help"

## Tasks

- [ ] **Task 1:** Prepare Reddit post drafts for r/LocalLLaMA and r/MachineLearning with approved messaging
- [ ] **Task 2:** Record demo video (RTX 3060, end-to-end `python train.py --check` → first epoch)
- [ ] **Task 3:** Set up Discord server with #welcome, #start-here, #help channels; write pinned messages
- [ ] **Task 4:** Prepare X/Twitter thread (hook tweet + 5-tweet thread: problem → architecture → demo → benchmark → call to action)
- [ ] **Task 5:** Execute launch sequence in order per AC1

## Tests Required

- Checklist completion — all 6 launch actions executed within 4-hour window

## Dev Agent Record

*(To be filled during execution)*

**Launch date:**
**Actions completed:**
**Initial star count (T+1h, T+4h, T+24h):**

## File List

*(No files changed — operational story)*
