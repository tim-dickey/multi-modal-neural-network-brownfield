---
id: "6.3"
epic: 6
title: "Create Dockerfile for Reproducible Environment"
status: "ready"
priority: "high"
estimate: "S"
assignee: ""
sprint: 6
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "Dockerfile"
  - "docker-compose.yml"
---

# Story 6.3: Create Dockerfile for Reproducible Environment

## User Story

As an independent AI developer,
I want a Dockerfile that sets up the complete NeuralMix training environment,
So that I can reproduce any published results exactly without manually managing CUDA, PyTorch, and dependency versions.

## Context

**Epic:** Epic 6 — Public Release and Community Launch
**Brownfield context:** No `Dockerfile` or `docker-compose.yml` currently exists. `requirements.txt` is present.

**Primary files:** `Dockerfile` (new), `docker-compose.yml` (new)

> ⚠️ **Dependency:** `requirements.txt` must be finalized (Story 5.5 may add Sphinx extras).

## Acceptance Criteria

**AC1 — Docker image builds:**
**Given** the NeuralMix repository root
**When** `docker build -t neuralmix:v1.0 .` is run
**Then** the Docker image builds successfully using NVIDIA NGC PyTorch base image (`nvcr.io/nvidia/pytorch:24.01-py3` or equivalent)
**And** all `requirements.txt` dependencies are installed in the image

**AC2 — Dry-run check passes in container:**
**Given** the built Docker image
**When** `docker run --gpus all neuralmix:v1.0 python train.py --check` is run
**Then** the `--check` dry run completes successfully and reports CUDA available
**And** the container exits with code 0

**AC3 — docker-compose training service:**
**Given** `docker-compose.yml` exists
**When** `docker compose up training` is run
**Then** the training service starts with the GPU device mounted, the project directory bind-mounted, and `configs/default.yaml` as the active config

## Tasks

- [ ] **Task 1:** Create `Dockerfile` using `nvcr.io/nvidia/pytorch:24.01-py3` base; copy `requirements.txt`; run `pip install -r requirements.txt`; set `WORKDIR /app`; copy source
- [ ] **Task 2:** Create `docker-compose.yml` with `training` service: `--gpus all`, bind mount project dir, default config command
- [ ] **Task 3:** Manual verification: `docker build` succeeds; `docker run --check` exits 0

## Tests Required

- Manual: `docker build` and `docker run --check` verification
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
