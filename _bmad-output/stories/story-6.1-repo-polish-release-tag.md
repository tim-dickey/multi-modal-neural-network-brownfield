---
id: "6.1"
epic: 6
title: "Repository Polish and v1.0 Release Tagging"
status: "ready"
priority: "high"
estimate: "M"
assignee: ""
sprint: 6
sourceEpic: "_bmad-output/planning-artifacts/epics.md"
files:
  - "README.md"
  - "CONTRIBUTING.md"
  - "CHANGELOG.md"
  - ".github/ISSUE_TEMPLATE/"
---

# Story 6.1: Repository Polish and v1.0 Release Tagging

## User Story

As the NeuralMix project maintainer,
I want the repository to be fully polished and tagged as v1.0.0,
So that the public release is professional, discoverable, and sets the right first impression.

## Context

**Epic:** Epic 6 — Public Release and Community Launch
**Brownfield context:** `CONTRIBUTING.md` exists (1064 bytes — partial). `LICENSE` is Apache 2.0 — already correct. `.github/ISSUE_TEMPLATE/` directory exists under `.github/`. `CHANGELOG.md` does not exist. This story creates the missing files and polishes existing ones for release.

**Primary files:** `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `.github/ISSUE_TEMPLATE/`

> ⚠️ **Dependency:** Story 5.4 (README complete) must be done first.

## Acceptance Criteria

**AC1 — Repository release readiness:**
**Given** the repository at release time
**When** a developer visits the GitHub repo
**Then** `README.md` is complete (Story 5.4 done), `LICENSE` is Apache 2.0 ✅, `CONTRIBUTING.md` exists with contribution guidelines and code of conduct reference, `.github/ISSUE_TEMPLATE/` contains at minimum a bug report and feature request template, and `CHANGELOG.md` has a v1.0.0 entry with all major features listed

**AC2 — v1.0.0 git tag release notes:**
**Given** the v1.0.0 git tag is created
**When** `git tag -a v1.0.0` is run
**Then** the tagged release on GitHub includes: full release notes (features, known limitations, hardware compatibility), links to Hugging Face model hub, link to Discord server, and the demo video URL

**AC3 — Known limitations framing:**
**Given** the `known-limitations` section of the README
**When** it is read
**Then** it clearly states: "Wolfram Alpha auxiliary supervision is implemented (v1.5)", "WebDataset streaming is not yet implemented (v1.5)", "Auto-regressive text generation is not yet implemented (v1.5)", "Multi-GPU DDP training is not activated (v1.5)"

## Tasks

- [ ] **Task 1:** Update `CONTRIBUTING.md` — add contribution workflow, code style guide, test requirements, code of conduct reference
- [ ] **Task 2:** Create `CHANGELOG.md` with v1.0.0 entry listing all Epic 1–5 major features
- [ ] **Task 3:** Create `.github/ISSUE_TEMPLATE/bug_report.md` and `.github/ISSUE_TEMPLATE/feature_request.md`
- [ ] **Task 4:** Verify known limitations section in README (AC3)
- [ ] **Task 5:** Prepare v1.0.0 release notes draft for GitHub release page

## Tests Required

- Manual review against AC1–AC3 checklist
- All existing tests must still pass

## Dev Agent Record

*(To be filled by dev agent during implementation)*

**Implementation notes:**
**Files changed:**
**Tests created/modified:**
**Decisions made:**

## File List

*(To be filled by dev agent after implementation)*
