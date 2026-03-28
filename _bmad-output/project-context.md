---
project_name: 'multi-modal-neural-network-brownfield'
user_name: 'Tim_D'
date: '2026-03-28'
sections_completed: ['technology_stack', 'language_rules', 'framework_rules', 'testing_rules', 'quality_rules', 'workflow_rules', 'anti_patterns']
status: 'complete'
rule_count: 31
optimized_for_llm: true
existing_patterns_found: 8
---

# Project Context for AI Agents

_This file contains critical rules and patterns that AI agents must follow when implementing code in this project. Focus on unobvious details that agents might otherwise miss._

---

## Technology Stack & Versions

- Python >=3.10 with a `src/` package layout.
- Core stack: `torch>=2.8.0`, `torchvision>=0.23.0`, `transformers>=4.57.3,<6`.
- Supporting stack: `datasets>=2.14.0`, `webdataset>=0.2.0`, `numpy>=1.24.0`, `scipy>=1.11.0`, `scikit-learn>=1.5.0`, `wandb>=0.15.0`, `redis>=4.5.4`, `diskcache>=5.6.0`, `wolframalpha>=5.0.0`, `requests>=2.32.4,<3`, `httpx>=0.28.0`.
- Prefer `safetensors>=0.4.0` and existing safe-load utilities when checkpoint format choices exist.
- Dev toolchain is part of the contract: `pytest`, `pytest-cov`, `pytest-benchmark`, `pytest-asyncio`, `black`, `isort`, `flake8`, `mypy`, and Ruff via `pyproject.toml`.
- Treat `docs/ARCHITECTURE.md` as the source of truth, and preserve existing bounds unless the task is explicitly a dependency upgrade: `transformers<6`, `pyyaml<7`, `requests<3`, `huggingface-hub<2`.

## Critical Implementation Rules

### Language-Specific Rules

- New Python code should be fully typed by default; do not drift toward partial typing, untyped public functions, or casual `Any` usage.
- Preserve existing naming and structure conventions: snake_case modules/functions/files, PascalCase classes, and current boundaries under `src/data`, `src/models`, `src/training`, `src/integrations`, and `src/utils`.
- Follow the existing formatting and import contract enforced by Black, isort, Flake8, and Ruff.
- Prefer explicit helpers over dense inline tensor logic, preserve documented input/output shapes and return contracts, and raise explicit errors for invalid modality inputs or configuration.
- Reuse existing safe-load, checkpoint, config, logging, and device-management utilities instead of writing one-off replacements.

### Framework-Specific Rules

- Treat PyTorch model interfaces as stable contracts: preserve tensor shapes, modality-specific inputs, and dictionary output keys unless the task explicitly changes the interface.
- Keep multimodal responsibilities separated: vision encoder, text encoder, fusion, task heads, controller logic, and training orchestration stay in their current modules.
- Follow the architecture document literally for core design choices: ViT-style vision encoder, BERT-style text encoder, early fusion, explicit task-head separation, and documented memory constraints are design decisions, not suggestions.
- Do not claim the double-loop controller is active end to end unless the trainer path is explicitly updated to pass `current_loss`, `current_accuracy`, and `gradient_norm`.
- Do not treat the placeholder tokenizer as benchmark-ready or production-ready, and do not confuse structural integration support with live training-path wiring for systems like Wolfram Alpha.

### Testing Rules

- Keep tests under `tests/` using the existing discovery pattern: `test_*.py`, `Test*`, and `test_*`.
- Reuse shared fixtures from `tests/conftest.py` and mocks from `tests/mock_utils.py`; avoid duplicate fixtures, fixture shadowing, and ad hoc integration mocks.
- Prefer deterministic unit coverage over heavier integration behavior when the logic can be validated at a lower layer.
- Use pytest markers consistently, especially `unit`, `integration`, `gpu`, `training`, `model`, `api`, `benchmark`, and `async`.
- Keep tests CPU-safe by default, avoid hidden network calls and environment-coupled assumptions, and validate tensor shapes, interface contracts, and returned keys for model/training code.

### Code Quality & Style Rules

- Treat strict typing as a quality gate and keep files narrowly scoped; extend existing domain modules instead of creating broad helper dumps or catch-all utilities.
- Preserve stable dictionary return structures and configuration-driven behavior where the repository already exposes configuration for that concern.
- Prefer small, explicit helpers over dense control flow in model, training, and integration code where tensor or state bugs are easy to hide.
- Avoid unnecessary comments; document intent, invariants, or architectural constraints only when they are not obvious from the code.
- When fixing a targeted issue, avoid opportunistic refactors unless they are required to make the change correct and testable.

### Development Workflow Rules

- Treat the current repository structure and architecture docs as baseline constraints; do not reorganize the project unless the task explicitly calls for structural change.
- Keep changes narrowly scoped; do not mix dependency upgrades, architectural rewrites, and cleanup into unrelated work.
- Verify training, architecture, and integration changes against documented implementation gaps in `docs/ARCHITECTURE.md` instead of assuming adjacent systems are complete.
- Prefer existing scripts, configs, and utilities over parallel workflows, and keep optional integrations and hardware-specific paths CPU-safe and fallback-safe unless the task explicitly narrows the support matrix.
- Keep artifacts out of core source paths unless the repository already defines a destination for them, and pair new behavior with focused tests aligned to current module boundaries.

### Critical Don't-Miss Rules

- Do not present structurally implemented components as fully wired, benchmark-ready, or production-ready when the architecture document says otherwise.
- Do not break tensor shape contracts, modality input expectations, or output dictionary keys without an explicit interface-change task.
- Do not replace research placeholders, mock-friendly seams, or optional integrations with hard dependencies unless the change is explicitly requested and tested.
- Do not assume CUDA, large-memory hardware, or active external services are available; preserve CPU-safe and fallback-safe behavior by default.
- Do not silently relax dependency bounds, typing rigor, or lint/test expectations to make a change easier to land.
- Do not perform broad rewrites when the real task is a narrow architecture-gap fix, test addition, or behavior correction.

---

## Usage Guidelines

**For AI Agents:**

- Read this file before implementing any code.
- Follow all rules exactly as documented.
- When in doubt, prefer the more restrictive option.
- Update this file if new patterns emerge.

**For Humans:**

- Keep this file lean and focused on agent needs.
- Update when the technology stack changes.
- Review periodically for outdated rules.
- Remove rules that become obvious over time.

Last Updated: 2026-03-28
