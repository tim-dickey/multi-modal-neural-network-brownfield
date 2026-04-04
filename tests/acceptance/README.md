# Acceptance Test Lane

This acceptance test gate was established using red-first test-driven development, with tests written before implementation to guide feature development. It now serves as a regression gate to preserve sprint-critical behavior across iterations.

**Status:** 14/14 passing on 2026-04-04

Execution order:
1. Run the frozen gate before changing sprint-critical behavior.
2. Implement or tighten acceptance criteria deliberately.
3. Re-run until the gate is green again.

Command:

```bash
pytest tests/acceptance/test_sprint_thread_tdd_red.py -m "acceptance and tdd_red" -q
```

Rules:
- New acceptance criteria are append-only.
- Existing acceptance tests may be tightened, but not removed.
- The gate is the authoritative regression check for the April 4 sprint thread.
