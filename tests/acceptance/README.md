# Acceptance Test Lane

This folder contains ATDD acceptance tests for sprint-critical behavior.

Execution order:
1. Run RED tests before implementation begins.
2. Implement feature slices.
3. Re-run until green.

Command:

```bash
pytest tests/acceptance/test_sprint_thread_tdd_red.py -m "acceptance and tdd_red"
```
