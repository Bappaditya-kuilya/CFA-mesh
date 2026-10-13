"""Root conftest: repo root on sys.path so bare `pytest` works.

`tests/test_interventions.py` imports `verdict.*` with no sys.path shim of
its own, so a single-file run (`pytest tests/test_interventions.py`) fails
with `ModuleNotFoundError` unless the repo root is importable. Inserting it
here (repo root conftest loads before any test module) fixes every
invocation with no PYTHONPATH needed.

Flag -> env bridge lives in tests/conftest.py (sole owner of
--run-eval/--limit/--tasks); pytest_configure below only re-applies the
same defaults defensively so running a subset that skips tests/conftest
still honors explicit env. Explicit env always wins.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pytest_configure(config):
    if "CFA_LIMIT" not in os.environ:
        try:
            os.environ["CFA_LIMIT"] = str(config.getoption("limit"))
        except ValueError:
            os.environ["CFA_LIMIT"] = "20"
    if "CFA_TASKS" not in os.environ:
        try:
            os.environ["CFA_TASKS"] = config.getoption("tasks") or "evals/goldens/v1.jsonl"
        except ValueError:
            os.environ["CFA_TASKS"] = "evals/goldens/v1.jsonl"
