import os


def pytest_addoption(parser):
    parser.addoption("--run-eval", action="store_true", default=False)
    parser.addoption("--limit", type=int, default=20)
    parser.addoption("--tasks", default="evals/goldens/v1.jsonl")


def pytest_configure(config):
    # Bridge CLI flags -> env so the gate honors
    # `pytest --run-eval --limit N --tasks PATH` with no custom CLI.
    # Explicit env wins over flags.
    if "CFA_LIMIT" not in os.environ:
        os.environ["CFA_LIMIT"] = str(config.getoption("limit"))
    if "CFA_TASKS" not in os.environ:
        os.environ["CFA_TASKS"] = config.getoption("tasks") or "evals/goldens/v1.jsonl"
