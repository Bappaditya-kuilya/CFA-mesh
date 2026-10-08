def pytest_addoption(parser):
    parser.addoption("--run-eval", action="store_true", default=False)
    parser.addoption("--limit", type=int, default=20)
    parser.addoption("--tasks", default="evals/goldens/v1.jsonl")
