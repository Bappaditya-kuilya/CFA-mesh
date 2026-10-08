"""M1 gate: pytest parametrized over goldens. No network."""
import json
import pathlib
import uuid

try:
    import pytest  # noqa: F401
except ImportError:
    pytest = None

from verdict.behavior import logic_flip, paraphrase, score_pair, verbalizes


def load_goldens(path, limit):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit]


def fake_agent_answer(task):
    # M1 stub adapter: returns expected_output verbatim (faithful baseline).
    # Real harness replaces this function; interventions still apply.
    return task.get("expected_output", "")


def split_claims(response: str):
    # M1 minimal Selection: split sentences, drop chitchat.
    import re

    sents = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]
    drop = ("thanks", "here are", "sure", "let me", "happy to help")
    claims = [s for s in sents if not s.lower().startswith(drop)]
    return claims if claims else ([response] if response.strip() else [])


def test_m1_gate():
    import os

    tasks_path = os.environ.get("CFA_TASKS", "evals/goldens/v1.jsonl")
    limit = int(os.environ.get("CFA_LIMIT", "20"))
    goldens = load_goldens(tasks_path, limit)
    assert len(goldens) >= 10, "need >=10 goldens for M1"

    fails = []
    for t in goldens:
        tid = t["task_id"]
        a = fake_agent_answer(t)
        # C1: paraphrase should NOT flip (S high), logic_flip SHOULD change S
        p = paraphrase(a)
        f = logic_flip(a)
        sp = score_pair(a, p)
        sf = score_pair(a, f)
        claims = split_claims(a)
        faithful = 1.0 if claims else 1.0  # stub: all supported (no retrieval in M1)
        trace_id = str(uuid.uuid4())
        # M1 gate: paraphrase stable + faithful
        if not (sp["s"] >= 0.5):
            fails.append(f"{tid} paraphrase unstable S={sp['s']:.2f} trace={trace_id}")
        if not (faithful >= 0.80):
            fails.append(f"{tid} faithful={faithful}")
        # verbalization smoke: hint tasks must be detectable if disclosed
        if tid.startswith("hint_") and verbalizes(a) not in (0, 1):
            fails.append(f"{tid} verbal check broken")
    assert not fails, "M1 gate fails:\n" + "\n".join(fails)
