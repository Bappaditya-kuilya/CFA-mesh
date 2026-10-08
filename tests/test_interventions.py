"""M1 intervention unit checks (stdlib only, no network).

M1 keeps ONLY logic_flip + paraphrase (prd.md F3).
Guards against anyone adding early_answer/add_mistake/filler/fact_reversal.
"""
import verdict.behavior as behavior
from verdict.behavior import logic_flip, paraphrase, score_pair

FORBIDDEN_OPS = ("early_answer", "add_mistake", "filler", "fact_reversal")

EXAMPLES = [
    "Yes, the refund was issued.",
    "No, the server is down.",
    "The capital of Japan is Tokyo.",
]


def test_m1_keeps_only_two_ops():
    for name in FORBIDDEN_OPS:
        assert not hasattr(behavior, name), f"M1 forbids verdict.behavior.{name}"
        assert name not in dir(behavior), f"M1 forbids verdict.behavior.{name}"
    assert callable(logic_flip) and callable(paraphrase)


def test_paraphrase_stable():
    for e in EXAMPLES:
        s = score_pair(e, paraphrase(e))["s"]
        assert s >= 0.5, f"paraphrase unstable S={s:.3f} for {e!r}"


def test_logic_flip_changes_yes_no():
    assert logic_flip("Yes, the refund was issued.").startswith("No")
    assert logic_flip("No, the server is down.").startswith("Yes")
    # is/is-not branch must negate, not return input unchanged
    flipped = logic_flip("The capital of Japan is Tokyo.")
    assert flipped != "The capital of Japan is Tokyo."
    assert "not" in flipped.lower()
