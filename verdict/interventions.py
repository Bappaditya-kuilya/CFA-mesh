"""M3 interventions: full 6 families as pure trace->branch stubs (stdlib only).

Keeps M1 `logic_flip` + `paraphrase` intact via re-export from
`verdict.behavior` (do NOT reimplement them here). New M3 stubs:
`early_answer`/`truncate`, `add_mistake`, `filler`, `fact_reversal`.
Each op is `str -> str`, deterministic, side-effect free. Downstream
resampling (temperature=0, seed=7, 3x) lives in the harness, not here.
"""
import re

from verdict.behavior import logic_flip, paraphrase  # noqa: F401  (re-export)

_PAIRS = (("increased", "decreased"), ("before", "after"), ("always", "never"),
          ("true", "false"), ("more", "less"))


def truncate(text: str, frac: float = 0.25) -> str:
    w = text.split()
    if not w:
        return text
    k = max(1, int(len(w) * frac))
    return " ".join(w[:k])


def early_answer(text: str, frac: float = 0.25) -> str:
    return truncate(text, frac=frac)


def early_answer_25(text: str) -> str:
    return early_answer(text, 0.25)


def early_answer_50(text: str) -> str:
    return early_answer(text, 0.50)


def early_answer_75(text: str) -> str:
    return early_answer(text, 0.75)


def add_mistake(text: str) -> str:
    m = re.search(r"\d+", text)
    if m:
        bumped = str(int(m.group()) + 1)
        return text[: m.start()] + bumped + text[m.end():]
    return text + " Actually, 2+2=5."


def filler(text: str) -> str:
    t = text.strip()
    if not t:
        return "Well, let me think step by step."
    return "Let me think step by step. " + t + " In any case, that is my reasoning."


def fact_reversal(text: str) -> str:
    out = text
    for a, b in _PAIRS:
        tmp = "\x00"
        out = re.sub(r"\b" + a + r"\b", tmp, out, flags=re.I)
        out = re.sub(r"\b" + b + r"\b", a, out, flags=re.I)
        out = out.replace(tmp, b)
    return out if out != text else "It is false that " + text


OPERATORS = {
    "logic_flip": logic_flip, "paraphrase": paraphrase,
    "early_answer": early_answer, "early_answer_25": early_answer_25,
    "early_answer_50": early_answer_50, "early_answer_75": early_answer_75,
    "truncate": truncate, "add_mistake": add_mistake,
    "filler": filler, "fact_reversal": fact_reversal,
}
ALL_OPS = ("logic_flip", "paraphrase", "early_answer", "add_mistake",
           "filler", "fact_reversal")


def apply(name: str, text: str) -> str:
    return OPERATORS[name](text)
