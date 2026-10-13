"""M1 claim splitter: FActScore decompose-then-verify shape, stdlib-only.

Decompose (Selection): sentence-split the response, drop greetings/narration.
Disambiguation: M1 no-op (no pronoun/coreference resolution).
Verify: M1 stub only — no LLM, no HHEM. A claim linked to tool span ids is
SUPPORTED; a claim with empty evidence_span_ids is UNVERIFIABLE.

Mirrors the `split_claims` Selection rule in tests/test_faithfulness.py so the
M1 gate and this module agree on what counts as a claim.
"""
import re

# Selection: drop greetings / narration lines starting with these prefixes.
DROP_PREFIXES = ("thanks", "here are", "sure", "let me", "happy to help")

# PRD F5: evidence chains are capped at 8 spans; longer chains split the claim.
MAX_EVIDENCE_SPANS = 8

# P1 slice 3/6: tool grounding (name-only M1). Tasks whose id starts with
# this prefix require tool evidence; kind check only, args are not checked.
TOOL_TASK_PREFIX = "tool_"
TOOL_SPAN_KIND = "tool_result_span"

_SENT_SPLIT = re.compile(r"[.!?]+|\n+")
_BULLET = re.compile(r"^(?:[-*•\d]+[.)\]]?\s+)")


def _split_span_ids(spans=None) -> tuple:
    """Split span refs into (generic_ids, tool_ids) preserving order.

    Accepts span-id strings (no kind info) and span dicts with
    span_id/id + kind keys. Name-only M1: a span counts as tool evidence
    iff dict kind == TOOL_SPAN_KIND; args/tool_name are never checked.
    """
    generic_ids = []
    tool_ids = []
    for s in (spans or []):
        if isinstance(s, dict):
            sid = s.get("span_id", "") or s.get("id", "") or s.get("evidence_id", "")
            if not sid:
                continue
            generic_ids.append(sid)
            if s.get("kind", "") == TOOL_SPAN_KIND:
                tool_ids.append(sid)
        elif isinstance(s, str):
            if s:
                generic_ids.append(s)
        # ignore any other type: no id to cite
    return generic_ids, tool_ids


def _task_needs_tool(task_id=None) -> bool:
    if isinstance(task_id, dict):
        task_id = task_id.get("task_id", "")
    return isinstance(task_id, str) and task_id.startswith(TOOL_TASK_PREFIX)


def _clean(segment: str) -> str:
    text = segment.strip()
    text = _BULLET.sub("", text).strip()
    return text


def _is_chitchat(text: str) -> bool:
    return text.lower().startswith(DROP_PREFIXES)


def split_claims(response: str) -> list:
    """Decompose a response into atomic claim strings (Selection only).

    Drops greetings/narration; resolves nothing (M1). Returns list[str].
    Falls back to [response] when nothing survives, [] when input is blank.
    """
    if not response or not response.strip():
        return []
    claims = []
    for seg in _SENT_SPLIT.split(response):
        text = _clean(seg)
        if not text or _is_chitchat(text):
            continue
        claims.append(text)
    return claims if claims else [response.strip()]


def verify_claims(claims: list, spans=None, task_id=None) -> list:
    """Verify step (M1 stub): link every claim to the given tool span ids.

    spans: iterable of span-id strings or span dicts with span_id/kind keys
      (M1 stub: tool spans from the trace). Non-empty spans -> SUPPORTED;
      empty/missing -> UNVERIFIABLE with evidence_span_ids == [].
      Never emits CONTRADICTED in M1 (no NLI).

    Tool grounding (P1 3/6, name-only M1): when task_id starts with
      "tool_" (cf. expected_tools in evals/goldens/v1.jsonl: tool_001
      read_issue, tool_002 git_diff, tool_003 run_tests), a claim is
      SUPPORTED only if spans contain at least one dict with
      kind == "tool_result_span". String-only ids carry no kind and do
      not satisfy tool tasks; other-span kinds do not satisfy either.
      Args/tool_name are never checked. Contract stays
      SUPPORTED/UNVERIFIABLE with evidence capped at 8 spans.
    """
    _, tool_ids = _split_span_ids(spans)
    if _task_needs_tool(task_id):
        ids = tool_ids[:MAX_EVIDENCE_SPANS]
    else:
        generic_ids, _ = _split_span_ids(spans)
        ids = generic_ids[:MAX_EVIDENCE_SPANS]
    verified = []
    for claim in claims:
        text = claim if isinstance(claim, str) else claim.get("text", "")
        if ids:
            verified.append({
                "text": text,
                "evidence_span_ids": list(ids),
                "verdict": "SUPPORTED",
            })
        else:
            verified.append({
                "text": text,
                "evidence_span_ids": [],
                "verdict": "UNVERIFIABLE",
            })
    return verified


def faithful_ratio(verified: list) -> float:
    """FActScore-style score: #SUPPORTED / #claims. Vacuous 1.0 when empty."""
    if not verified:
        return 1.0
    supported = sum(1 for c in verified if c.get("verdict") == "SUPPORTED")
    return supported / len(verified)


def audit_response(response: str, spans=None, task_id=None) -> dict:
    """Decompose-then-verify in one call. Returns claims + faithful_ratio."""
    claims = verify_claims(split_claims(response), spans=spans, task_id=task_id)
    return {"claims": claims, "faithful_ratio": faithful_ratio(claims)}
