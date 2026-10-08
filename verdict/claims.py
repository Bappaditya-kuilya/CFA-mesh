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

_SENT_SPLIT = re.compile(r"[.!?]+|\n+")
_BULLET = re.compile(r"^(?:[-*•\d]+[.)\]]?\s+)")


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


def verify_claims(claims: list, spans=None) -> list:
    """Verify step (M1 stub): link every claim to the given tool span ids.

    spans: iterable of span-id strings (M1 stub: tool spans from the trace).
    Non-empty spans -> SUPPORTED; empty/missing -> UNVERIFIABLE with
    evidence_span_ids == []. Never emits CONTRADICTED in M1 (no NLI).
    """
    ids = [s for s in (spans or [])][:MAX_EVIDENCE_SPANS]
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


def audit_response(response: str, spans=None) -> dict:
    """Decompose-then-verify in one call. Returns claims + faithful_ratio."""
    claims = verify_claims(split_claims(response), spans=spans)
    return {"claims": claims, "faithful_ratio": faithful_ratio(claims)}
