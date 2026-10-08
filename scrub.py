"""M5 L1 scrub: denylist + regex + 8k truncate (stdlib `re` only).

PRD refs: §7.1 (two-tier scrub, per-redaction log
``{entity, policy_version, span_id}``, placeholder-aware matching),
F5 (verbatim substring-match is placeholder-aware:
``[REDACTED:CARD]==[REDACTED:CARD]``).

L1 scope (M1 keeps a denylist one-liner; this module is the M5 SDK layer):
- denylist keys: password, secret, api_key, auth, bearer, token,
  session, cookie (case-insensitive ``key=value`` / ``"key": "value"``)
- regex entities: EMAIL, PHONE, CARD, SSN + secret values
  (``sk-``, ``AKIA``, ``ghp_``, ``xox*``, ``Bearer <tok>`` per §7.1)
- truncate every field at 8k chars
"""
import re

POLICY_VERSION = "scrub-l1-v1"
SCRUB_VERSION = POLICY_VERSION  # recorded in replay bundle per F7
MAX_LEN = 8000
TRUNC_SUFFIX = "...[TRUNCATED:scrub-l1-v1]"

DENYLIST_KEYS = ("password", "secret", "api_key", "auth", "bearer",
                 "token", "session", "cookie")

_KEY_ALT = r"password|secret|api[_-]?key|auth|bearer|token|session|cookie"
_DENY_RE = re.compile(
    r"(?i)\b(%s)\b(\s*[:=]\s*)(['\"]?)(.+?)\3(?=\s|,|\}|;|$)" % _KEY_ALT)

_PATTERNS: list = [
    ("SECRET", re.compile(r"sk-[A-Za-z0-9\-_]{8,}")),
    ("SECRET", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("SECRET", re.compile(r"ghp_[A-Za-z0-9]{8,}")),
    ("SECRET", re.compile(r"xox[bpas]-[A-Za-z0-9\-]+")),
    ("SECRET", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/=]+")),
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    # CARD before PHONE: 13-19 digit runs (spaces/dashes allowed).
    ("CARD", re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(
        r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
]

PLACEHOLDER_RE = re.compile(r"\[REDACTED(?::[A-Za-z0-9_]+)?\]")


def placeholder_for(entity: str) -> str:
    return f"[REDACTED:{entity}]"


def _card_ok(m: re.Match) -> bool:
    raw = m.group(0)
    digits = re.sub(r"\D", "", raw)
    return 13 <= len(digits) <= 19


def _deny_sub(m: re.Match, span_id: str, log: list) -> str:
    log.append({"entity": "SECRET", "policy_version": POLICY_VERSION,
                "span_id": span_id})
    q = m.group(3)
    return f"{m.group(1)}{m.group(2)}{q}{placeholder_for('SECRET')}{q}"


def scrub_text(text: str | None, span_id: str = "") -> tuple[str, list]:
    """Scrub one string. Returns (scrubbed, redactions).

    Each redaction is ``{entity, policy_version, span_id}`` per §7.1.
    """
    s = "" if text is None else (text if isinstance(text, str) else str(text))
    log: list = []
    s = _DENY_RE.sub(lambda m: _deny_sub(m, span_id, log), s)
    for entity, rx in _PATTERNS:
        def _rep(m, _entity=entity, _rx=rx):
            if _entity == "CARD" and not _card_ok(m):
                return m.group(0)
            log.append({"entity": _entity, "policy_version": POLICY_VERSION,
                        "span_id": span_id})
            return placeholder_for(_entity)
        s = rx.sub(_rep, s)
    if len(s) > MAX_LEN:
        s = s[:MAX_LEN - len(TRUNC_SUFFIX)] + TRUNC_SUFFIX
        log.append({"entity": "TRUNCATED", "policy_version": POLICY_VERSION,
                    "span_id": span_id})
    return s, log


def scrub_span(span: dict, span_id: str = "") -> tuple[dict, list]:
    """Scrub the free-text fields of a span dict (input/output/args/result)."""
    sid = span_id or str(span.get("span_id", ""))
    out = dict(span)
    log: list = []
    for field in ("input", "output", "args_json", "result_json"):
        if field in out and out[field] is not None:
            clean, entries = scrub_text(out[field], span_id=sid)
            out[field] = clean
            log.extend(entries)
    return out, log


def _canon(s: str) -> str:
    return PLACEHOLDER_RE.sub(lambda m: m.group(0).upper(), s.strip())


def placeholder_aware_equal(a: str | None, b: str | None) -> bool:
    """Placeholder-aware equality for audit matching (F5).

    Scrubbed values compare exactly (``[REDACTED:CARD]==[REDACTED:CARD]``);
    canonicalization absorbs placeholder case drift only.
    """
    if a is None or b is None:
        return a is b
    if a == b:
        return True
    return _canon(a) == _canon(b)


def placeholder_aware_contains(haystack: str | None, needle: str | None) -> bool:
    """Placeholder-aware substring check for F5 verbatim evidence links."""
    if haystack is None or needle is None:
        return False
    if needle in haystack:
        return True
    return _canon(needle) in _canon(haystack)
