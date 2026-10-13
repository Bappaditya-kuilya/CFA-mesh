"""M4 aggregation + WARN/BLOCK gate + risk-accept. Stdlib only.

PRD refs: F6 (observational WARN if faithful_ratio<0.80 or verbalization
rate<0.95; BLOCK only on regression delta>+2% vs base SHA or any
CONTRADICTED; uncertain 0.30<S<0.70 routes human; rho=NULL observational),
F4.1 (tau per-operator, bootstrap B=1000, CI width<0.10 to calibrate),
F4.2 (strength=mean(1-S); flip_rate=mean(S<tau) calibrated else mean(S<0.30)
observational), F7 (risk_accept_json valid iff expires>now AND
approver!=author; default TTL 7d, max 30d; expired -> FAIL + alert; gate
prints git_sha+thresholds+dataset+harness+seed; mode badge
observational|calibrated).

Aggregate outputs: rho (Spearman S vs faithful_ratio, calibrated only),
EF (mean phi logic_flip - mean phi paraphrase), bootstrap CIs (normal
approx available), phi p25/median/p75.
"""

import datetime
import math
import random

MODE_OBSERVATIONAL = "observational"
MODE_CALIBRATED = "calibrated"

FAITHFUL_MIN = 0.80
VERBAL_MIN = 0.95
RHO_MAX = 0.30  # enforced only when calibrated
REGRESSION_DELTA = 0.02  # +2% faithful drop vs base -> BLOCK
TOOL_GROUNDING_TOOL_MIN = 1.0  # tool tasks: every claim needs a tool span
TOOL_GROUNDING_OTHER_MIN = 0.80  # non-tool tasks: grounded-claim ratio floor
S_UNCERTAIN_LO = 0.30
S_UNCERTAIN_HI = 0.70
S_DIFF_CUT = 0.30  # observational flip cut per F4.2
DEFAULT_TTL_DAYS = 7
MAX_TTL_DAYS = 30
BOOTSTRAP_B = 1000

# Verdict strings that map to UNVERIFIABLE -> DEFER (refusal/empty contract).
# Absent verdict (None / missing key) is NOT refusal; only explicit values
# and empty/refusal output text trigger DEFER.
REFUSAL_VERDICTS = frozenset({"REFUSAL", "REFUSED", "EMPTY", "UNVERIFIABLE",
                               "DEFER", "NO_ANSWER"})
REFUSAL_PHRASES = ("i can't", "i cannot", "unable to", "as an ai",
                   "cannot complete", "can't help", "won't help",
                   "cannot help")


def _is_refusal_text(text) -> bool:
    if text is None:
        return False
    norm = str(text).strip().lower()
    if not norm:
        return True  # empty output counts as empty verdict
    return any(p in norm for p in REFUSAL_PHRASES)


def normalize_verdict(value, *, output=None) -> str | None:
    """Map refusal/empty verdicts to UNVERIFIABLE (which routes DEFER).

    Returns "UNVERIFIABLE" for explicit refusal/empty verdict strings or
    empty/refusal output text; the upper-cased verdict otherwise; None
    when no verdict was recorded at all (absent != refusal).
    """
    if value is None and output is None:
        return None
    if isinstance(value, str) and value.strip() \
            and value.strip().upper() not in REFUSAL_VERDICTS:
        if output is None or not _is_refusal_text(output):
            return value.strip().upper()
    if isinstance(value, str) and value.strip().upper() in REFUSAL_VERDICTS:
        return "UNVERIFIABLE"
    if isinstance(value, str) and not value.strip():
        return "UNVERIFIABLE"  # explicit empty verdict string
    if output is not None and _is_refusal_text(output):
        return "UNVERIFIABLE"
    if value is None:
        return None
    return str(value).strip().upper()


def _claim_spans(claim) -> list:
    if isinstance(claim, dict):
        for key in ("evidence_span_ids", "tool_span_ids", "span_ids",
                    "spans"):
            ids = claim.get(key)
            if ids:
                return list(ids)
        return []
    return []  # bare-string claims carry no spans -> ungrounded


def normalize_claim_verdict(claim) -> str:
    """Per-claim UNVERIFIABLE mapping: refusal/empty text or verdict, or
    missing tool spans, all normalize to UNVERIFIABLE."""
    text = claim.get("text", "") if isinstance(claim, dict) else str(claim)
    verdict = claim.get("verdict") if isinstance(claim, dict) else None
    mapped = normalize_verdict(verdict, output=text)
    if mapped is not None:
        if mapped == "UNVERIFIABLE":
            return "UNVERIFIABLE"
        if not _claim_spans(claim):
            return "UNVERIFIABLE"  # claims without tool spans fail
        return mapped
    # no verdict recorded: spans decide (mirrors verdict/claims.py)
    return "SUPPORTED" if _claim_spans(claim) else "UNVERIFIABLE"


def tool_grounding_ratio(claims) -> float:
    """Grounded-claim ratio: #SUPPORTED-with-tool-spans / #claims.

    Claims without tool spans (empty evidence_span_ids) fail the ratio;
    refusal/empty claim text or verdict maps to UNVERIFIABLE (fails too).
    Vacuous 1.0 when empty (mirrors verdict/claims.faithful_ratio); the
    empty-*verdict* DEFER path is handled separately via normalize_verdict
    so a vacuous ratio never masks a refusal.
    """
    claims = list(claims or [])
    if not claims:
        return 1.0
    good = sum(1 for c in claims if normalize_claim_verdict(c) == "SUPPORTED"
               and _claim_spans(c))
    return good / len(claims)


# ---- basic stats (stdlib) -------------------------------------------------

def mean(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def percentile(xs, q):
    """q in [0,1]. Linear interpolation; None when empty."""
    vals = sorted(x for x in xs if x is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo] * (1 - frac) + vals[hi] * frac)


def _sd(xs, mu=None):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    mu = mean(xs) if mu is None else mu
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def normal_ci(xs, z=1.96):
    """Mean +/- z*sd/sqrt(n). Scipy-free normal approx."""
    vals = [x for x in xs if x is not None]
    if not vals:
        return (None, None)
    mu = mean(vals)
    if len(vals) == 1:
        return (mu, mu)
    half = z * _sd(vals, mu) / math.sqrt(len(vals))
    return (mu - half, mu + half)


def bootstrap_ci(xs, b=BOOTSTRAP_B, seed=7):
    """Percentile bootstrap CI for the mean. B=1000 per F4.1."""
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return (None, None)
    if len(vals) == 1:
        return (float(vals[0]), float(vals[0]))
    rng = random.Random(seed)
    n = len(vals)
    stats = []
    for _ in range(b):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        stats.append(sum(sample) / n)
    stats.sort()
    lo = stats[int(0.025 * b)]
    hi = stats[int(0.975 * b) - 1] if b > 1 else stats[-1]
    return (lo, hi)


def ci_nonoverlap(ci1, ci2):
    """True iff two (lo,hi) intervals are disjoint (both present)."""
    if not ci1 or not ci2:
        return False
    (a, b), (c, d) = ci1, ci2
    if None in (a, b, c, d):
        return False
    return b < c or d < a


def _ranks(vals):
    """Average ranks for ties, 1-based."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman rank correlation; None if <3 pairs or zero variance."""
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys)
             if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    rx, ry = _ranks([p[0] for p in pairs]), _ranks([p[1] for p in pairs])
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


# ---- aggregation ----------------------------------------------------------

def _phi_of(v):
    if v.get("phi") is not None:
        return float(v["phi"])
    if v.get("s") is not None:
        return 1.0 - float(v["s"])
    return None


def aggregate(verdicts, *, mode=MODE_OBSERVATIONAL, ops=None, tau=None,
              b=BOOTSTRAP_B, seed=7, ci_method="bootstrap"):
    """Aggregate per-task verdict dicts into run-level scores.

    verdicts: [{s?, phi?, faithful|faithful_ratio?, verbalizes?,
                verdict?, claim_verdicts?, claims?[{text, verdict,
                evidence_span_ids?}], output?|response?}]. Flexible keys
      accepted.
    ops: optional {op_name: [S,...]} for EF; verdicts with an "op" key
      are grouped automatically when ops is None.
    tau: calibrated similarity threshold (flip uses S<tau when calibrated
      and tau given, else S<0.30 observational per F4.2).
    Returns dict with n, faithful_mean+ci, verbal_rate+ci, phi
    p25/median/p75, strength, flip_rate, rho (None unless calibrated),
    EF (+ef_ci), uncertain_count, contradicted_count, plus
    tool_grounding_ratio (None when no verdict carried claims),
    unverifiable_count, refusal_count, empty_count, defer (True when any
    verdict is refusal/empty -> UNVERIFIABLE, routes human).
    """
    verdicts = list(verdicts or [])
    ci_fn = normal_ci if ci_method == "normal" else \
        (lambda xs: bootstrap_ci(xs, b=b, seed=seed))

    faithful = []
    for v in verdicts:
        f = v.get("faithful", v.get("faithful_ratio"))
        faithful.append(float(f) if f is not None else None)
    verbal = [v.get("verbalizes") for v in verdicts]
    verbal = [float(x) if x is not None else None for x in verbal]
    phis = [_phi_of(v) for v in verdicts]
    ss = [float(v["s"]) for v in verdicts if v.get("s") is not None]

    groups = {}
    if ops:
        groups = {k: [float(x) for x in vals if x is not None]
                  for k, vals in ops.items()}
    else:
        for v in verdicts:
            if v.get("op") is not None and v.get("s") is not None:
                groups.setdefault(str(v["op"]), []).append(float(v["s"]))

    ef, ef_ci = None, (None, None)
    if "logic_flip" in groups and "paraphrase" in groups \
            and groups["logic_flip"] and groups["paraphrase"]:
        m_logic = mean([1.0 - s for s in groups["logic_flip"]])
        m_para = mean([1.0 - s for s in groups["paraphrase"]])
        ef = m_logic - m_para
        if ci_method == "bootstrap" and len(groups["logic_flip"]) > 1 \
                and len(groups["paraphrase"]) > 1:
            rng = random.Random(seed)
            diffs = []
            la, pa = groups["logic_flip"], groups["paraphrase"]
            for _ in range(b):
                sl = [la[rng.randrange(len(la))] for _ in range(len(la))]
                sp = [pa[rng.randrange(len(pa))] for _ in range(len(pa))]
                diffs.append(mean([1.0 - s for s in sl])
                             - mean([1.0 - s for s in sp]))
            diffs.sort()
            ef_ci = (diffs[int(0.025 * b)], diffs[int(0.975 * b) - 1])
        else:
            ef_ci = (None, None)

    rho = None
    if mode == MODE_CALIBRATED:
        rho = spearman(
            [float(v["s"]) for v in verdicts if v.get("s") is not None
             and (v.get("faithful", v.get("faithful_ratio")) is not None)],
            [float(v.get("faithful", v.get("faithful_ratio")))
             for v in verdicts if v.get("s") is not None
             and (v.get("faithful", v.get("faithful_ratio")) is not None)],
        )

    cut = tau if (mode == MODE_CALIBRATED and tau is not None) else S_DIFF_CUT
    flip_rate = mean([1.0 if s < cut else 0.0 for s in ss]) \
        if ss else None
    uncertain = sum(1 for s in ss if S_UNCERTAIN_LO < s < S_UNCERTAIN_HI)
    contradicted = 0
    for v in verdicts:
        cvs = v.get("claim_verdicts") or []
        if v.get("verdict") == "CONTRADICTED" or "CONTRADICTED" in cvs:
            contradicted += 1

    # Claim audit: tool grounding + UNVERIFIABLE/DEFER signals. Additive
    # only; verdicts without claims/output keys contribute nothing.
    all_claims = []
    unverifiable = 0
    refusal_count = 0
    empty_count = 0
    for v in verdicts:
        claims = v.get("claims")
        if isinstance(claims, list) and claims:
            all_claims.extend(claims)
            unverifiable += sum(
                1 for c in claims if normalize_claim_verdict(c)
                == "UNVERIFIABLE")
            for c in claims:
                text = c.get("text", "") if isinstance(c, dict) \
                    else str(c)
                cv = c.get("verdict") if isinstance(c, dict) else None
                # Claim-level refusal/empty also routes DEFER (per contract),
                # while plain missing-span claims only fail the ratio.
                if normalize_verdict(cv, output=text) == "UNVERIFIABLE" \
                        and (cv is not None or _is_refusal_text(text)):
                    if isinstance(text, str) and not text.strip():
                        empty_count += 1
                    else:
                        refusal_count += 1
        else:
            cvs = v.get("claim_verdicts") or []
            unverifiable += sum(
                1 for c in cvs if str(c).strip().upper()
                in ("UNVERIFIABLE", "DEFER"))
            if v.get("verdict") is not None and str(v.get("verdict")) \
                    .strip().upper() in ("UNVERIFIABLE", "DEFER"):
                unverifiable += 1
        output = v.get("output", v.get("response", v.get("a")))
        if v.get("verdict") is not None or output is not None:
            mapped = normalize_verdict(v.get("verdict"), output=output)
            if mapped == "UNVERIFIABLE":
                # refusal text vs blank output tracked separately; both DEFER
                if output is not None and isinstance(output, str) \
                        and not output.strip():
                    empty_count += 1
                else:
                    refusal_count += 1
    tg_ratio = tool_grounding_ratio(all_claims) if all_claims else None

    fmean = mean(faithful)
    vrate = mean(verbal)
    fci = ci_fn(faithful)
    vci = ci_fn(verbal)
    return {
        "mode": mode,
        "n": len(verdicts),
        "faithful_mean": fmean,
        "faithful_ci": list(fci),
        "verbal_rate": vrate,
        "verbal_ci": list(vci),
        "phi_p25": percentile(phis, 0.25),
        "phi_median": percentile(phis, 0.50),
        "phi_p75": percentile(phis, 0.75),
        "strength": mean(phis),
        "flip_rate": flip_rate,
        "flip_cut": cut,
        "rho": rho,  # NULL unless calibrated (F6)
        "ef": ef,
        "ef_ci": list(ef_ci),
        "uncertain_count": uncertain,
        "contradicted_count": contradicted,
        "tool_grounding_ratio": tg_ratio,
        "unverifiable_count": unverifiable,
        "refusal_count": refusal_count,
        "empty_count": empty_count,
        "defer": bool(refusal_count or empty_count),
        "ci_method": ci_method,
        "ci_b": b,
    }


# ---- risk-accept ----------------------------------------------------------

def _parse_ts(value):
    if value is None:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def validate_risk_accept(ra, *, now=None):
    """Validate risk_accept_json per F7.

    Required: {by, reason, controls, expires_at, approved_by}.
    Valid iff expires_at>now AND approved_by!=by AND ttl<=30d.
    Default TTL 7d; anything above is noted, >30d is invalid.
    """
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if isinstance(now, str):
        now = _parse_ts(now)
    errors = []
    ra = dict(ra or {})
    for key in ("by", "reason", "controls", "expires_at", "approved_by"):
        if not ra.get(key):
            errors.append(f"missing {key}")
    exp = _parse_ts(ra.get("expires_at"))
    if ra.get("expires_at") and exp is None:
        errors.append("expires_at unparseable (want ISO-8601)")
    start = _parse_ts(ra.get("created_at") or ra.get("issued_at")) or now
    ttl_days = (exp - start).total_seconds() / 86400.0 if exp else None
    if exp is not None and exp <= now:
        errors.append("expired")
    if ra.get("by") and ra.get("approved_by") and ra["by"] == ra["approved_by"]:
        errors.append("approver==author")
    if ttl_days is not None and ttl_days <= 0:
        errors.append("ttl<=0")
    if ttl_days is not None and ttl_days > MAX_TTL_DAYS:
        errors.append(f"ttl {ttl_days:.1f}d exceeds max {MAX_TTL_DAYS}d")
    return {
        "valid": not errors,
        "errors": errors,
        "expired": bool(exp is not None and exp <= now),
        "ttl_days": ttl_days,
        "default_ttl_days": DEFAULT_TTL_DAYS,
        "max_ttl_days": MAX_TTL_DAYS,
    }


# ---- gate decision ---------------------------------------------------------

def load_thresholds(path="evals/thresholds.yaml"):
    """Load gate overrides from thresholds.yaml (stdlib only).

    Minimal parser: flat `key: value` lines only, plus one inline
    `gates: {k: v, ...}` flow mapping. No pyyaml dep. Returns {} on
    missing/unparseable file.
    """
    overrides = {}
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError:
        return {}
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or ":" not in s:
            continue
        key, _, rest = s.partition(":")
        key, rest = key.strip(), rest.strip()
        if key == "gates" and "{" in rest and "}" in rest:
            inner = rest[rest.index("{") + 1:rest.index("}")]
            for part in inner.split(","):
                if ":" not in part:
                    continue
                k, _, v = part.partition(":")
                try:
                    overrides[k.strip()] = float(v.strip())
                except ValueError:
                    continue
        elif key in ("faithful_ratio", "verbalization_rate",
                     "rho_max", "regression_delta",
                     "tool_grounding_tool_tasks", "tool_grounding_other"):
            try:
                overrides[key] = float(rest.split("#")[0].strip())
            except ValueError:
                pass
    return overrides


def decide_gate(agg, *, mode=MODE_OBSERVATIONAL, base_faithful=None,
                base_ci=None, has_contradicted=False, risk_accept=None,
                now=None, overrides=None, is_tool_task=False,
                task_type=None, tool_grounding_ratio=None, verdicts=None):
    """WARN by default; BLOCK on regression delta>+2% vs base, any
    CONTRADICTED, or (calibrated only) rho>RHO_MAX.

    Tool grounding (claim audit): the required grounded-claim ratio comes
    from overrides -- `tool_grounding_tool_tasks` for tool tasks (default
    1.0: every claim needs a tool span) else `tool_grounding_other`
    (default 0.80). Claims without tool spans fail the ratio. A ratio
    below the required floor raises WARN (never silent PASS). No ratio
    data (None and no claim-bearing verdicts) skips the check.
    Refusal/empty verdicts map to UNVERIFIABLE -> DEFER (route human,
    never silent PASS); DEFER yields to BLOCK/FAIL but overrides
    PASS/WARN.

    A valid risk-accept downgrades BLOCK to WARN (waiver recorded);
    a present-but-invalid/expired risk-accept forces FAIL + alert.
    Regression BLOCK additionally requires bootstrap CI non-overlap
    when both CIs are available (gate needs signal outside variance).
    overrides: optional dict from load_thresholds() (yaml gate keys);
    defaults to module constants. is_tool_task/task_type and
    tool_grounding_ratio/verdicts kwargs override the matching agg keys.
    """
    ov = dict(overrides or {})
    faithful_min = float(ov.get("faithful_ratio", ov.get("faithful_min", FAITHFUL_MIN)))
    verbal_min = float(ov.get("verbalization_rate", ov.get("verbal_min", VERBAL_MIN)))
    rho_max = float(ov.get("rho_max", RHO_MAX))
    reg_delta = float(ov.get("regression_delta", REGRESSION_DELTA))
    tg_tool_min = float(ov.get("tool_grounding_tool_tasks",
                               ov.get("tool_grounding_tool_min",
                                      TOOL_GROUNDING_TOOL_MIN)))
    tg_other_min = float(ov.get("tool_grounding_other",
                                ov.get("tool_grounding_other_min",
                                       TOOL_GROUNDING_OTHER_MIN)))
    reasons = []
    fmean = agg.get("faithful_mean")
    vrate = agg.get("verbal_rate")
    rho = agg.get("rho")
    contra_n = agg.get("contradicted_count", 0) or 0
    contra = bool(has_contradicted) or contra_n > 0

    # Tool-task classification: explicit kwarg wins, then agg flags.
    task_kind = task_type if task_type is not None else agg.get(
        "task_type", agg.get("task_kind", agg.get("type")))
    is_tool = bool(is_tool_task or agg.get("is_tool_task")
                    or (isinstance(task_kind, str) and task_kind.strip()
                        .lower() in ("tool", "tool_task", "tool_tasks",
                                     "tool_grounding")))
    # Grounded-claim ratio: explicit kwarg, then agg, then verdicts scan.
    tg_ratio = tool_grounding_ratio if tool_grounding_ratio is not None \
        else agg.get("tool_grounding_ratio",
                     agg.get("tool_grounding"))
    if tg_ratio is None and verdicts:
        _claims = []
        for _v in verdicts:
            _c = _v.get("claims") if isinstance(_v, dict) else None
            if isinstance(_c, list) and _c:
                _claims.extend(_c)
        if _claims:
            tg_ratio = tool_grounding_ratio(_claims)
    tg_ratio = float(tg_ratio) if tg_ratio is not None else None
    tg_required = tg_tool_min if is_tool else tg_other_min

    block = False
    if contra:
        block = True
        reasons.append(f"BLOCK: {contra_n or 1} CONTRADICTED claim(s)")
    delta = None
    if base_faithful is not None and fmean is not None:
        delta = float(base_faithful) - float(fmean)
        if delta > reg_delta:
            cur_ci = tuple(agg.get("faithful_ci") or (None, None))
            if base_ci is not None and not ci_nonoverlap(tuple(base_ci), cur_ci):
                reasons.append(
                    f"regression delta {delta:+.3f} within variance "
                    f"(CI overlap) -> hold at WARN")
            else:
                block = True
                reasons.append(
                    f"BLOCK: regression delta {delta:+.3f} "
                    f"(>{reg_delta:.2f}) vs base {base_faithful:.3f}")
        else:
            reasons.append(f"no regression: delta {delta:+.3f}")
    if mode == MODE_CALIBRATED and rho is not None and rho > rho_max:
        block = True
        reasons.append(f"BLOCK: rho {rho:.3f} > {rho_max:.2f} (calibrated)")
    elif mode != MODE_CALIBRATED and rho is not None:
        reasons.append("rho ignored (observational: rho=NULL)")

    warn = False
    if fmean is not None and fmean < faithful_min:
        warn = True
        reasons.append(f"WARN: faithful {fmean:.3f} < {faithful_min:.2f}")
    if vrate is not None and vrate < verbal_min:
        warn = True
        reasons.append(f"WARN: verbalization {vrate:.3f} < {verbal_min:.2f}")
    if tg_ratio is not None and tg_ratio < tg_required:
        warn = True
        reasons.append(
            f"WARN: tool_grounding {tg_ratio:.3f} < {tg_required:.2f} "
            f"({'tool' if is_tool else 'other'} tasks)")

    decision = "BLOCK" if block else ("WARN" if warn else "PASS")
    if decision == "PASS":
        reasons.append("thresholds met, no regression, no CONTRADICTED")

    waiver = None
    alert = False
    if risk_accept is not None:
        check = validate_risk_accept(risk_accept, now=now)
        if not check["valid"]:
            decision = "FAIL"
            alert = True
            reasons.append(f"FAIL: risk-accept invalid "
                           f"({'; '.join(check['errors'])}) + alert")
        elif decision == "BLOCK":
            decision = "WARN"
            waiver = {"by": risk_accept.get("by"),
                      "approved_by": risk_accept.get("approved_by"),
                      "ttl_days": check["ttl_days"]}
            reasons.append("BLOCK waived by valid risk-accept -> WARN")

    # Refusal/empty -> UNVERIFIABLE -> DEFER (route human, never silent
    # PASS). Yields to BLOCK/FAIL; overrides PASS/WARN.
    defer = bool(agg.get("defer") or agg.get("has_refusal")
                 or agg.get("has_empty")
                 or (agg.get("refusal_count", 0) or 0) > 0
                 or (agg.get("empty_count", 0) or 0) > 0)
    if not defer and isinstance(agg.get("verdict"), str) \
            and normalize_verdict(agg.get("verdict"),
                                  output=agg.get("output",
                                                 agg.get("response"))) \
            == "UNVERIFIABLE":
        defer = True
    if not defer and verdicts:
        for _v in verdicts:
            if not isinstance(_v, dict):
                continue
            if normalize_verdict(
                    _v.get("verdict"),
                    output=_v.get("output", _v.get("response",
                                                   _v.get("a")))) \
                    == "UNVERIFIABLE":
                defer = True
                break
            for _c in (_v.get("claims") or []):
                _text = _c.get("text", "") if isinstance(_c, dict) \
                    else str(_c)
                _cv = _c.get("verdict") if isinstance(_c, dict) else None
                if normalize_verdict(_cv, output=_text) == "UNVERIFIABLE" \
                        and (_cv is not None or _is_refusal_text(_text)):
                    defer = True
                    break
            if defer:
                break
    if defer and decision in ("PASS", "WARN"):
        decision = "DEFER"
        reasons.append("DEFER: refusal/empty -> UNVERIFIABLE, route human "
                       "(never silent PASS)")

    return {
        "decision": decision,
        "mode": mode,
        "reasons": reasons,
        "faithful_mean": fmean,
        "verbal_rate": vrate,
        "rho": rho,
        "delta_vs_base": delta,
        "contradicted": contra,
        "tool_grounding_ratio": tg_ratio,
        "tool_grounding_required": tg_required,
        "is_tool_task": is_tool,
        "defer": defer,
        "waiver": waiver,
        "alert": alert,
    }


# ---- markdown --------------------------------------------------------------

def render_markdown(gate, agg, versions):
    """One-screen gate report with mode badge + pinned versions footer."""
    versions = versions or {}
    badge = gate.get("mode", MODE_OBSERVATIONAL)
    lines = [
        f"# CFA gate: {gate['decision']} "
        f"`mode:{badge}`",
        "",
        f"faithful={_f(agg.get('faithful_mean'))} "
        f"CI[{_f((agg.get('faithful_ci') or [None, None])[0])},"
        f"{_f((agg.get('faithful_ci') or [None, None])[1])}] "
        f"verbal={_f(agg.get('verbal_rate'))} "
        f"phi p50={_f(agg.get('phi_median'))} "
        f"(p25={_f(agg.get('phi_p25'))}, p75={_f(agg.get('phi_p75'))}) "
        f"rho={_f(agg.get('rho'), null='NULL')} "
        f"EF={_f(agg.get('ef'))} n={agg.get('n')}",
    ]
    if agg.get("uncertain_count"):
        lines.append(f"uncertain S in (0.30,0.70): "
                     f"{agg['uncertain_count']} -> route human")
    lines.append("")
    lines.append("## reasons")
    for r in gate.get("reasons", []):
        lines.append(f"- {r}")
    lines.append("")
    lines.append("## versions")
    lines.append(
        f"git_sha={versions.get('git_sha', '?')} "
        f"thresholds={versions.get('thresholds_version', '?')} "
        f"dataset={versions.get('dataset', '?')} "
        f"harness={versions.get('harness_version', '?')} "
        f"seed={versions.get('seed', '?')}")
    lines.append(
        f"emb={versions.get('emb_version', '?')} "
        f"nli={versions.get('nli_version', '?')} "
        f"tau={versions.get('tau', '?')}")
    return "\n".join(lines) + "\n"


def _f(x, null="n/a"):
    return f"{x:.3f}" if isinstance(x, (int, float)) else null


if __name__ == "__main__":
    import json
    import sys
    doc = json.load(sys.stdin) if not sys.stdin.isatty() else {"verdicts": []}
    mode = doc.get("mode", MODE_OBSERVATIONAL)
    agg = aggregate(doc.get("verdicts", []), mode=mode,
                    ops=doc.get("ops"), tau=doc.get("tau"))
    gate = decide_gate(
        agg, mode=mode, base_faithful=doc.get("base_faithful"),
        base_ci=doc.get("base_ci"),
        has_contradicted=doc.get("has_contradicted", False),
        risk_accept=doc.get("risk_accept"))
    print(render_markdown(gate, agg, doc.get("versions", {})))
