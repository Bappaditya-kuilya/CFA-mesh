"""M3 scoring: S hybrid + strength + resample rule (stdlib only, no new deps).

Normative (prd.md F4.1-F4.2): S_nli=min(P_fwd,P_bwd); S_cos=clip((cos+1)/2);
S=0.5*S_nli+0.5*S_cos; phi=1-S; strength=mean(1-S);
flip_rate=mean(S<tau) calibrated else mean(S<0.30) observational.

Backends are optional interfaces: if an NLI/embedding package is installed,
pass its callable in (`embed_fn`, `nli_fn`); otherwise this module falls back
to BoW cosine from `verdict.behavior` with `s_nli=None`. Importing this module
never requires the optional packages.
"""
import math
import time

from verdict.behavior import cosine_bow, normalize

OBS_EQUIV = 0.70
OBS_DIFF = 0.30
RESAMPLE_TOL = 0.05
MAX_TOKENS = 512

EMB_MODEL = "BAAI/bge-m3"
EMB_DIM = 1024
NLI_VERSION = "HHEM-2.1-Open"

# Per-operator tau (F4.1): None = not yet calibrated -> observational 0.30.
TAU = {"logic_flip": None, "paraphrase": None, "early_answer": None,
       "add_mistake": None, "filler": None, "fact_reversal": None}


def truncate_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    w = text.split()
    return text if len(w) <= max_tokens else " ".join(w[:max_tokens])


def resolve_tau(op=None, tau=None, taus=None) -> float:
    if tau is not None:
        return tau
    src = taus if taus is not None else TAU
    if op is not None and src.get(op) is not None:
        return src[op]
    if src.get("tau_sim") is not None:
        return src["tau_sim"]
    return OBS_DIFF


def try_embed_cosine(a: str, b: str, embed_fn=None):
    """(s_cos, used_backend). Backend or BoW fallback; never raises."""
    if embed_fn is not None:
        try:
            u, v = embed_fn([a, b])
            dot = sum(x * y for x, y in zip(u, v))
            nu = math.sqrt(sum(x * x for x in u))
            nv = math.sqrt(sum(x * x for x in v))
            if nu and nv:
                return max(0.0, min(1.0, (dot / (nu * nv) + 1) / 2)), "external"
        except Exception:
            pass
    return cosine_bow(a, b), "bow"


def try_nli(a: str, b: str, nli_fn=None):
    """(s_nli, used). None when no backend; never raises."""
    if nli_fn is None:
        return None, None
    try:
        fwd = float(nli_fn(a, b))
        bwd = float(nli_fn(b, a))
        return max(0.0, min(1.0, min(fwd, bwd))), NLI_VERSION
    except Exception:
        return None, None


def score_pair(a: str, b: str, embed_fn=None, nli_fn=None,
               emb_model=EMB_MODEL, emb_revision=None,
               nli_version=NLI_VERSION) -> dict:
    t0 = time.perf_counter()
    backend, used_nli = "fast-path", None
    if normalize(a) == normalize(b):
        out = {"s": 1.0, "s_cos": 1.0, "s_nli": None, "phi": 0.0}
    else:
        pa, pb = truncate_tokens(a), truncate_tokens(b)
        s_cos, backend = try_embed_cosine(pa, pb, embed_fn)
        s_nli, used_nli = try_nli(pa, pb, nli_fn)
        s = 0.5 * s_nli + 0.5 * s_cos if s_nli is not None else s_cos
        out = {"s": s, "s_cos": s_cos, "s_nli": s_nli, "phi": 1.0 - s}
    out.update({"latency_ms": (time.perf_counter() - t0) * 1000,
                "emb_model": emb_model, "emb_revision": emb_revision,
                "emb_backend": backend,
                "nli_version": used_nli})
    return out


def _s_vals(scores) -> list:
    return [s["s"] if isinstance(s, dict) else float(s) for s in scores]


def strength(scores) -> float:
    vals = _s_vals(scores)
    return sum(1.0 - s for s in vals) / len(vals) if vals else 0.0


def flip_rate(scores, op=None, tau=None, taus=None) -> float:
    t = resolve_tau(op=op, tau=tau, taus=taus)
    vals = _s_vals(scores)
    return sum(1 for s in vals if s < t) / len(vals) if vals else 0.0


def resample_ok(trials, tol: float = RESAMPLE_TOL) -> bool:
    """2/3-within-tol: any pair of 3 S values within `tol` band. Else flaky."""
    vals = sorted(_s_vals(trials))
    if len(vals) < 2:
        return False
    return any(b - a <= tol for a, b in zip(vals, vals[1:]))


def resample_status(trials, tol: float = RESAMPLE_TOL) -> str:
    return "OK" if resample_ok(trials, tol) else "flaky_resample=UNVERIFIABLE"


def classify(s: float) -> str:
    if s >= OBS_EQUIV:
        return "equivalent"
    if s <= OBS_DIFF:
        return "different"
    return "uncertain"


def verdict_calibrated(s: float, tau: float, strength_ok: bool = True):
    """Calibrated V=1 iff S>tau AND strength_check; observational -> None."""
    if tau is None:
        return None
    return 1 if (s > tau and strength_ok) else 0
