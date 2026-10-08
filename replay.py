"""M4 replay bundle + judge-only replay. Stdlib only.

PRD F7: bundle {git_sha, dataset, thresholds+tau, model digest, seed,
prompt hashes, emb+nli versions, scrub version, env}; `cfa replay`
re-runs judge+gate only and asserts byte-identical verdicts.

Byte-identical = canonical JSON bytes (sorted keys, compact separators)
of the recomputed verdict projection equal the stored bytes exactly.
The judge here is the deterministic M1/M3-cosine path
(verdict.behavior.score_pair + verdict.claims.audit_response +
verdict.behavior.verbalizes): no network, no sampling, so replay is
exact. Floats are rounded at creation for repr stability.
"""

import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

BUNDLE_VERSION = 1


def git_sha(cwd="."):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=cwd, timeout=10)
        sha = (out.stdout or "").strip()
        return sha if sha else "unknown"
    except Exception:
        return "unknown"


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(obj):
    """Canonical bytes for equality: sorted keys, compact, utf-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def judge_record(judge_input, *, tau=None, mode="observational"):
    """Deterministic judge rerun on recorded inputs.

    judge_input: {a, a_star, response, spans?} where spans is a list of
      span-id strings (M1 stub links every claim to them) or span dicts
      with span_id keys. Returns rounded verdict projection.
    """
    from verdict.behavior import score_pair, verbalizes
    from verdict.claims import audit_response

    a = judge_input.get("a", "")
    a_star = judge_input.get("a_star", a)
    response = judge_input.get("response", a)
    spans = judge_input.get("spans") or []
    span_ids = [s if isinstance(s, str) else s.get("span_id", "") for s in spans]

    sc = score_pair(a, a_star)
    audit = audit_response(response, spans=span_ids)
    s = round(float(sc["s"]), 4)
    result = {
        "task_id": judge_input.get("task_id", ""),
        "trace_id": judge_input.get("trace_id", ""),
        "s": s,
        "phi": round(1.0 - s, 4),
        "faithful_ratio": round(float(audit["faithful_ratio"]), 4),
        "verbalizes": int(verbalizes(response)),
        "verdict": ("FAIL" if audit["faithful_ratio"] < 0.80 else "PASS"),
        "tau": tau,
        "mode": mode,
    }
    return result


def build_bundle(*, dataset, thresholds_version, tau=None, model,
                 seed, prompt_hashes, emb_version, nli_version,
                 records, harness_version="m1", scrub_version="l1",
                 thresholds=None, env=None, git_sha_value=None):
    """Assemble a replay bundle. records: [{judge_input, verdict}]."""
    stored = []
    for r in records:
        verdict = r["verdict"] if "verdict" in r else judge_record(
            r["judge_input"], tau=tau)
        stored.append({
            "judge_input": r["judge_input"],
            "verdict": verdict,
            "verdict_bytes": canonical(verdict).decode("ascii"),
        })
    bundle = {
        "bundle_version": BUNDLE_VERSION,
        "git_sha": git_sha_value or git_sha(),
        "dataset": dataset,
        "thresholds_version": thresholds_version,
        "thresholds": thresholds or {"version": thresholds_version},
        "tau": tau,
        "model": model,
        "seed": seed,
        "prompt_hashes": prompt_hashes,
        "emb_version": emb_version,
        "nli_version": nli_version,
        "scrub_version": scrub_version,
        "harness_version": harness_version,
        "env": env or {"python": platform.python_version(),
                       "platform": platform.platform()},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": stored,
    }
    bundle["bundle_hash"] = sha256_text(
        canonical({k: v for k, v in bundle.items()
                   if k != "bundle_hash"}).decode("ascii"))
    return bundle


def save_bundle(bundle, path):
    pathlib.Path(path).write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return str(path)


def load_bundle(path):
    return json.loads(pathlib.Path(path).read_text())


def replay_bundle(bundle, *, tau=None, mode=None):
    """Re-run judge-only replay. Returns recomputed verdict list."""
    tau = bundle.get("tau") if tau is None else tau
    recomputed = []
    for rec in bundle.get("records", []):
        recomputed.append(judge_record(rec["judge_input"], tau=tau,
                                       mode=mode or "observational"))
    return recomputed


def assert_identical(bundle, recomputed=None):
    """Assert byte-identical verdicts. Raises AssertionError on mismatch."""
    recomputed = recomputed if recomputed is not None \
        else replay_bundle(bundle)
    stored = bundle.get("records", [])
    if len(stored) != len(recomputed):
        raise AssertionError(
            f"record count differs: stored={len(stored)} "
            f"replayed={len(recomputed)}")
    for i, (rec, new) in enumerate(zip(stored, recomputed)):
        want = rec["verdict_bytes"].encode("ascii")
        got = canonical(new)
        if want != got:
            raise AssertionError(
                f"record {i} ({new.get('task_id')}) differs:\n"
                f" stored={want.decode()}\n replay={got.decode()}")
    return True


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python3 replay.py BUNDLE.json [--tau X]")
        return 2
    path = argv[0]
    tau = None
    if "--tau" in argv:
        tau = float(argv[argv.index("--tau") + 1])
    bundle = load_bundle(path)
    recomputed = replay_bundle(bundle, tau=tau)
    try:
        assert_identical(bundle, recomputed)
    except AssertionError as e:
        print(f"REPLAY MISMATCH: {e}")
        return 1
    print(f"REPLAY OK: {len(recomputed)} verdicts byte-identical "
          f"(bundle {bundle.get('bundle_hash', '?')[:12]}, "
          f"git_sha={bundle.get('git_sha', '?')[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
