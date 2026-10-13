"""Pipeline tests: cache key, tier runs, gate, risk-accept, replay, export.

Stdlib only, pytest-optional: plain `test_*` functions (same style as
test_security.py) plus a `__main__` runner, so both
`pytest tests/test_pipeline.py` and `python3 tests/test_pipeline.py` work.
No network.

PRD refs: orchestrator content_cache_key/ContentCache/BudgetEnforcer/
needs_full_gate/select_tier/run; gate aggregate/decide_gate/
validate_risk_accept; replay judge_record/build_bundle/assert_identical;
export materialize_evidence/build_export.
"""
import copy
import datetime
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest  # noqa: F401
except ImportError:
    pytest = None

from orchestrator import (BudgetEnforcer, BudgetExceeded, ContentCache,
                          content_cache_key, needs_full_gate, run,
                          select_tier)
from gate import aggregate, decide_gate, validate_risk_accept
from replay import (assert_identical, build_bundle, canonical, judge_record,
                    replay_bundle)
from export import build_export, materialize_evidence


def _base_kwargs():
    return dict(model_id="m", sys_prompt_hash="sp", user_template_hash="ut",
                temp=0.0, tools_schema_hash="ts", rag_corpus_version="c1",
                task_id="t", branch_config="b0", git_sha="sha",
                harness_version="m2-1", nli_version="n1",
                thresholds_version="th1")


def test_cache_key_determinism_and_invalidation():
    k0 = content_cache_key(**_base_kwargs())
    assert k0 == content_cache_key(**_base_kwargs()), "same inputs must key equal"
    base = _base_kwargs()
    triggers = [("model_id", "m2"), ("sys_prompt_hash", "sp2"),
                ("user_template_hash", "ut2"), ("tools_schema_hash", "ts2"),
                ("rag_corpus_version", "c2"), ("task_id", "t2")]
    assert len(triggers) == 6, "must cover the 6 invalidation triggers"
    for field, other in triggers:
        mod = dict(base)
        mod[field] = other
        assert content_cache_key(**mod) != k0, f"{field} change must invalidate"
    assert len(k0) == 64, f"sha256 hex expected, got {k0!r}"


def test_cache_hit_skip_second_run():
    with tempfile.TemporaryDirectory() as d:
        cache = ContentCache(path=os.path.join(d, "c.jsonl"))
        tasks = [{"task_id": "t1", "expected_output": "a1"},
                 {"task_id": "t2", "expected_output": "a2"}]
        calls = []

        def compute(task, b):
            calls.append((task["task_id"], b))
            return {"output": task["expected_output"], "tokens": 10,
                    "cost_usd": 0.0, "elapsed_s": 0.0}

        r1 = run(tasks, tier="pr", cache=cache, git_sha="abc",
                 compute=compute)
        assert len(r1["results"]) == 4, r1  # pr: 2 tasks x 2 branches
        assert all(r["cached"] is False for r in r1["results"]), r1["results"]
        n_calls = len(calls)
        assert n_calls == 4, f"compute once per miss, got {n_calls}"
        r2 = run(tasks, tier="pr", cache=cache, git_sha="abc",
                 compute=compute)
        assert all(r["cached"] is True for r in r2["results"]), r2["results"]
        assert len(calls) == n_calls, "cache hit must skip compute"
        assert cache.hits > 0 and cache.hit_rate > 0, cache.log_line()


def test_budget_abort_on_over_limit_trace():
    enf = BudgetEnforcer(tier="pr")
    try:
        enf.check_trace(tokens=10 ** 9, cost_usd=0.0, elapsed_s=0.0,
                        n_branches=2, trace_id="t0")
        raise AssertionError("over-token trace must raise")
    except BudgetExceeded as e:
        assert "budget_exceeded" in str(e), str(e)
    try:
        enf.add_spend(99.0)
        raise AssertionError("over-cap spend must raise")
    except BudgetExceeded as e:
        assert "budget_exceeded" in str(e), str(e)

    def pricey(task, b):
        return {"output": "x", "tokens": 10 ** 9, "cost_usd": 99.0,
                "elapsed_s": 0.0}

    with tempfile.TemporaryDirectory() as d:
        try:
            run([{"task_id": "t0"}], tier="pr",
                cache=ContentCache(path=os.path.join(d, "c.jsonl")),
                budgets=BudgetEnforcer(tier="pr"), compute=pricey)
            raise AssertionError("run must abort on over-limit trace")
        except BudgetExceeded as e:
            assert "budget_exceeded" in str(e), str(e)


def test_needs_full_gate():
    assert needs_full_gate(["README.md", "docs/x.md"]) is False
    assert needs_full_gate(["prompts/judge.md"]) is True
    assert needs_full_gate(["agents/judge.py"]) is True
    assert needs_full_gate(["schemas/tools.json"]) is True


def test_select_tier():
    pr = select_tier("pr", 1000)
    assert pr["n_tasks"] == 20, pr
    assert pr["branches"] == 2, pr
    assert pr["concurrent"] == 10, pr
    ng = select_tier("nightly", 1000)
    assert ng["n_tasks"] == 200, ng
    assert ng["branches"] == 7, ng
    assert ng["concurrent"] == 20, ng
    assert select_tier("nightly", 1000, limit=50)["n_tasks"] == 50


def test_aggregate():
    verdicts = [{"faithful": 1.0, "s": 1.0, "phi": 0.0, "verbalizes": 1},
                {"faithful": 0.0, "s": 0.0, "phi": 1.0, "verbalizes": 1}]
    agg = aggregate(verdicts)
    assert agg["n"] == 2, agg
    assert agg["faithful_mean"] == 0.5, agg
    assert agg["strength"] == 0.5, agg
    assert agg["phi_median"] == 0.5, agg
    assert agg["phi_p25"] == 0.25, agg
    assert agg["phi_p75"] == 0.75, agg
    assert agg["rho"] is None, "rho must be NULL when observational"


def test_decide_gate():
    def _agg(faithful, verbal=1.0, contra=0):
        return {"faithful_mean": faithful, "verbal_rate": verbal,
                "faithful_ci": [faithful - 0.01, faithful + 0.01],
                "contradicted_count": contra, "rho": None}

    warn = decide_gate(_agg(0.5))
    assert warn["decision"] == "WARN", warn
    assert any("faithful" in r for r in warn["reasons"]), warn["reasons"]
    ok = decide_gate(_agg(0.95))
    assert ok["decision"] == "PASS", ok
    blocked = decide_gate(_agg(0.95, contra=1))
    assert blocked["decision"] == "BLOCK", blocked
    assert any("CONTRADICTED" in r for r in blocked["reasons"]), blocked
    reg = decide_gate(_agg(0.80), base_faithful=0.95, base_ci=None)
    assert reg["decision"] == "BLOCK", reg
    assert reg["delta_vs_base"] is not None and reg["delta_vs_base"] > 0.02


def _ra(now, by="alice", approved_by="bob", ttl_days=7):
    exp = now + datetime.timedelta(days=ttl_days)
    return {"by": by, "reason": "test waiver", "controls": "monitor",
            "expires_at": exp.isoformat(), "approved_by": approved_by,
            "created_at": now.isoformat()}


def test_risk_accept():
    now = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    good = validate_risk_accept(_ra(now), now=now)
    assert good["valid"] is True, good
    self_appr = validate_risk_accept(_ra(now, approved_by="alice"), now=now)
    assert self_appr["valid"] is False, self_appr
    assert any("approver==author" in e for e in self_appr["errors"]), self_appr
    expired = validate_risk_accept(_ra(now, ttl_days=-1), now=now)
    assert expired["valid"] is False, expired
    assert expired["expired"] is True, expired
    assert any("expired" in e for e in expired["errors"]), expired
    missing = validate_risk_accept({}, now=now)
    assert missing["valid"] is False and missing["errors"], missing


def test_replay_roundtrip_and_tamper():
    ji1 = {"task_id": "t1", "trace_id": "tr1", "a": "hello world",
           "a_star": "hello world", "response": "hello world",
           "spans": ["sp1"]}
    ji2 = {"task_id": "t2", "trace_id": "tr2", "a": "the sky is blue",
           "a_star": "the sky is green", "response": "the sky is blue",
           "spans": []}
    bundle = build_bundle(dataset="d1", thresholds_version="th1", tau=None,
                          model={"id": "stub"}, seed=7,
                          prompt_hashes={"judge": "h"}, emb_version="e1",
                          nli_version="n1",
                          records=[{"judge_input": ji1}, {"judge_input": ji2}],
                          git_sha_value="abc123")
    assert len(bundle["records"]) == 2, bundle.keys()
    recomputed = replay_bundle(bundle)
    assert len(recomputed) == 2, recomputed
    assert canonical(recomputed[0]).decode("ascii") == \
        bundle["records"][0]["verdict_bytes"]
    assert assert_identical(bundle, recomputed) is True
    tampered = copy.deepcopy(bundle)
    tampered["records"][0]["verdict_bytes"] = json.dumps({"s": 0.0})
    try:
        assert_identical(tampered)
        raise AssertionError("tampered bundle must not verify")
    except AssertionError as e:
        assert "differs" in str(e), str(e)


def test_export_fail_embeds_span_text():
    spans = {"sp1": "telescope reading 42"}
    fail_v = {"task_id": "t1", "verdict": "FAIL", "s": 0.2,
              "faithful": 0.0,
              "claims": [{"text": "the sky is green", "verdict": "FAIL",
                          "evidence_span_ids": ["sp1"]}]}
    pass_v = {"task_id": "t2", "verdict": "PASS", "s": 1.0,
              "faithful": 1.0,
              "claims": [{"text": "the sky is blue", "verdict": "SUPPORTED",
                          "evidence_span_ids": ["sp1"]}]}
    ev = materialize_evidence(fail_v, spans=spans)
    assert len(ev) == 1, ev
    assert ev[0]["evidence"][0]["content"] == "telescope reading 42", ev
    gate = {"decision": "FAIL", "mode": "observational",
            "reasons": ["FAIL: test"]}
    agg = {"n": 2, "faithful_mean": 0.5, "verbal_rate": 1.0,
           "phi_median": 0.5}
    versions = {"git_sha": "abc", "thresholds_version": "th1",
                "dataset": "d1", "harness_version": "m2-1", "seed": 7,
                "emb_version": "e1", "nli_version": "n1"}
    out = build_export(gate=gate, agg=agg, verdicts=[fail_v, pass_v],
                       versions=versions, spans=spans)
    rows = {r["task_id"]: r for r in out["rows"]}
    assert rows["t1"]["fail_evidence"], "FAIL must carry evidence"
    assert rows["t2"]["fail_evidence"] == [], "PASS must carry none"
    assert "telescope reading 42" in out["markdown"], out["markdown"]
    assert "telescope reading 42" in out["jsonl"], "jsonl must embed spans"


if __name__ == "__main__":
    tests = sorted((k, v) for k, v in globals().items()
                   if k.startswith("test_") and callable(v))
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    sys.exit(1 if failed else 0)
