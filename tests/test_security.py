"""M5 security tests: budgets, retention, lockout, tenant isolation + scrub.

Stdlib only, pytest-optional: plain `test_*` functions (same style as
test_interventions.py) plus a `__main__` runner, so both
`pytest tests/test_security.py` and `python tests/test_security.py` work.

PRD refs: §5/F2 (budgets), §7.5 (retention purge), §7.2 (lockout persists,
tenant isolation), §7.1 (scrub + placeholder-aware match).
"""
import os
import sys
import json
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest  # noqa: F401
except ImportError:
    pytest = None

from auth import (AuthError, AuthStore, LockedOut, check_tenant, hash_key,
                  require_role, tenant_where, verify_bearer)
from scrub import (POLICY_VERSION, placeholder_aware_contains,
                   placeholder_aware_equal, scrub_span, scrub_text)
from store import BUDGETS, Store, over_budget
from evidence_bank import (export_fail_bundle, get_evidence, list_for_task,
                           pin_evidence)
from export import build_export, materialize_evidence


def _tmp_store(d):
    return Store(db_path=os.path.join(d, "c.db"),
                 jsonl_path=os.path.join(d, "t.jsonl"))


def test_budget_abuse():
    assert BUDGETS["max_tokens_per_trace"] == 8000
    assert BUDGETS["max_cost_usd"] == 0.05
    ok = over_budget(tokens_in=4000, tokens_out=1000, cost_usd=0.01,
                     latency_s=10.0, branches=2)
    assert ok == [], f"in-budget usage flagged: {ok}"
    assert "tokens_exceeded" in over_budget(tokens_in=8001)
    assert "tokens_exceeded" in over_budget(tokens_in=7000, tokens_out=1001)
    assert "cost_exceeded" in over_budget(cost_usd=0.051)
    assert "timeout_exceeded" in over_budget(latency_s=120.1)
    assert "branches_exceeded" in over_budget(branches=8)
    # Global run caps abort `budget_exceeded` FAIL (F2).
    assert over_budget(run_cost_usd=2.0, tier="pr") == []
    assert "budget_exceeded" in over_budget(run_cost_usd=2.01, tier="pr")
    assert over_budget(run_cost_usd=20.0, tier="nightly") == []
    assert "budget_exceeded" in over_budget(run_cost_usd=20.01,
                                            tier="nightly")


def test_retention_purge():
    with tempfile.TemporaryDirectory() as d:
        st = _tmp_store(d)
        st.save_trace("tr_old", "task1", "i", "o", tenant_id="A")
        st.save_trace("tr_new", "task1", "i", "o", tenant_id="A")
        old = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                            time.gmtime(time.time() - 31 * 86400))
        st._conn.execute("UPDATE trajectories SET created_at=? "
                         "WHERE trace_id='tr_old'", (old,))
        st._conn.commit()
        counts = st.purge_older_than(days=30)
        assert counts["trajectories"] == 1, counts
        assert st.get_trace("tr_old", tenant_id="A") is None
        assert st.get_trace("tr_new", tenant_id="A") is not None
        st.close()


def test_lockout_persists():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "auth.db")
        a1 = AuthStore(db)
        a1.create_key("user-secret-1", kind="user", tenant_id="A",
                      user_id="u1", role="member")
        bad = "wrong-token"
        ip = "10.0.0.1"
        for _ in range(4):
            try:
                a1.authenticate(bad, ip=ip)
            except LockedOut:
                raise AssertionError("locked out before 5 fails")
            except AuthError:
                pass
        assert not a1.is_locked_out(ip)
        try:
            a1.authenticate(bad, ip=ip)
            raise AssertionError("5th bad attempt should lock out")
        except LockedOut:
            pass
        assert a1.is_locked_out(ip)
        # Persists across reopen (new connection, same sqlite file).
        a2 = AuthStore(db)
        assert a2.is_locked_out(ip)
        try:
            a2.authenticate(bad, ip=ip)
            raise AssertionError("reopened store should still lock out")
        except LockedOut:
            pass
        # Real key works and is constant-time verified (other IP clean).
        got = a1.authenticate("user-secret-1", ip="10.9.9.9")
        assert got["tenant_id"] == "A" and got["role"] == "member"
        assert verify_bearer("user-secret-1",
                             hash_key("user-secret-1"))
        assert not verify_bearer("user-secret-2",
                                 hash_key("user-secret-1"))
        # Unknown-token and known-token branches both compare_digest:
        # a bad token from a fresh IP must NOT be locked out by ip's counter.
        assert not a2.is_locked_out("10.9.9.9")
        # User keys die with the user; unknown role rejected.
        assert a1.revoke_user_keys("u1") == 1
        try:
            require_role("viewer", "write")
            raise AssertionError("viewer must not write")
        except AuthError:
            pass
        a1.close()
        a2.close()


def test_lockout_per_ip():
    """Lockout is keyed per-IP: one bad IP never locks out another,
    and rotating tokens from the same IP still trips the counter."""
    with tempfile.TemporaryDirectory() as d:
        a = AuthStore(os.path.join(d, "auth.db"))
        a.create_key("k-good", kind="user", tenant_id="A",
                     user_id="u9", role="member")
        bad_ip, good_ip = "192.168.1.10", "192.168.1.11"
        for i in range(5):
            try:
                a.authenticate(f"bad-token-{i}", ip=bad_ip)
            except (AuthError, LockedOut):
                pass
        assert a.is_locked_out(bad_ip), "bad IP must lock out after 5 fails"
        assert not a.is_locked_out(good_ip), \
            "per-IP: good IP must stay clean"
        # Good key from clean IP still works despite bad IP lockout.
        got = a.authenticate("k-good", ip=good_ip)
        assert got["tenant_id"] == "A"
        # Same bad IP stays locked even with the good token (per-IP).
        try:
            a.authenticate("k-good", ip=bad_ip)
            raise AssertionError("locked IP must stay locked")
        except LockedOut:
            pass
        a.close()


def test_tenant_isolation():
    with tempfile.TemporaryDirectory() as d:
        st = _tmp_store(d)
        st.save_trace("tr_a", "task1", "i", "o", tenant_id="A")
        # token_A cannot GET tenant_B's trace: scoped read returns None.
        assert st.get_trace("tr_a", tenant_id="B") is None
        assert st.get_trace("tr_a", tenant_id="A") is not None
        # Raw SQL with the tenant_where helper returns only own rows.
        st.save_trace("tr_b", "task1", "i", "o", tenant_id="B")
        rows = st._conn.execute(
            f"SELECT trace_id FROM trajectories WHERE {tenant_where()}",
            ("A",)).fetchall()
        assert [r[0] for r in rows] == ["tr_a"], rows
        # Cross-tenant upsert guard: same trace_id, other tenant denied.
        try:
            st.save_trace("tr_a", "task1", "i", "o", tenant_id="B")
            raise AssertionError("cross-tenant overwrite must raise")
        except (ValueError, Exception) as e:
            assert "cross-tenant" in str(e).lower(), e
        assert st.get_trace("tr_a", tenant_id="A") is not None
        # purge_older_than tenant filter: old row of A purged, B kept.
        old = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                            time.gmtime(time.time() - 31 * 86400))
        st._conn.execute("UPDATE trajectories SET created_at=? "
                         "WHERE trace_id='tr_a'", (old,))
        st._conn.commit()
        counts = st.purge_older_than(days=30, tenant_id="B")
        assert counts["trajectories"] == 0, counts
        assert st.get_trace("tr_a", tenant_id="A") is not None
        counts = st.purge_older_than(days=30, tenant_id="A")
        assert counts["trajectories"] == 1, counts
        assert st.get_trace("tr_a", tenant_id="A") is None
        st.close()
    try:
        check_tenant("A", "B")
        raise AssertionError("cross-tenant access must raise")
    except AuthError:
        pass
    check_tenant("A", "A")  # same tenant: no raise


def test_cross_tenant_denied():
    """Evidence bank + export enforce tenant scope end to end."""
    import sqlite3
    with tempfile.TemporaryDirectory() as d:
        conn = sqlite3.connect(os.path.join(d, "ev.db"))
        pin_evidence(conn, "t1", "span", "secret-A", tenant_id="A")
        pin_evidence(conn, "t1", "span", "secret-B", tenant_id="B")
        rows_a = list_for_task(conn, "t1", tenant_id="A")
        rows_b = list_for_task(conn, "t1", tenant_id="B")
        assert len(rows_a) == 1 and rows_a[0]["tenant_id"] == "A", rows_a
        assert len(rows_b) == 1 and rows_b[0]["tenant_id"] == "B", rows_b
        # Cross-tenant get returns None.
        assert get_evidence(conn, rows_a[0]["evidence_id"],
                            tenant_id="B") is None
        # Scoped export carries only own rows.
        bundle_a = export_fail_bundle(conn, "t1", tenant_id="A")
        assert "secret-A" in bundle_a and "secret-B" not in bundle_a
        # Export tenant check: mixed verdict tenants denied.
        gate = {"decision": "FAIL", "mode": "observational", "reasons": []}
        agg = {"n": 2}
        versions = {}
        va = {"task_id": "t1", "verdict": "FAIL", "tenant_id": "A",
              "claims": []}
        vb = {"task_id": "t2", "verdict": "FAIL", "tenant_id": "B",
              "claims": []}
        try:
            build_export(gate=gate, agg=agg, verdicts=[va, vb],
                         versions=versions)
            raise AssertionError("mixed-tenant export must raise")
        except ValueError as e:
            assert "cross-tenant" in str(e).lower(), e
        try:
            build_export(gate=gate, agg=agg, verdicts=[vb],
                         versions=versions, tenant_id="A")
            raise AssertionError("wrong-tenant export must raise")
        except ValueError as e:
            assert "cross-tenant" in str(e).lower(), e
        # Same-tenant export passes.
        out = build_export(gate=gate, agg=agg, verdicts=[va],
                           versions=versions, tenant_id="A")
        assert out["rows"], out
        conn.close()


def test_scrub_redacts_and_logs():
    raw = ("login password=hunter2 email jane@example.com "
           "card 4111-1111-1111-1111 ssn 123-45-6789 "
           "call +1-415-555-2671 key sk-abcDEF123456 "
           'api_key: "AKIAIOSFODNN7EXAMPLE" Bearer abc.def.ghi')
    clean, log = scrub_text(raw, span_id="sp_1")
    for secret in ("hunter2", "jane@example.com", "4111-1111-1111-1111",
                   "123-45-6789", "sk-abcDEF123456", "AKIAIOSFODNN7EXAMPLE"):
        assert secret not in clean, f"leaked {secret!r} in {clean!r}"
    assert "[REDACTED:SECRET]" in clean
    assert "[REDACTED:EMAIL]" in clean
    assert "[REDACTED:CARD]" in clean
    assert "[REDACTED:SSN]" in clean
    assert "[REDACTED:PHONE]" in clean
    assert log, "per-redaction log must be non-empty"
    for entry in log:
        assert set(entry) == {"entity", "policy_version", "span_id"}
        assert entry["policy_version"] == POLICY_VERSION
        assert entry["span_id"] == "sp_1"
    # 8k truncate.
    big, big_log = scrub_text("x" * 9000)
    assert len(big) <= 8000 and big.endswith("TRUNCATED:scrub-l1-v1]")
    assert any(e["entity"] == "TRUNCATED" for e in big_log)
    # GROQ_API_KEY denylist: underscore boundary must still match.
    gclean, _ = scrub_text("GROQ_API_KEY=hunter2-secret-value")
    assert "hunter2-secret-value" not in gclean, gclean
    assert "[REDACTED:SECRET]" in gclean, gclean


def test_live_patterns_redacted():
    """Live key shapes (Groq gsk_, Google AIza, AQ.) are SECRET-redacted,
    and scrub_span covers name/model/session/tool_name metadata."""
    groq = "gsk_" + "abcDEF1234567890xyz"
    google = "AIza" + "SyB4mTxAbcDefGhIjKlMnOpQrStUvWxYz12"
    aq = "AQ." + "Ab8RN6kdexample12345678"
    for secret in (groq, google, aq):
        clean, log = scrub_text(f"key={secret}", span_id="sp_live")
        assert secret not in clean, f"leaked {secret!r} in {clean!r}"
        assert "[REDACTED:SECRET]" in clean, clean
        assert any(e["entity"] == "SECRET" for e in log), log
    span = {"span_id": "sp_9", "input": "ok",
            "name": f"call {groq}", "model": google,
            "session_id": aq, "tool_name": f"t {groq}",
            "args_json": "clean", "result_json": "clean"}
    out, log = scrub_span(span, span_id="sp_9")
    for field in ("name", "model", "session_id", "tool_name"):
        for secret in (groq, google, aq):
            assert secret not in str(out[field]), \
                f"leaked {secret!r} in {field}={out[field]!r}"
    assert log, "span metadata redactions must be logged"
    # Export path scrubs spans before write: no live key survives.
    fail_v = {"task_id": "t9", "verdict": "FAIL", "tenant_id": "A",
              "claims": [{"text": "uses key", "verdict": "FAIL",
                          "evidence_span_ids": ["sp_9"]}]}
    spans = {"sp_9": {"span_id": "sp_9", "output": f"token {groq}",
                      "tenant_id": "A"}}
    ev = materialize_evidence(fail_v, spans=spans, tenant_id="A")
    assert groq not in json.dumps(ev), ev


def test_placeholder_aware_match():
    assert placeholder_aware_equal("[REDACTED:CARD]", "[REDACTED:CARD]")
    assert not placeholder_aware_equal("[REDACTED:CARD]", "4111")
    assert placeholder_aware_contains("paid with [REDACTED:CARD] today",
                                      "[REDACTED:CARD]")
    assert not placeholder_aware_contains("paid cash today",
                                          "[REDACTED:CARD]")


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
