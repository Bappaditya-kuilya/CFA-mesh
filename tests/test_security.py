"""M5 security tests: budgets, retention, lockout, tenant isolation + scrub.

Stdlib only, pytest-optional: plain `test_*` functions (same style as
test_interventions.py) plus a `__main__` runner, so both
`pytest tests/test_security.py` and `python tests/test_security.py` work.

PRD refs: §5/F2 (budgets), §7.5 (retention purge), §7.2 (lockout persists,
tenant isolation), §7.1 (scrub + placeholder-aware match).
"""
import os
import sys
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
                   placeholder_aware_equal, scrub_text)
from store import BUDGETS, Store, over_budget


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
        for _ in range(4):
            try:
                a1.authenticate(bad)
            except LockedOut:
                raise AssertionError("locked out before 5 fails")
            except AuthError:
                pass
        assert not a1.is_locked_out(hash_key(bad))
        try:
            a1.authenticate(bad)
            raise AssertionError("5th bad attempt should lock out")
        except LockedOut:
            pass
        assert a1.is_locked_out(hash_key(bad))
        # Persists across reopen (new connection, same sqlite file).
        a2 = AuthStore(db)
        assert a2.is_locked_out(hash_key(bad))
        try:
            a2.authenticate(bad)
            raise AssertionError("reopened store should still lock out")
        except LockedOut:
            pass
        # Real key works and is constant-time verified.
        got = a1.authenticate("user-secret-1")
        assert got["tenant_id"] == "A" and got["role"] == "member"
        assert verify_bearer("user-secret-1",
                             hash_key("user-secret-1"))
        assert not verify_bearer("user-secret-2",
                                 hash_key("user-secret-1"))
        # User keys die with the user; unknown role rejected.
        assert a1.revoke_user_keys("u1") == 1
        try:
            require_role("viewer", "write")
            raise AssertionError("viewer must not write")
        except AuthError:
            pass
        a1.close()
        a2.close()


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
        st.close()
    try:
        check_tenant("A", "B")
        raise AssertionError("cross-tenant access must raise")
    except AuthError:
        pass
    check_tenant("A", "A")  # same tenant: no raise


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
