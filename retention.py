"""M5 retention/sampling: 30d purge + tail keep. Pure functions, stdlib only.

PRD refs: section 7.5 (30d purge cron, metrics computed before sampling,
tail keeps ``errors + slow>2000ms + 10% baseline``, manual delete),
section 5 (required test ``test_retention_purge``).

Design: functions operate on plain sqlite row dicts / a sqlite3 connection,
so no OTel collector is needed. Pinned ``evidence_bank`` rows are never
purged here -- callers must exclude FAIL-evidence trace_ids from purge.

Row shape (lenient; missing keys default to keep-neutral)::

    {"trace_id": str, "created_at": "ISO-8601Z", "is_error": bool,
     "verdict": "PASS|FAIL|...", "latency_ms": float, "status": str}
"""

import datetime
import hashlib
import sqlite3

RETENTION_DAYS = 30
SLOW_MS = 2000.0
BASELINE_FRAC = 0.10


def parse_ts(s: str) -> datetime.datetime:
    """Parse ISO-8601 (accepts trailing Z). Unparseable -> epoch (expired)."""
    try:
        s = str(s or "").strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt
    except Exception:
        return datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def _now_utc(now: datetime.datetime | None) -> datetime.datetime:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    return now


def is_expired(created_at: str, now: datetime.datetime | None = None,
               days: int = RETENTION_DAYS) -> bool:
    cutoff = _now_utc(now) - datetime.timedelta(days=days)
    return parse_ts(created_at) < cutoff


def _is_error(row: dict) -> bool:
    if row.get("is_error") is True:
        return True
    if str(row.get("status", "")).lower() in ("error", "failed", "fail"):
        return True
    if str(row.get("verdict", "")).upper() in ("FAIL", "ERROR"):
        return True
    if row.get("exit_code") not in (None, 0):
        return True
    return False


def _latency(row: dict) -> float:
    for k in ("latency_ms", "max_latency_ms", "duration_ms", "p95_ms"):
        try:
            v = row.get(k)
            if v is not None:
                return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def _baseline_keep(trace_id: str, frac: float = BASELINE_FRAC) -> bool:
    """Deterministic ~frac baseline sample: sha256(trace_id) % 100 < frac*100."""
    if frac <= 0:
        return False
    h = hashlib.sha256(str(trace_id).encode()).hexdigest()
    return (int(h[:8], 16) % 100) < round(frac * 100)


def tail_keep(row: dict, frac: float = BASELINE_FRAC) -> bool:
    """Tail-sampling predicate: errors + slow>2000ms + 10% baseline."""
    if _is_error(row):
        return True
    if _latency(row) > SLOW_MS:
        return True
    return _baseline_keep(str(row.get("trace_id", "")), frac)


def partition(rows: list[dict], now: datetime.datetime | None = None,
              days: int = RETENTION_DAYS,
              frac: float = BASELINE_FRAC) -> tuple[list[dict], list[dict]]:
    """Split rows into (keep, purge).

    - Fresh (<days old): always keep.
    - Expired: keep only the tail sample, purge the rest.
    """
    now = _now_utc(now)
    keep, purge = [], []
    for r in rows:
        if not is_expired(str(r.get("created_at", "")), now, days):
            keep.append(r)
        elif tail_keep(r, frac):
            keep.append(r)
        else:
            purge.append(r)
    return keep, purge


def summarize(rows: list[dict], now: datetime.datetime | None = None,
              days: int = RETENTION_DAYS,
              frac: float = BASELINE_FRAC) -> dict:
    """Metrics computed BEFORE sampling (PRD 7.5 requires this order)."""
    now = _now_utc(now)
    keep, purge = partition(rows, now, days, frac)
    expired = [r for r in rows if is_expired(str(r.get("created_at", "")), now, days)]
    tail_kept = [r for r in expired if r in keep] if expired else []
    return {
        "total": len(rows),
        "expired": len(expired),
        "keep": len(keep),
        "purge": len(purge),
        "tail_kept_expired": len(tail_kept),
        "retention_days": days,
        "slow_ms": SLOW_MS,
        "baseline_frac": frac,
    }


def purge_trajectories(conn: sqlite3.Connection,
                       now: datetime.datetime | None = None,
                       days: int = RETENTION_DAYS,
                       keep_trace_ids: set | frozenset = frozenset(),
                       table: str = "trajectories",
                       ts_col: str = "created_at",
                       id_col: str = "trace_id") -> int:
    """DELETE expired rows from a sqlite table, except keep_trace_ids.

    Returns the number of deleted rows. Pinned evidence trace_ids must be
    passed in keep_trace_ids by the caller (evidence_bank is never purged).
    """
    now = _now_utc(now)
    cutoff = (now - datetime.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    keep = set(keep_trace_ids or ())
    try:  # flush any implicit txn (e.g. uncommitted executemany) before BEGIN
        conn.commit()
    except Exception:
        pass
    conn.execute("BEGIN IMMEDIATE;")
    try:
        if keep:
            qs = ",".join("?" for _ in keep)
            cur = conn.execute(
                f"DELETE FROM {table} WHERE {ts_col} < ?"
                f" AND {id_col} NOT IN ({qs})",
                (cutoff, *keep),
            )
        else:
            cur = conn.execute(
                f"DELETE FROM {table} WHERE {ts_col} < ?", (cutoff,))
        n = cur.rowcount if cur.rowcount is not None else 0
        conn.execute("COMMIT;")
        return int(n)
    except Exception:
        try:
            conn.execute("ROLLBACK;")
        except Exception:
            pass
        raise
