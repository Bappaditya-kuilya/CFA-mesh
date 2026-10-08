"""M5 UI auth RBAC stub (stdlib only).

PRD refs: §7.2 (system key for CI vs user keys that die with the user,
roles admin/member/viewer, persistent lockout 5 fails / 5 min,
``WHERE tenant_id=?`` on every query), §3.2 (tenant_id enforced).

Lockout state lives in sqlite so it survives process restarts
(test_lockout_persists). Bearer comparison uses hmac.compare_digest
on sha256 digests (never the raw secret).
"""
import hashlib
import hmac
import sqlite3
import time

POLICY_VERSION = "auth-rbac-v1"

MAX_FAILS = 5
LOCKOUT_WINDOW_S = 5 * 60

ROLES = ("admin", "member", "viewer")
_PERMISSIONS = {
    "admin": frozenset({"read", "write", "manage"}),
    "member": frozenset({"read", "write"}),
    "viewer": frozenset({"read"}),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
  key_hash TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  user_id TEXT NOT NULL DEFAULT '',
  role TEXT NOT NULL DEFAULT 'viewer',
  revoked INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS auth_failures (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  key_id TEXT NOT NULL,
  attempted_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_failures_key_time ON auth_failures(key_id, attempted_at);
"""


class AuthError(Exception):
    pass


class LockedOut(AuthError):
    pass


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def verify_bearer(provided: str, expected_hash: str) -> bool:
    """Constant-time bearer check. Never touches raw stored secrets."""
    if not isinstance(provided, str) or not provided:
        return False
    try:
        digest = hash_key(provided)
    except Exception:
        return False
    return hmac.compare_digest(digest, expected_hash)


def can(role: str, action: str) -> bool:
    return action in _PERMISSIONS.get(role, frozenset())


def require_role(role: str, action: str) -> str:
    if not can(role, action):
        raise AuthError(f"role {role!r} may not {action!r}")
    return role


def require_tenant(tenant_id: str) -> str:
    if not tenant_id:
        raise AuthError("tenant_id required")
    return tenant_id


def tenant_where(column: str = "tenant_id") -> str:
    """Tenant enforcement fragment: ``... WHERE <fragment>`` + bind tenant_id.

    Usage: ``conn.execute(f"SELECT ... WHERE {tenant_where()}",
    (..., tenant_id))`` — mirrors store.py reads per §3.2.
    """
    return f"{column} = ?"


def check_tenant(record_tenant: str, request_tenant: str) -> None:
    if record_tenant != request_tenant:
        raise AuthError("cross-tenant access denied")


class AuthStore:
    """Persistent API keys + lockout counters (sqlite, stdlib only)."""

    def __init__(self, db_path: str = "cfa_auth.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, timeout=5.0,
                                     check_same_thread=False)
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.executescript(SCHEMA)

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    # -- keys: system (CI) vs user (die with user) -------------------------
    def create_key(self, key: str, kind: str = "user",
                   tenant_id: str = "default", user_id: str = "",
                   role: str = "viewer") -> dict:
        if kind not in ("system", "user"):
            raise AuthError("kind must be 'system' or 'user'")
        if role not in ROLES:
            raise AuthError(f"unknown role {role!r}")
        row = {"key_hash": hash_key(key), "kind": kind,
               "tenant_id": require_tenant(tenant_id), "user_id": user_id,
               "role": role, "revoked": 0, "created_at": time.time()}
        self._conn.execute(
            "INSERT OR REPLACE INTO api_keys "
            "(key_hash,kind,tenant_id,user_id,role,revoked,created_at)"
            " VALUES (?,?,?,?,?,?,?)", tuple(row.values()))
        self._conn.commit()
        return row

    def revoke_user_keys(self, user_id: str) -> int:
        """User keys die with the user; system keys are untouched."""
        cur = self._conn.execute(
            "UPDATE api_keys SET revoked=1 WHERE user_id=? AND kind='user'",
            (user_id,))
        self._conn.commit()
        return cur.rowcount

    # -- lockout: persistent counter, 5 fails / 5 min ----------------------
    def record_failure(self, key_id: str, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self._conn.execute("DELETE FROM auth_failures WHERE attempted_at < ?",
                           (now - LOCKOUT_WINDOW_S,))
        self._conn.execute(
            "INSERT INTO auth_failures (key_id, attempted_at) VALUES (?,?)",
            (key_id, now))
        self._conn.commit()

    def clear_failures(self, key_id: str) -> None:
        self._conn.execute("DELETE FROM auth_failures WHERE key_id=?",
                           (key_id,))
        self._conn.commit()

    def recent_failures(self, key_id: str, now: float | None = None) -> int:
        now = time.time() if now is None else now
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM auth_failures WHERE key_id=? AND attempted_at > ?",
            (key_id, now - LOCKOUT_WINDOW_S))
        return int(cur.fetchone()[0])

    def is_locked_out(self, key_id: str, now: float | None = None) -> bool:
        return self.recent_failures(key_id, now) >= MAX_FAILS

    # -- authenticate ------------------------------------------------------
    def authenticate(self, bearer: str, now: float | None = None) -> dict:
        """Returns {tenant_id, role, kind, user_id} or raises.

        Unknown/revoked/wrong tokens count as failures under the
        sha256 of the presented token (slows targeted brute force).
        """
        now = time.time() if now is None else now
        kid = hash_key(bearer) if isinstance(bearer, str) else ""
        if self.is_locked_out(kid, now):
            raise LockedOut("locked out: 5 fails / 5min")
        cur = self._conn.execute(
            "SELECT key_hash,kind,tenant_id,user_id,role,revoked FROM api_keys"
            " WHERE key_hash=?", (kid,))
        row = cur.fetchone()
        ok = (row is not None and not row[5]
              and verify_bearer(bearer, row[0]))
        if not ok:
            self.record_failure(kid, now)
            if self.is_locked_out(kid, now):
                raise LockedOut("locked out: 5 fails / 5min")
            raise AuthError("invalid bearer")
        self.clear_failures(kid)
        return {"tenant_id": row[2], "role": row[4], "kind": row[1],
                "user_id": row[3]}
