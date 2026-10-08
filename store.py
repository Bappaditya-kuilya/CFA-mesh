"""M1 store: sqlite3 stdlib + JSONL fallback. No Postgres, no SQLAlchemy.

PRD refs: F1 (one trace_id per task, JSONL record), F7 (tables
trajectories/spans/branches/verdicts), section 5 (WAL, busy_timeout=5000,
BEGIN IMMEDIATE, single-writer), 3.2 (indexes, tenant_id enforced).

Fallback: if sqlite raises OperationalError("database is locked"),
the record is appended to a JSONL file (Langfuse-dataset-style local
rows) instead of raising, so eval runs never lose traces.
"""
import json
import os
import sqlite3
import time
import uuid

DEFAULT_DB = os.environ.get("CFA_DB", "cfa.db")
DEFAULT_JSONL = os.environ.get("CFA_JSONL", "traces.jsonl")
DEFAULT_TENANT = "default"

# F2 budgets: per-trace caps + global run caps. Exceed -> abort
# `budget_exceeded` FAIL. Pure check so M2 orchestrator + tests share it.
BUDGETS = {
    "max_tokens_per_trace": 8000,
    "max_cost_usd": 0.05,
    "timeout_s": 120,
    "max_branches": 7,
    "run_cap_pr_usd": 2.0,
    "run_cap_nightly_usd": 20.0,
}


def over_budget(tokens_in: int = 0, tokens_out: int = 0,
                cost_usd: float = 0.0, latency_s: float = 0.0,
                branches: int = 0, run_cost_usd: float = 0.0,
                tier: str = "pr") -> list:
    """Return reason codes (empty == within budget)."""
    reasons = []
    if tokens_in + tokens_out > BUDGETS["max_tokens_per_trace"]:
        reasons.append("tokens_exceeded")
    if cost_usd > BUDGETS["max_cost_usd"]:
        reasons.append("cost_exceeded")
    if latency_s > BUDGETS["timeout_s"]:
        reasons.append("timeout_exceeded")
    if branches > BUDGETS["max_branches"]:
        reasons.append("branches_exceeded")
    cap = (BUDGETS["run_cap_nightly_usd"] if tier == "nightly"
           else BUDGETS["run_cap_pr_usd"])
    if run_cost_usd > cap:
        reasons.append("budget_exceeded")
    return reasons

SCHEMA = """
CREATE TABLE IF NOT EXISTS trajectories (
  trace_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  input TEXT,
  output TEXT,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spans (
  span_id TEXT PRIMARY KEY,
  trace_id TEXT NOT NULL REFERENCES trajectories(trace_id),
  parent_id TEXT,
  kind TEXT,
  name TEXT,
  input TEXT,
  output TEXT,
  model TEXT,
  tokens_in INTEGER,
  tokens_out INTEGER,
  latency_ms REAL,
  cost_usd REAL,
  session_id TEXT,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  started_at TEXT,
  ended_at TEXT,
  tool_name TEXT,
  args_json TEXT,
  result_json TEXT,
  exit_code INTEGER
);
CREATE TABLE IF NOT EXISTS branches (
  branch_id TEXT PRIMARY KEY,
  trace_id TEXT NOT NULL REFERENCES trajectories(trace_id),
  op TEXT NOT NULL,
  input TEXT,
  output TEXT,
  s REAL,
  phi REAL,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verdicts (
  trace_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  s REAL,
  phi REAL,
  faithful_ratio REAL,
  verbalizes INTEGER,
  verdict TEXT,
  provenance_json TEXT,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_branches_trace ON branches(trace_id);
CREATE INDEX IF NOT EXISTS idx_verdicts_task ON verdicts(task_id);
CREATE INDEX IF NOT EXISTS idx_traj_task ON trajectories(task_id);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(SCHEMA)
    return conn


def _is_locked(exc: Exception) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower()


class Store:
    """Single-writer store. Writes use BEGIN IMMEDIATE; locked DB falls back to JSONL."""

    def __init__(self, db_path: str = DEFAULT_DB, jsonl_path: str = DEFAULT_JSONL):
        self.db_path = db_path
        self.jsonl_path = jsonl_path
        self._conn = connect(db_path)

    def _fallback_append(self, table: str, row: dict) -> dict:
        rec = {"_fallback_table": table, **row}
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        rec["_fallback"] = True
        return rec

    def _insert(self, table: str, row: dict) -> dict:
        cols = ",".join(row.keys())
        qs = ",".join("?" for _ in row)
        try:
            self._conn.execute("BEGIN IMMEDIATE;")
            self._conn.execute(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({qs})",
                               tuple(row.values()))
            self._conn.execute("COMMIT;")
            return row
        except Exception as e:
            try:
                self._conn.execute("ROLLBACK;")
            except Exception:
                pass
            if _is_locked(e):
                return self._fallback_append(table, row)
            raise

    def save_trace(self, trace_id, task_id, input, output,
                   tenant_id: str = DEFAULT_TENANT, spans: list | None = None) -> dict:
        row = {"trace_id": trace_id, "task_id": task_id, "input": input,
               "output": output, "tenant_id": tenant_id, "created_at": _now()}
        out = self._insert("trajectories", row)
        # Always mirror the M1 JSONL record per F1: {trace_id,task_id,input,output,spans}
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps({"trace_id": trace_id, "task_id": task_id,
                                "input": input, "output": output,
                                "spans": spans or []}) + "\n")
        for s in (spans or []):
            self.save_span(trace_id=trace_id, tenant_id=tenant_id, **s)
        return out

    def save_span(self, span_id: str | None = None, trace_id: str = "",
                  parent_id: str | None = None, kind: str = "", name: str = "",
                  input: str = "", output: str = "", model: str = "",
                  tokens_in: int = 0, tokens_out: int = 0, latency_ms: float = 0.0,
                  cost_usd: float = 0.0, session_id: str = "",
                  tenant_id: str = DEFAULT_TENANT, started_at: str = "",
                  ended_at: str = "", tool_name: str = "", args_json: str = "",
                  result_json: str = "", exit_code: int | None = None) -> dict:
        return self._insert("spans", {
            "span_id": span_id or f"sp_{uuid.uuid4().hex[:12]}",
            "trace_id": trace_id, "parent_id": parent_id, "kind": kind,
            "name": name, "input": input, "output": output, "model": model,
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "latency_ms": latency_ms, "cost_usd": cost_usd,
            "session_id": session_id, "tenant_id": tenant_id,
            "started_at": started_at or _now(), "ended_at": ended_at or _now(),
            "tool_name": tool_name, "args_json": args_json,
            "result_json": result_json, "exit_code": exit_code,
        })

    def save_branch(self, trace_id, op, input, output,
                    s: float | None = None, phi: float | None = None,
                    tenant_id: str = DEFAULT_TENANT,
                    branch_id: str | None = None) -> dict:
        return self._insert("branches", {
            "branch_id": branch_id or f"br_{uuid.uuid4().hex[:12]}",
            "trace_id": trace_id, "op": op, "input": input, "output": output,
            "s": s, "phi": phi, "tenant_id": tenant_id, "created_at": _now(),
        })

    def save_verdict(self, trace_id, task_id, s: float | None = None,
                     phi: float | None = None, faithful_ratio: float | None = None,
                     verbalizes: int | None = None, verdict: str = "",
                     provenance: dict | None = None,
                     tenant_id: str = DEFAULT_TENANT) -> dict:
        return self._insert("verdicts", {
            "trace_id": trace_id, "task_id": task_id, "s": s, "phi": phi,
            "faithful_ratio": faithful_ratio, "verbalizes": verbalizes,
            "verdict": verdict,
            "provenance_json": json.dumps(provenance or {}),
            "tenant_id": tenant_id, "created_at": _now(),
        })

    # Reads always scope tenant_id per 3.2.
    def get_trace(self, trace_id: str, tenant_id: str = DEFAULT_TENANT) -> dict | None:
        cur = self._conn.execute(
            "SELECT trace_id,task_id,input,output FROM trajectories WHERE trace_id=? AND tenant_id=?",
            (trace_id, tenant_id))
        r = cur.fetchone()
        if not r:
            return None
        spans = self._conn.execute(
            "SELECT span_id,kind,name,input,output FROM spans WHERE trace_id=? AND tenant_id=?",
            (trace_id, tenant_id)).fetchall()
        branches = self._conn.execute(
            "SELECT branch_id,op,input,output,s,phi FROM branches WHERE trace_id=? AND tenant_id=?",
            (trace_id, tenant_id)).fetchall()
        return {"trace_id": r[0], "task_id": r[1], "input": r[2], "output": r[3],
                "spans": spans, "branches": branches}

    def purge_older_than(self, days: int = 30) -> dict:
        """Retention purge (§7.5, default 30d). Children first (FK-safe)."""
        cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                               time.gmtime(time.time() - days * 86400))
        counts: dict = {}
        try:
            self._conn.execute("BEGIN IMMEDIATE;")
            for table in ("spans", "branches", "verdicts"):
                cur = self._conn.execute(
                    f"DELETE FROM {table} WHERE trace_id IN "
                    "(SELECT trace_id FROM trajectories WHERE created_at < ?)",
                    (cutoff,))
                counts[table] = cur.rowcount
            cur = self._conn.execute(
                "DELETE FROM trajectories WHERE created_at < ?", (cutoff,))
            counts["trajectories"] = cur.rowcount
            self._conn.execute("COMMIT;")
            return counts
        except Exception:
            try:
                self._conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


# Module-level convenience API backed by a lazily created default store.
_default: Store | None = None


def _store() -> Store:
    global _default
    if _default is None:
        _default = Store()
    return _default


def save_trace(trace_id, task_id, input, output, tenant_id: str = DEFAULT_TENANT,
               spans: list | None = None, store: Store | None = None) -> dict:
    return (store or _store()).save_trace(trace_id, task_id, input, output, tenant_id, spans)


def save_branch(trace_id, op, input, output, s=None, phi=None,
                tenant_id: str = DEFAULT_TENANT, branch_id=None,
                store: Store | None = None) -> dict:
    return (store or _store()).save_branch(trace_id, op, input, output, s, phi,
                                           tenant_id, branch_id)


def save_verdict(trace_id, task_id, s=None, phi=None, faithful_ratio=None,
                 verbalizes=None, verdict="", provenance=None,
                 tenant_id: str = DEFAULT_TENANT, store: Store | None = None) -> dict:
    return (store or _store()).save_verdict(trace_id, task_id, s, phi, faithful_ratio,
                                            verbalizes, verdict, provenance, tenant_id)
