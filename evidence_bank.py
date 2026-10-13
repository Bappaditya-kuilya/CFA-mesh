"""M5 evidence bank: immutable INSERT-only store + provenance writer. Stdlib only.

PRD refs: F7 (``evidence_bank`` table, FAIL evidence materialized into the
export at gate time, ``provenance_json={nli_version,emb_version,judge_model,
judge_id,cited_hashes,seed,harness_version}``), section 5 (WAL,
busy_timeout=5000, BEGIN IMMEDIATE single-writer).

Schema (superset of the task brief, matching PRD F7)::

    evidence_bank(evidence_id, task_id, kind, content_hash, content, pinned_at)

INSERT-only: this module exposes INSERT + SELECT only. There is deliberately
no UPDATE/DELETE API; retention must never purge pinned FAIL evidence
(caller filters it out before purging trajectories).
"""

import hashlib
import json
import sqlite3
import time
import uuid

BANK_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_bank (
  evidence_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  content TEXT NOT NULL,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  pinned_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_task ON evidence_bank(task_id);
CREATE INDEX IF NOT EXISTS idx_evidence_hash ON evidence_bank(content_hash);
CREATE INDEX IF NOT EXISTS idx_evidence_tenant ON evidence_bank(tenant_id);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_bank(conn: sqlite3.Connection) -> None:
    """Create the evidence_bank table/indexes if missing (idempotent)."""
    conn.executescript(BANK_SCHEMA)
    # Migrate pre-P2 DBs lacking tenant_id.
    try:
        cols = [r[1] for r in conn.execute(
            "PRAGMA table_info(evidence_bank)").fetchall()]
        if "tenant_id" not in cols:
            conn.execute("ALTER TABLE evidence_bank ADD COLUMN "
                         "tenant_id TEXT NOT NULL DEFAULT 'default'")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_tenant "
                         "ON evidence_bank(tenant_id)")
    except Exception:
        pass


def _conn_of(target) -> sqlite3.Connection:
    # Accept a raw sqlite3 connection or a store.Store (has _conn).
    if isinstance(target, sqlite3.Connection):
        return target
    conn = getattr(target, "_conn", None)
    if isinstance(conn, sqlite3.Connection):
        return conn
    raise TypeError("target must be sqlite3.Connection or store.Store")


def pin_evidence(target, task_id: str, kind: str, content: str,
                 evidence_id: str | None = None,
                 tenant_id: str = "default") -> dict:
    """INSERT-only pin of one evidence row. Returns the inserted row dict.

    Raises sqlite3.IntegrityError on duplicate evidence_id (no REPLACE).
    """
    conn = _conn_of(target)
    ensure_bank(conn)
    row = {
        "evidence_id": evidence_id or f"ev_{uuid.uuid4().hex[:12]}",
        "task_id": task_id,
        "kind": kind,
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "content": content,
        "tenant_id": tenant_id or "default",
        "pinned_at": _now(),
    }
    cols = ",".join(row.keys())
    qs = ",".join("?" for _ in row)
    try:  # flush any implicit txn before BEGIN IMMEDIATE
        conn.commit()
    except Exception:
        pass
    conn.execute("BEGIN IMMEDIATE;")
    try:
        conn.execute(
            f"INSERT INTO evidence_bank ({cols}) VALUES ({qs})",
            tuple(row.values()),
        )
        conn.execute("COMMIT;")
    except Exception:
        try:
            conn.execute("ROLLBACK;")
        except Exception:
            pass
        raise
    return row


def get_evidence(target, evidence_id: str,
                 tenant_id: str | None = None) -> dict | None:
    conn = _conn_of(target)
    ensure_bank(conn)
    if tenant_id is None:
        cur = conn.execute(
            "SELECT evidence_id,task_id,kind,content_hash,content,"
            "tenant_id,pinned_at"
            " FROM evidence_bank WHERE evidence_id=?",
            (evidence_id,),
        )
    else:
        cur = conn.execute(
            "SELECT evidence_id,task_id,kind,content_hash,content,"
            "tenant_id,pinned_at"
            " FROM evidence_bank WHERE evidence_id=? AND tenant_id=?",
            (evidence_id, tenant_id),
        )
    r = cur.fetchone()
    if not r:
        return None
    return dict(zip(("evidence_id", "task_id", "kind", "content_hash",
                     "content", "tenant_id", "pinned_at"), r))


def list_for_task(target, task_id: str,
                  tenant_id: str | None = None) -> list[dict]:
    """All pinned evidence rows for one task (gate-time export source)."""
    conn = _conn_of(target)
    ensure_bank(conn)
    if tenant_id is None:
        cur = conn.execute(
            "SELECT evidence_id,task_id,kind,content_hash,content,"
            "tenant_id,pinned_at"
            " FROM evidence_bank WHERE task_id=? ORDER BY pinned_at",
            (task_id,),
        )
    else:
        cur = conn.execute(
            "SELECT evidence_id,task_id,kind,content_hash,content,"
            "tenant_id,pinned_at"
            " FROM evidence_bank WHERE task_id=? AND tenant_id=? "
            "ORDER BY pinned_at",
            (task_id, tenant_id),
        )
    return [dict(zip(("evidence_id", "task_id", "kind", "content_hash",
                      "content", "tenant_id", "pinned_at"), r))
            for r in cur.fetchall()]


def export_fail_bundle(target, task_id: str,
                       tenant_id: str | None = None) -> str:
    """Materialize FAIL evidence for one task as self-contained JSONL.

    Per F7 the gate export must be replayable from the bundle alone
    (UI link is convenience only), so each line carries its content_hash.
    When tenant_id is given only that tenant's rows are exported.
    """
    rows = list_for_task(target, task_id, tenant_id=tenant_id)
    return "".join(json.dumps(r) + "\n" for r in rows)


def build_provenance(nli_version: str = "", emb_version: str = "",
                     judge_model: str = "", judge_id: str = "",
                     cited_hashes: list | None = None, seed: int = 7,
                     harness_version: str = "") -> dict:
    """Build the F7 provenance_json payload stored on verdicts."""
    return {
        "nli_version": nli_version,
        "emb_version": emb_version,
        "judge_model": judge_model,
        "judge_id": judge_id,
        "cited_hashes": list(cited_hashes or []),
        "seed": seed,
        "harness_version": harness_version,
    }
