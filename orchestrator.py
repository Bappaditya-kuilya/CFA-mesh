"""M2 orchestrator: tiered runs + content cache + budgets. Stdlib only.

PRD refs: F2 (tiered task matrix, content cache key, budgets, path-filter).

Usage:
  python3 orchestrator.py --tier pr --limit 20
  python3 orchestrator.py --tier nightly --limit 200
  python3 -c "import orchestrator; orchestrator.demo()"
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

# F2 tier matrix. PR: 20 tasks x 2 branches, max_concurrent=10, cache on.
# Nightly: --limit 200 default, max 1000 x 7 branches, max_concurrent=20.
TIER_POLICY = {
    "pr": {"tasks": 20, "branches": 2, "concurrent": 10},
    "nightly": {"limit": 200, "max": 1000, "branches": 7, "concurrent": 20},
}

# F2 budgets: per-trace 8000 tok / $0.05 / 120s / 7 branches
# + global run cap $2 PR / $20 nightly. Exceed -> abort `budget_exceeded`.
BUDGETS = {
    "max_tokens_per_trace": 8000,
    "max_cost_usd": 0.05,
    "timeout_s": 120,
    "max_branches": 7,
    "run_cap_usd": {"pr": 2.0, "nightly": 20.0},
}

# P4 live cost estimate: conservative API rate ($0.50 / 1M tokens).
# Local ollama resampling meters $0 (no billed spend); hosted calls accrue
# total_tokens * LIVE_COST_USD_PER_TOKEN into BudgetEnforcer via add_spend.
LIVE_COST_USD_PER_TOKEN = 0.5e-6

# F2 path-filter: full gate ONLY on prompt/agent/tool-schema changes.
# Anything else (docs, tests, infra) runs the cheap skip path.
FULL_GATE_PATTERNS = (
    r"prompt",
    r"agent",
    r"tool",  # tool schema / wiring: tool_schema, tool-schema, tools.json, ...
    r"harness",
    r"adapter",
    r"verdict",
    r"evals/goldens",
    r"thresholds\.yaml",
)

DEFAULT_CACHE_PATH = os.environ.get("CFA_CACHE", "cache/content.jsonl")
HARNESS_VERSION = "m2-1"


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_cache_key(model_id: str = "", sys_prompt_hash: str = "",
                      user_template_hash: str = "", temp: str | float = 0.0,
                      tools_schema_hash: str = "", rag_corpus_version: str = "",
                      task_id: str = "", branch_config: str = "",
                      git_sha: str = "", harness_version: str = HARNESS_VERSION,
                      nli_version: str = "", thresholds_version: str = "",
                      seed: int = 7) -> str:
    """F2 content cache key (+ F3 seed decode param).

    sha256(model_id|sys_prompt_hash|user_template_hash|temp|tools_schema_hash|
           rag_corpus_version|task_id|branch_config|git_sha|harness_version|
           nli_version|thresholds_version|seed).
    Any of the 6 triggers (model, prompt template, tool schema, corpus,
    system prompt, user state) changes the key -> automatic invalidation.
    temp/seed are decode params (F3: temperature=0, seed=7): any change
    re-resamples instead of reusing a cached verdict.
    """
    joined = "|".join(str(p) for p in (
        model_id, sys_prompt_hash, user_template_hash, temp,
        tools_schema_hash, rag_corpus_version, task_id, branch_config,
        git_sha, harness_version, nli_version, thresholds_version, seed))
    return sha256_str(joined)


class ContentCache:
    """Dict + JSONL content cache.

    DeepEval cache-hit-skip pattern: `get()` before compute; on hit the
    caller skips the model call entirely and reuses the stored verdict.
    JSONL file persists hits across runs; dict is the hot path.
    """

    def __init__(self, path: str = DEFAULT_CACHE_PATH, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._mem: dict = {}
        self.hits = 0
        self.misses = 0
        self._load()

    def _load(self):
        if not self.enabled:
            return
        try:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if "key" in rec and "value" in rec:
                            self._mem[rec["key"]] = rec["value"]
                    except (ValueError, TypeError):
                        continue
        except OSError:
            pass  # cold cache: miss everything, never fail the run on I/O

    def get(self, key: str):
        if not self.enabled:
            self.misses += 1
            return None
        if key in self._mem:
            self.hits += 1
            return self._mem[key]
        self.misses += 1
        return None

    def put(self, key: str, value: dict) -> dict:
        self._mem[key] = value
        if not self.enabled:
            return value
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps({"key": key, "value": value}) + "\n")
        except OSError:
            pass  # cache write failure must not fail the gate
        return value

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return (self.hits / self.total) if self.total else 0.0

    def log_line(self) -> str:
        return (f"cache_hit_rate={self.hit_rate:.2f} "
                f"(hits={self.hits} misses={self.misses})")


class BudgetExceeded(RuntimeError):
    """Raised when any F2 budget trips. Verdict for the trace: FAIL(budget_exceeded)."""

    def __init__(self, reason: str):
        super().__init__(f"budget_exceeded: {reason}")
        self.reason = reason


class BudgetEnforcer:
    """F2 budget enforcer: per-trace tok/cost/time/branches + global run cap."""

    def __init__(self, tier: str = "pr", budgets: dict | None = None):
        self.tier = tier
        b = budgets or BUDGETS
        self.max_tokens = b["max_tokens_per_trace"]
        self.max_cost = b["max_cost_usd"]
        self.timeout_s = b["timeout_s"]
        self.max_branches = b["max_branches"]
        self.run_cap = b["run_cap_usd"][tier]
        self.spent_usd = 0.0

    def check_trace(self, tokens: int = 0, cost_usd: float = 0.0,
                    elapsed_s: float = 0.0, n_branches: int = 0,
                    trace_id: str = "") -> None:
        if tokens > self.max_tokens:
            raise BudgetExceeded(f"trace {trace_id}: tokens {tokens} > {self.max_tokens}")
        if cost_usd > self.max_cost:
            raise BudgetExceeded(f"trace {trace_id}: cost ${cost_usd:.4f} > ${self.max_cost:.2f}")
        if elapsed_s > self.timeout_s:
            raise BudgetExceeded(f"trace {trace_id}: {elapsed_s:.1f}s > {self.timeout_s}s")
        if n_branches > self.max_branches:
            raise BudgetExceeded(f"trace {trace_id}: branches {n_branches} > {self.max_branches}")

    def add_spend(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd
        if self.spent_usd > self.run_cap:
            raise BudgetExceeded(
                f"run cap: spent ${self.spent_usd:.4f} > ${self.run_cap:.2f} ({self.tier})")


def needs_full_gate(changed_files: list[str]) -> bool:
    """F2 path-filter: full gate only on prompt/agent/tool-schema changes."""
    pats = [re.compile(p, re.IGNORECASE) for p in FULL_GATE_PATTERNS]
    return any(p.search(f or "") for f in changed_files for p in pats)


def select_tier(tier: str, n_available: int, limit: int | None = None) -> dict:
    """Resolve tier -> concrete run plan (task count, branches, concurrency).

    PR caps at 20 tasks; nightly defaults to --limit 200, hard max 1000.
    limit=None means tier default (pr 20, nightly 200); explicit limit
    is honored up to the tier cap (pr 20, nightly 1000).
    """
    if tier not in TIER_POLICY:
        raise ValueError(f"unknown tier {tier!r}; want one of {sorted(TIER_POLICY)}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit!r}")
    pol = TIER_POLICY[tier]
    if tier == "pr":
        want = pol["tasks"] if limit is None else min(limit, pol["tasks"])
        n_tasks = min(want, n_available)
        branches = pol["branches"]
    else:
        want = pol["limit"] if limit is None else limit
        n_tasks = min(want, pol["max"], n_available)
        branches = pol["branches"]
    return {"tier": tier, "n_tasks": n_tasks, "branches": branches,
            "concurrent": pol["concurrent"], "cache": True}


def _has_model_key() -> bool:
    """True when a hosted-model key is exported in env (live path allowed).

    GOOGLE_API_KEY / GROQ_API_KEY in os.environ only; .env is NOT read
    here so a checked-in placeholder .env can never flip a run to live.
    No key -> stub compute (today's behavior, zero network).
    """
    return bool(os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("GROQ_API_KEY"))


def _live_compute_for(model_id: str, temp: float = 0.0, seed: int = 7,
                      max_tokens: int = 512):
    """Build compute(task, branch) over complete_with_usage (stdlib only).

    Maps usage -> compute shape: tokens = tokens_in + tokens_out,
    cost estimated via LIVE_COST_USD_PER_TOKEN ($0 for local ollama),
    elapsed_s straight from the client. The caller's check_trace /
    add_spend then meters live spend and aborts budget_exceeded.
    Module-attribute call (not from-import) so tests can monkeypatch
    adapters.model_client.complete_with_usage.
    """
    from adapters import model_client

    def live(task: dict, branch_idx: int) -> dict:
        prompt = task.get("input", task.get("prompt", task.get("task_id", "")))
        usage = model_client.complete_with_usage(
            model_id, str(prompt), temperature=temp, seed=seed,
            max_tokens=max_tokens)
        total = int(usage.get("tokens_in", 0) or 0) \
            + int(usage.get("tokens_out", 0) or 0)
        kind = model_id.split("/")[0] if "/" in model_id else ""
        cost = 0.0 if kind in ("ollama", "stub", "") \
            else total * LIVE_COST_USD_PER_TOKEN
        return {"output": usage.get("text", ""), "tokens": total,
                "cost_usd": cost, "elapsed_s": usage.get("elapsed_s", 0.0)}
    return live


def run(tasks: list[dict], tier: str = "pr", limit: int | None = None,
        changed_files: list[str] | None = None,
        cache: ContentCache | None = None,
        budgets: BudgetEnforcer | None = None,
        git_sha: str = "", model_id: str = "stub",
        compute=None, temp: float = 0.0, seed: int = 7,
        sys_prompt_hash: str = "", user_template_hash: str = "",
        tools_schema_hash: str = "", rag_corpus_version: str = "",
        nli_version: str = "", thresholds_version: str = "",
        max_tokens: int = 512) -> dict:
    """Tiered run with cache-hit-skip + budget enforcement.

    compute(task, branch_idx) -> {"output": str, "tokens": int,
        "cost_usd": float, "elapsed_s": float}. Explicit compute always
    wins. Default: live resampling via complete_with_usage when a model
    key (GOOGLE_API_KEY/GROQ_API_KEY) is exported AND model_id != "stub";
    otherwise the stub echoes the golden's expected_output at zero cost
    (no network, M1-style). temp/seed (F3 defaults 0/7) feed both the
    live call and the cache key. Returns {"results": [...],
    "cache_hit_rate": float, "spent_usd": float}.
    """
    plan = select_tier(tier, len(tasks), limit)
    if changed_files is not None and not needs_full_gate(changed_files):
        return {"plan": plan, "results": [], "skipped": True,
                "reason": "path-filter: no prompt/agent/tool-schema change",
                "cache_hit_rate": 0.0, "spent_usd": 0.0}
    cache = cache or ContentCache(enabled=True)
    budgets = budgets or BudgetEnforcer(tier=tier)
    if compute is None:
        if model_id != "stub" and _has_model_key():
            compute = _live_compute_for(model_id, temp=temp, seed=seed,
                                        max_tokens=max_tokens)
        else:
            compute = (lambda task, b: {
                "output": task.get("expected_output", ""), "tokens": 10,
                "cost_usd": 0.0, "elapsed_s": 0.0})

    chosen = tasks[:plan["n_tasks"]]
    results = []
    for task in chosen:
        tid = task.get("task_id", "?")
        for b in range(plan["branches"]):
            key = content_cache_key(
                model_id=model_id, sys_prompt_hash=sys_prompt_hash,
                user_template_hash=user_template_hash, temp=temp,
                tools_schema_hash=tools_schema_hash,
                rag_corpus_version=rag_corpus_version, task_id=tid,
                branch_config=f"branch_{b}", git_sha=git_sha,
                harness_version=HARNESS_VERSION, nli_version=nli_version,
                thresholds_version=thresholds_version, seed=seed)
            hit = cache.get(key)  # DeepEval pattern: lookup first, skip compute on hit
            if hit is not None:
                results.append({"task_id": tid, "branch": b,
                                "cached": True, **hit})
                continue
            t0 = time.monotonic()
            try:
                out = compute(task, b)
                elapsed = out.get("elapsed_s") or (time.monotonic() - t0)
                budgets.check_trace(tokens=out.get("tokens", 0),
                                    cost_usd=out.get("cost_usd", 0.0),
                                    elapsed_s=elapsed,
                                    n_branches=plan["branches"],
                                    trace_id=tid)
                budgets.add_spend(out.get("cost_usd", 0.0))
                val = {"output": out.get("output", ""),
                       "verdict": "PASS"}
                cache.put(key, val)
                results.append({"task_id": tid, "branch": b,
                                "cached": False, **val})
            except BudgetExceeded as e:
                results.append({"task_id": tid, "branch": b,
                                "verdict": "FAIL", "reason": str(e),
                                "cached": False})
                raise  # abort the run per F2 (global FAIL budget_exceeded)
    return {"plan": plan, "results": results, "skipped": False,
            "cache_hit_rate": round(cache.hit_rate, 4),
            "spent_usd": round(budgets.spent_usd, 4),
            "cache_log": cache.log_line()}


def demo() -> None:
    """Evidence demo: tier select + cache hit/miss + budget abort. No network."""
    import tempfile
    print("== tier select ==")
    print("pr:     ", select_tier("pr", 1000))
    print("nightly:", select_tier("nightly", 1000))
    print("nightly --limit 50:", select_tier("nightly", 1000, limit=50))

    print("== cache hit/miss (DeepEval cache-hit-skip) ==")
    tmp = os.path.join(tempfile.mkdtemp(), "content.jsonl")
    cache = ContentCache(path=tmp)
    tasks = [{"task_id": f"t{i}", "expected_output": f"answer {i}"} for i in range(3)]
    r1 = run(tasks, tier="pr", cache=cache, git_sha="abc123")
    print("run1:", r1["cache_log"], "results:", len(r1["results"]))
    cache2 = ContentCache(path=tmp)  # fresh dict, replays JSONL persistence
    r2 = run(tasks, tier="pr", cache=cache2, git_sha="abc123")
    print("run2:", cache2.log_line(),
          "all_cached:", all(r.get("cached") for r in r2["results"]))
    assert all(r.get("cached") for r in r2["results"]), "run2 must be all cache hits"
    # key includes 12 F2 fields: same inputs -> same key, any trigger flips it
    k1 = content_cache_key(model_id="m", task_id="t", branch_config="b0")
    assert k1 == content_cache_key(model_id="m", task_id="t", branch_config="b0")
    assert k1 != content_cache_key(model_id="m2", task_id="t", branch_config="b0")

    print("== budget abort ==")
    enforcer = BudgetEnforcer(tier="pr")

    def pricey(task, b):
        return {"output": "x", "tokens": 10**9, "cost_usd": 99.0, "elapsed_s": 0.0}

    try:
        run([{"task_id": "t0"}], tier="pr",
            cache=ContentCache(path=os.path.join(tempfile.mkdtemp(), "c.jsonl")),
            budgets=enforcer, compute=pricey)
        print("ERROR: budget did not abort")
    except BudgetExceeded as e:
        print("aborted:", e)

    print("== path-filter ==")
    print("docs-only full gate?", needs_full_gate(["README.md", "docs/x.md"]))
    print("prompt change full gate?", needs_full_gate(["prompts/judge.md"]))
    assert not needs_full_gate(["README.md"])
    assert needs_full_gate(["agents/judge.py"])
    assert needs_full_gate(["schemas/tools.json"])

    print("== live path (stubbed client, no network) ==")
    from adapters import model_client as _mc
    real_complete = _mc.complete_with_usage

    def _fake_complete(model, prompt, **kw):
        assert kw.get("temperature") == 0.0 and kw.get("seed") == 7, kw
        return {"text": f"live:{prompt}", "tokens_in": 5000,
                "tokens_out": 3000, "elapsed_s": 1.5}

    old_key = os.environ.get("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = "demo-sentinel"
    _mc.complete_with_usage = _fake_complete
    try:
        ltasks = [{"task_id": "live0", "input": "say hi"}]
        live_budgets = BudgetEnforcer(tier="pr")
        lc = os.path.join(tempfile.mkdtemp(), "live.jsonl")
        rl = run(ltasks, tier="pr", cache=ContentCache(path=lc),
                 budgets=live_budgets, model_id="google/gemini-2.5-flash")
        assert all(r.get("cached") is False for r in rl["results"]), rl
        assert live_budgets.spent_usd > 0, "live usage must meter spend"
        assert rl["results"][0]["output"] == "live:say hi", rl["results"][0]
        # temp+seed in key: same inputs hit, any decode change misses
        kA = content_cache_key(model_id="m", task_id="t",
                               branch_config="b0", temp=0.0, seed=7)
        assert kA == content_cache_key(model_id="m", task_id="t",
                                       branch_config="b0", temp=0.0, seed=7)
        assert kA != content_cache_key(model_id="m", task_id="t",
                                       branch_config="b0", temp=0.5, seed=7)
        assert kA != content_cache_key(model_id="m", task_id="t",
                                       branch_config="b0", temp=0.0, seed=8)
        r_hit = run(ltasks, tier="pr", cache=ContentCache(path=lc),
                    budgets=BudgetEnforcer(tier="pr"),
                    model_id="google/gemini-2.5-flash")
        assert all(r.get("cached") for r in r_hit["results"]), "repeat must hit"
        print("live spend=$%.6f results=%d repeat_all_cached=True"
              % (live_budgets.spent_usd, len(rl["results"])))
        # no key -> stub again even with a live model_id
        del os.environ["GOOGLE_API_KEY"]
        rs = run(ltasks, tier="pr",
                 cache=ContentCache(
                     path=os.path.join(tempfile.mkdtemp(), "s.jsonl")),
                 model_id="google/gemini-2.5-flash")
        assert rs["spent_usd"] == 0.0 and \
            rs["results"][0]["output"] == "", rs  # stub: no expected_output
        print("no-key falls back to stub: OK")
    finally:
        _mc.complete_with_usage = real_complete
        if old_key is None:
            os.environ.pop("GOOGLE_API_KEY", None)
        else:
            os.environ["GOOGLE_API_KEY"] = old_key
    print("demo OK")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M2 tiered CFA run (stdlib only)")
    ap.add_argument("--tier", default="pr", choices=sorted(TIER_POLICY))
    ap.add_argument("--limit", type=int, default=None,
                      help="task cap; None = tier default (pr 20, nightly 200)")
    ap.add_argument("--tasks", default="evals/goldens/v1.jsonl")
    ap.add_argument("--changed", default="",
                    help="comma-separated changed files for path-filter")
    ap.add_argument("--git-sha", default=os.environ.get("GIT_SHA", ""))
    ap.add_argument("--model", default="stub")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="resample temperature (F3 default 0.0; part of cache key)")
    ap.add_argument("--seed", type=int, default=7,
                    help="resample seed (F3 default 7; part of cache key)")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args(argv)

    try:
        with open(args.tasks) as f:
            tasks = [json.loads(l) for l in f if l.strip()]
    except OSError as e:
        print(f"no tasks file {args.tasks}: {e}", file=sys.stderr)
        return 2
    changed = [c for c in args.changed.split(",") if c] or None
    try:
        out = run(tasks, tier=args.tier, limit=args.limit,
                  changed_files=changed, git_sha=args.git_sha,
                  model_id=args.model, temp=args.temp, seed=args.seed,
                  max_tokens=args.max_tokens)
    except BudgetExceeded as e:
        print(json.dumps({"verdict": "FAIL", "reason": str(e)}))
        return 1
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))
    print(f"{out['plan']['n_tasks']} tasks x {out['plan']['branches']} branches "
          f"({out['plan']['tier']}, conc={out['plan']['concurrent']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
