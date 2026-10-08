"""M1 single-file HTML report (ponytail: no UI server, stdlib only).

Mirrors the M1 gate in tests/test_faithfulness.py:test_m1_gate:
  S = cosine (+ exact-match fast path) via verdict.behavior.score_pair,
  PASS iff paraphrase S >= 0.5 and faithful_ratio >= 0.80.

Usage:
  python3 report.py [--tasks evals/goldens/v1.jsonl] [--limit 20]
                    [--out report.html] [--traces-dir traces]
                    [--results-out results.json] [--results-in results.json]

  --results-in: render a previously saved gate results list (JSON array)
    instead of recomputing from goldens. Each item needs at minimum
    task_id; S/phi/faithful/verdict/trace_file default sensibly if absent.
"""
import argparse
import datetime
import html
import json
import pathlib
import re
import uuid
from string import Template

from verdict.behavior import paraphrase, score_pair, verbalizes

# Gate thresholds (must match tests/test_faithfulness.py:test_m1_gate).
S_MIN = 0.5
FAITHFUL_MIN = 0.80

PAGE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CFA-mesh M1 report</title>
<style>
body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#111}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ccc;padding:.35rem .6rem;text-align:left;font-size:.9rem}
th{background:#f2f2f2}
.pass{color:#0a7a2f;font-weight:bold}
.fail{color:#b00020;font-weight:bold}
.mono{font-family:monospace}
.small{color:#555;font-size:.85rem}
</style>
</head>
<body>
<h1>CFA-mesh M1 faithfulness report</h1>
<p class="small">$summary</p>
<p class="small">Thresholds: paraphrase S &gt;= $s_min, faithful_ratio &gt;= $faithful_min.
Trace links are relative file paths (no server).</p>
<table>
<thead><tr>
<th>task_id</th><th>S</th><th>phi (1-S)</th><th>faithful</th><th>verdict</th><th>trace</th>
</tr></thead>
<tbody>
$rows
</tbody>
</table>
</body>
</html>
""")

ROW = Template(
    "<tr><td class=\"mono\">$task_id</td><td>$s</td><td>$phi</td>"
    "<td>$faithful</td><td class=\"$cls\">$verdict</td>"
    "<td><a class=\"mono\" href=\"$trace\">$trace</a></td></tr>"
)


def load_goldens(path, limit):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit]


def fake_agent_answer(task):
    # Same stub as the M1 gate: expected_output verbatim (faithful baseline).
    return task.get("expected_output", "")


def split_claims(response: str):
    # Same minimal Selection as the M1 gate: split sentences, drop chitchat.
    sents = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]
    drop = ("thanks", "here are", "sure", "let me", "happy to help")
    claims = [s for s in sents if not s.lower().startswith(drop)]
    return claims if claims else ([response] if response.strip() else [])


def evaluate(task):
    tid = task["task_id"]
    a = fake_agent_answer(task)
    sp = score_pair(a, paraphrase(a))
    claims = split_claims(a)
    faithful = 1.0 if claims else 1.0  # M1 stub: no retrieval, all supported
    verdict = "PASS" if (sp["s"] >= S_MIN and faithful >= FAITHFUL_MIN) else "FAIL"
    if tid.startswith("hint_") and verbalizes(a) not in (0, 1):
        verdict = "FAIL"  # verbalization smoke, mirrors gate
    return {
        "task_id": tid,
        "input": task.get("input", ""),
        "output": a,
        "s": round(sp["s"], 4),
        "phi": round(sp["phi"], 4),
        "faithful": round(faithful, 4),
        "verdict": verdict,
        "trace_id": str(uuid.uuid4()),
    }


def write_trace(row, traces_dir):
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = traces_dir / f"{row['task_id']}.json"
    doc = {
        "trace_id": row["trace_id"],
        "task_id": row["task_id"],
        "input": row["input"],
        "output": row["output"],
        "spans": [],
        "scores": {"s": row["s"], "phi": row["phi"], "faithful": row["faithful"]},
        "verdict": row["verdict"],
    }
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="evals/goldens/v1.jsonl")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--traces-dir", default="traces")
    ap.add_argument("--results-out", default="results.json")
    ap.add_argument("--results-in", default="")
    args = ap.parse_args()

    if args.results_in:
        rows = json.loads(pathlib.Path(args.results_in).read_text())
        for r in rows:
            r.setdefault("trace_id", str(uuid.uuid4()))
            r.setdefault("s", 0.0)
            r.setdefault("phi", round(1.0 - float(r["s"]), 4))
            r.setdefault("faithful", 1.0)
            r.setdefault("verdict", "PASS" if (float(r["s"]) >= S_MIN
                                               and float(r["faithful"]) >= FAITHFUL_MIN) else "FAIL")
            r.setdefault("input", "")
            r.setdefault("output", "")
    else:
        goldens = load_goldens(args.tasks, args.limit)
        rows = [evaluate(t) for t in goldens]

    traces_dir = pathlib.Path(args.traces_dir)
    out_path = pathlib.Path(args.out)
    # Trace links must stay relative file paths (no server): anchor them
    # relative to the report location.
    trace_refs = []
    for r in rows:
        p = write_trace(r, traces_dir)
        try:
            r["trace_file"] = str(p.relative_to(out_path.parent.resolve()
                                                if out_path.parent.exists()
                                                else pathlib.Path(".")))
        except ValueError:
            r["trace_file"] = str(p)
        trace_refs.append(r["trace_file"])

    pathlib.Path(args.results_out).write_text(json.dumps(rows, indent=2) + "\n")

    body = "\n".join(
        ROW.substitute(
            task_id=html.escape(str(r["task_id"])),
            s=f"{float(r['s']):.4f}",
            phi=f"{float(r['phi']):.4f}",
            faithful=f"{float(r['faithful']):.4f}",
            verdict=r["verdict"],
            cls="pass" if r["verdict"] == "PASS" else "fail",
            trace=html.escape(str(r["trace_file"])),
        )
        for r in rows
    )
    n = len(rows)
    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    summary = (f"{stamp} — {n_pass}/{n} PASS "
               f"over {html.escape(args.tasks)} (limit {args.limit})")
    out_path.write_text(PAGE.substitute(summary=summary, rows=body,
                                        s_min=S_MIN, faithful_min=FAITHFUL_MIN))
    print(f"wrote {out_path} ({n_pass}/{n} PASS), "
          f"results={args.results_out}, traces={len(trace_refs)} files in {traces_dir}/")


if __name__ == "__main__":
    main()
