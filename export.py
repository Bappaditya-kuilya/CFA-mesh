"""M4 self-contained export. Stdlib only.

PRD F7: evidence_bank is INSERT-only; FAIL evidence is materialized into
the export at gate time (self-contained markdown/JSONL). UI links are
convenience only, so the export embeds claim texts + evidence span
contents, never bare ids. AC3: replay from export alone.
"""

import datetime
import json
import pathlib


def _span_text(span):
    if isinstance(span, str):
        return span
    for key in ("output", "content", "result_json", "input"):
        if span.get(key):
            return str(span[key])
    return json.dumps(span, sort_keys=True)


def materialize_evidence(verdict, spans=None, evidence_bank=None):
    """Inline FAIL evidence: claims + resolved span contents.

    verdict: {claims?[{text,verdict,evidence_span_ids?}], ...}.
    spans: {span_id: span-dict-or-text} or [span...] with span_id keys.
    evidence_bank: [{evidence_id,task_id,kind,content_hash,content?}].
    Returns list of {text, verdict, evidence[{span_id, content}]}.
    """
    span_map = {}
    if isinstance(spans, dict):
        span_map = dict(spans)
    elif isinstance(spans, list):
        for s in spans:
            if isinstance(s, dict) and s.get("span_id"):
                span_map[s["span_id"]] = s
    bank_map = {}
    for row in (evidence_bank or []):
        key = row.get("evidence_id") or row.get("content_hash")
        if key:
            bank_map[key] = row.get("content", "")
    out = []
    for claim in (verdict.get("claims") or []):
        ev = []
        for sid in (claim.get("evidence_span_ids") or []):
            if sid in span_map:
                content = _span_text(span_map[sid])
            elif sid in bank_map:
                content = bank_map[sid]
            else:
                content = "<missing: id without materialized span>"
            ev.append({"span_id": sid, "content": content})
        out.append({"text": claim.get("text", ""),
                    "verdict": claim.get("verdict", "UNVERIFIABLE"),
                    "evidence": ev})
    return out


def build_export(*, gate, agg, verdicts, versions, spans=None,
                 evidence_bank=None):
    """Build {markdown, jsonl_rows}. FAIL rows carry materialized evidence."""
    span_map = spans if isinstance(spans, dict) else None
    md = []
    badge = gate.get("mode", "observational")
    md.append(f"# CFA export: {gate.get('decision', '?')} `mode:{badge}`")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC")
    md.append(f"{stamp} — n={agg.get('n', len(verdicts))} "
              f"faithful={_f(agg.get('faithful_mean'))} "
              f"verbal={_f(agg.get('verbal_rate'))} "
              f"phi p50={_f(agg.get('phi_median'))}")
    md.append("")
    md.append("## gate reasons")
    for r in gate.get("reasons", []):
        md.append(f"- {r}")
    md.append("")
    md.append("## versions")
    md.append(f"git_sha={versions.get('git_sha', '?')} "
              f"thresholds={versions.get('thresholds_version', '?')} "
              f"dataset={versions.get('dataset', '?')} "
              f"harness={versions.get('harness_version', '?')} "
              f"seed={versions.get('seed', '?')} "
              f"emb={versions.get('emb_version', '?')} "
              f"nli={versions.get('nli_version', '?')}")
    md.append("")

    rows = []
    for v in verdicts:
        failed = v.get("verdict") in ("FAIL", "CONTRADICTED") \
            or "CONTRADICTED" in (v.get("claim_verdicts") or [])
        evidence = materialize_evidence(v, spans=spans,
                                        evidence_bank=evidence_bank) \
            if failed else []
        rows.append({**v, "fail_evidence": evidence,
                     "gate_decision": gate.get("decision"),
                     "mode": badge, "versions": versions})
        md.append(f"### {v.get('task_id', '?')} — "
                  f"{v.get('verdict', '?')} "
                  f"(s={_f(v.get('s'))} faithful={_f(v.get('faithful', v.get('faithful_ratio')))})")
        if failed:
            if not evidence:
                md.append("- FAIL with no claims recorded (stub verdict).")
            for cl in evidence:
                md.append(f"- claim [{cl['verdict']}]: {cl['text']}")
                for e in cl["evidence"]:
                    md.append(f"  - evidence {e['span_id']}: {e['content']}")
        md.append("")
    markdown = "\n".join(md)
    jsonl = "\n".join(json.dumps(r, sort_keys=True) for r in rows)
    if jsonl:
        jsonl += "\n"
    return {"markdown": markdown, "jsonl": jsonl, "rows": rows}


def write_export(base_path, bundle, *, spans=None, evidence_bank=None):
    """Write <base>.md + <base>.jsonl. bundle holds gate/agg/verdicts/versions."""
    out = build_export(gate=bundle.get("gate", {}), agg=bundle.get("agg", {}),
                       verdicts=bundle.get("verdicts", []),
                       versions=bundle.get("versions", {}),
                       spans=spans, evidence_bank=evidence_bank)
    base = pathlib.Path(base_path)
    md_path = base.with_suffix(".md")
    jsonl_path = pathlib.Path(str(base) + ".jsonl")
    md_path.write_text(out["markdown"])
    jsonl_path.write_text(out["jsonl"])
    return {"md": str(md_path), "jsonl": str(jsonl_path),
            "fail_count": sum(1 for r in out["rows"] if r["fail_evidence"])}


def _f(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"
