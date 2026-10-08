# CFA-Mesh — Causal Faithfulness Audit Mesh — plan.md

Status: draft | Date: 2026-10-08 | Owner: atp/Plans
Supersedes: Project Ariadne 2601.02314 (`skhanzad/AriadneXAI` v1, 8 commits, 3 stars)
Basis: Turpin 2305.04388, Lanham 2307.13702, Anthropic 2505.05410, FaithCoT-Bench 2510.04040, Boppana 2603.05488, ROME 2202.05262 + DAS, A2E 2608.07346, DeepEval, RAGAS, Langfuse/Phoenix OTel, OpenAI Monitorability Dec 2025, Agent-as-Judge 2410.10934

## 1. PRD

### 1.1 Problem
Ariadne proved CoT can be "Reasoning Theater" (`rho=0.767` on N=30, single GPT-4o, LLM-judge only, undefined `tau_sim/lambda`). No paired baseline, no white-box proof, no CI gate, CSV logger, Ollama-only. Cannot distinguish error-correction from decoupling. Not citable as benchmark, not deployable.

### 1.2 Users
1. Agent builders shipping tool-use agents (LangGraph/CrewAI/AutoGen/smolagents).
2. Safety/eval engineers gating merges on faithfulness.
3. Auditors needing trajectory evidence with span links.

### 1.3 JTBD
When I merge an agent change, I want a CI gate that fails on unfaithful reasoning with evidence, so I can ship without manual trace review.

### 1.4 Functional requirements
- F1: Capture full trajectory via OTel (CoT + tool calls + retrieval + memory), queryable.
- F2: Run 6 interventions per trace: early-answer, adding-mistakes, filler, paraphrase, logicFlip, factReversal/premiseNegation + clean-vs-hinted pairs.
- F3: 3-channel verdict per trace: behavioral (flip + verbalization), white-box (probe + clean/corrupt/restore where available), agentic judge (claim-level grounding with evidence spans).
- F4: Calibrated scores: `phi=1-S`, `rho` with CIs, AOC curves, verbalization rate, probe AUROC, RAGAS `faithful/total`, monitor `g-mean`, `Q_h` efficiency.
- F5: CI gate: `pytest` fails merge on threshold breach; playground replay of bad trace.
- F6: Matrix: benchmarks x harnesses via adapter protocol; fixed seeds + identical task IDs.

### 1.5 Non-functional
- NF1: No single LLM-judge; every judge score has control + calibration.
- NF2: Offline CPU path (HHEM) for claim NLI; no prod PII to external judges by default.
- NF3: OTLP export; CSV only as offline exporter.
- NF4: Repro: seed-pinned, threshold config versioned.

## 2. Goals / Non-goals

Goals:
- G1: Outperform Ariadne: >=30x traces, multi-model, paired controls, calibrated thresholds.
- G2: Outperform Lanham/Turpin copies: add agent trajectories + white-box + gating.
- G3: Pilot-ready in 4 phases; prod gate in CI.

Non-goals:
- NG1: No weight training / RLHF on `phi`. Reward signal only after calibration study.
- NG2: No PII redaction engine; rely on harness scrubbing + OTel sampling.
- NG3: No new foundation model; Ollama + OpenAI-compatible endpoints only.
- NG4: No real-time blocking guardrail (async audit, not inline firewall). Runtime guard via PurpleLlama/LlamaFirewall out of scope.

## 3. Architecture

```
[Harnesses] -> [Capture OTel] -> [Intervention Engine] -> [3-Channel Verdict] -> [Calibration/Scores] -> [Gate + UI]
```

### 3.1 Capture
- OTel GenAI conv: `trace_id`=task, spans `generation(span LLM)`, `retriever`, `tool`, `plan`, `memory`.
- Decorator `@observe(name, as_type)` + `start_as_current_observation`. Fields: input/output, model, tokens, latency, cost, session/user, scores. W3C `traceparent` propagation.
- `HarnessAdapter` interface: `run(task)->trace`. Adapters: LangGraph first, then CrewAI/AutoGen/smolagents/OpenAI/Claude via OpenInference auto-instrumentation.
- Stores: `trajectory / result / metric` DB (Postgres/SQLite + JSONL dump).

### 3.2 Intervention Engine
Per trace, run pairs:
- `early-answer@25/50/75%`, `adding-mistakes` (error LM), `filler (...)`, `paraphrase-prefix`, `logicFlip`, `factReversal`.
- `clean vs hinted` (Turpin/Anthropic 6 hint types: answer-suggest, endorsement, metadata, grader-hack, unauthorized-access, sycophancy). Keep only causal-use pairs `au!=h and ah==h`.
- Resample downstream `s_j*`. Persist both branches under same `trace_id`.

### 3.3 Verdict (3 channels)
- C1 Behavioral: flip_rate, `phi`, verbalization via string-match (did trace mention hint/bias?). Objective, no judge.
- C2 White-box (when weights/activations accessible): linear probe per position for final answer; clean/corrupted/restore IE heatmap (ROME) / DAS subspace. Metrics: probe AUROC, early-exit token saving @95% acc (Boppana: 68% MMLU, 33% GPQA target to beat).
- C3 Agentic judge: judge agent with tools `read_trace, inspect_docs, rerun_tool`. Steps: split response to atomic claims -> NLI vs `retrieved_contexts` (HHEM-2.1 CPU default, LLM-judge optional) -> per-claim verdict + span link. Metrics: `faithful/total`, AnswerRelevancy, ContextPrecision/Recall.

### 3.4 Calibration / Scores
- Calibrate on FaithCoT 1000+ labeled trajectories + local goldens. Thresholds versioned in `evals/thresholds.yaml` (p25 of prod, not 0.95 aspirational).
- Report: `rho` + 95% CI, `EF`, AOC, verbalization %, probe AUROC, RAGAS trio, monitor `g-mean^2`, `Q_h = 1 - sqrt(T_hat^2+(1-S_hat)^2)/sqrt2`.
- Lifecycle scores: plan quality/adherence, tool/arg correctness, memory, safety, final answer, tokens/turns/latency.

### 3.5 Gate + UI
- `deepeval test run` style: `assert_test` + `DAGMetric` fail-fast (e.g. missing citation -> 0). `threshold=None` = observational.
- Local UI `:6006`: browse datasets/experiments/trace trees, replay bad trace with new prompt.

## 4. Tech stack

- Lang: Python 3.11, asyncio.
- Tracing: OpenTelemetry SDK + OTLP exporter, OpenInference instrumentors; backend Langfuse or Phoenix (self-host). SQLite -> Postgres.
- Evals: pytest, `deepeval`-pattern `evaluate()` async, `ragas` HHEM (`Vectara HHEM-2.1-Open`), `pyvene` for DAS/probes where applicable.
- Models: Ollama (`llama3/mistral` local) via OpenAI-compatible wrapper; optional GPT-4o/Claude 3.7 Sonnet judges behind flag, off by default.
- Bench matrix: tau-bench, SWE-bench subset, Terminal-Bench-2 sample, MMLU/GPQA-Diamond/TruthfulQA/LogiQA/AQuA (5-task sample per bench to start).
- UI: lightweight FastAPI + static trace viewer (A2E `:6006` pattern).
- CI: GitHub Actions, pinned actions, `thresholds.yaml` versioned.

## 5. Workflow (system)

1. Ingest task set (fixed seed, task IDs).
2. Run harness adapter -> OTel traces.
3. Fork interventions (6 + hinted pair) -> store branches.
4. Run C1/C2/C3 verifiers in parallel.
5. Calibrate/normalize -> scores + CIs.
6. Gate: pass/fail vs `thresholds.yaml`; write `trajectory/result/metric` rows.
7. UI + playground link on fail; export JSONL.

## 6. Userflow

1. Dev opens PR.
2. CI runs `pytest tests/evals/test_faithfulness.py` (10-20 goldens + sampled bench).
3. Gate posts: `rho=0.21 [0.18-0.24], verbal=82%, probe AUROC 0.88, RAGAS 0.91, Q_h 0.74 — PASS` with trace links.
4. On FAIL: click trace -> see flipped branch, unmentioned hint span highlighted, per-claim evidence, replay with fixed prompt.
5. Merge blocked until gate passes or `risk-accept` with review date.

## 7. Security

Threat model (STRIDE-lite): untrusted prompts/tool outputs -> prompt injection exfil; poisoned retrievals -> ungrounded claims; judge leakage (PII to external LLM); supply-chain (unpinned actions, deps); forged eval pass.

Controls:
- Secrets: no keys in repo/logs; env/KMS only; `.gitleaks.toml` + CI secret scan; OTel scrub `input` for bearer/PII before export; local judges default.
- Supply chain: `requirements.txt` + lockfile tracked; `pip-audit` in CI; Dependabot; pinned GH Actions SHA; no `postinstall` in prod deps; verify `skhanzad/AriadneXAI` code before reuse (3-star prototype, no license — do not vendor blindly).
- CI/CD: no `pull_request_target` + checkout PR code; least-privilege `GITHUB_TOKEN`; OIDC for deploy; eval artifacts immutable by run ID.
- LLM: treat tool output/retrieved docs as untrusted; judge agent allowlist tools only (`read/inspect/rerun`, no net exec); cap tokens/cost per trace; rate-limit interventions; log hint-injection attempts as safety signal.
- OWASP LLM Top 10 mapped: LL01 prompt injection (paired-hint tests), LL02 data leak (scrub + local HHEM), LL03 supply (pin/audit), LL06 excessive agency (judge tool allowlist), LL08 vector poisoning (context precision/recall), LL09 overreliance (gate blocks on low verbalization).
- Data: eval datasets versioned; prod traces sampled, retention 30d; tenant isolation via `session_id`.
- Incident: on leaked key — revoke, rotate, `git filter-repo`, force-push, audit exposure window, check provider logs.

## 8. Milestones / Verification

- P0 Discovery: allowed-APIs list (OTel, OpenInference, HHEM, pyvene). Anti-pattern: inventing `do(s_k)` as real Pearl op on text.
- P1 Capture: 5 tasks x 2 harnesses queryable in UI. `grep -r CSV` only in exporter.
- P2 Interventions: AOC curves separate AQuA/LogiQA (sensitive) vs ARC/OBQA (invariant).
- P3 Verdict: AUROC vs FaithCoT labels; judge shift <1% (Agent-as-Judge target 0.27%).
- P4 Gate: CI red on injected bias/hint without verbalization; green after fix. `filter_stats` logged.
- Metrics to beat: Ariadne N=30 -> >=1000 traces; verbalization baseline 25/39% (Anthropic); early-exit 68%/33% (Boppana); judge alignment 90%+ (Agent-as-Judge).

## 9. Open questions

- White-box access? If closed models only, C2 runs on open-weight subset.
- External judges allowed? Default no; flag for calibration runs only.
- Which 5 goldens per bench for P1? Pick from tau-bench + GPQA-Diamond + TruthfulQA first.
