# CFA-mesh

Agents write long reasoning traces that look right while the answer comes from somewhere else. CFA-mesh catches that. It saves the full trace, flips one step, reruns it, and checks if the answer moves. If the answer stays put after the logic changed, the trace was decoration — the gate fails and points at the exact span.

M1 is small on purpose: pytest over 20 goldens, two perturbations (`logic_flip` + `paraphrase`), cosine check, claim-to-evidence links. Runs on `qwen2.5-coder:1.5b` or `gemma2:2b` locally, or the same small models through Groq with a key. No big models, no remote calls by default.

Use it as a merge gate: green means the reasoning held up, red means fix the prompt or the tool wiring and rerun.
