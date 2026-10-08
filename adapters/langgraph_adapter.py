"""LangGraph harness stub (M5 second harness).

PRD refs: section 3 (``cli -> orchestrator -> adapters -> capture``),
  ``HarnessAdapter.run(task, env) -> (trace_id, harness_version)``,
  F2 cache key includes ``harness_version``, F7 ``experiments.harness_version``.

Status: STUB. The ``langgraph`` package is NOT installed in this repo and
must not be added as a dependency for M1-M4. This file defines the faithful
interface now so the orchestrator, store, and cache key can use
``(trace_id, harness_version)`` without importing langgraph.

To wire the real harness later::

    pip install langgraph
    # Replace _run_graph() below with:
    #   from langgraph.graph import StateGraph
    #   graph = build_graph(env)   # nodes call model_client.complete(...)
    #   state = graph.invoke({"input": task["input"]})

Until then :meth:`HarnessAdapter.run` resolves the model via
:mod:`adapters.model_client` (proving the wiring), then raises
``NotImplementedError`` documenting the missing optional dependency.
"""

import uuid

from adapters import model_client

HARNESS_VERSION = "langgraph-stub-v0"
HARNESS_NAME = "langgraph"


def _prompt_for(task: dict) -> str:
    return str(task.get("input", task.get("prompt", "")))


class HarnessAdapter:
    """Faithful LangGraph adapter interface (stub)."""

    name: str = HARNESS_NAME
    harness_version: str = HARNESS_VERSION

    def run(self, task: dict, env: dict) -> tuple[str, str]:
        """Run one task via the LangGraph harness.

        Args:
            task: mapping with at least ``task_id`` and ``input`` (or ``prompt``).
            env: mapping with ``model`` plus optional provider overrides
                (``provider_kind``, ``base_url``, ``api_key_env``).

        Returns:
            ``(trace_id, harness_version)`` per PRD section 3.

        Raises:
            NotImplementedError: always for now, because ``langgraph`` is not
                installed. The model is still resolved via ``model_client``
                first so cache-key inputs are validated early.
        """
        model = str(env.get("model", "ollama/qwen2.5-coder:1.5b"))
        # Validate model/provider wiring with the real client (no network call).
        model_client.resolve(
            model,
            provider_kind=str(env.get("provider_kind", "")),
            base_url=str(env.get("base_url", "")),
            api_key_env=str(env.get("api_key_env", "")),
        )
        _prompt_for(task)  # prompt construction is real; graph invoke is stubbed
        trace_id = str(uuid.uuid4())
        try:
            import langgraph  # type: ignore # noqa: F401
        except ImportError as e:
            raise NotImplementedError(
                "langgraph harness not wired: pip install langgraph, then "
                "implement _run_graph() with StateGraph nodes calling "
                "adapters.model_client.complete(model, prompt, ...). "
                f"task_id={task.get('task_id', '?')}"
            ) from e
        # Real implementation would invoke the graph and return its trace_id.
        return (trace_id, HARNESS_VERSION)
