"""OpenAI harness stub (M5 second harness).

PRD refs: section 3 (``HarnessAdapter.run(task, env) ->
(trace_id, harness_version)``), F2 cache key includes ``harness_version``,
F7 ``experiments.harness_version``.

Status: STUB. The ``openai`` package is NOT installed in this repo; all
model calls go through :mod:`adapters.model_client` (stdlib urllib,
OpenAI-compatible ``/chat/completions``), which already talks to Ollama
local or Groq without new dependencies. This stub exposes the identical
``HarnessAdapter`` interface as :mod:`adapters.langgraph_adapter` so the
orchestrator can treat both harnesses uniformly.

To wire the real SDK later::

    pip install openai
    # Replace the raise with:
    #   from openai import OpenAI
    #   client = OpenAI(base_url=..., api_key=...)
    #   resp = client.chat.completions.create(model=..., messages=...)

Until then :meth:`HarnessAdapter.run` resolves via ``model_client`` and
raises ``NotImplementedError`` documenting the missing optional dependency.
"""

import uuid

from adapters import model_client

HARNESS_VERSION = "openai-stub-v0"
HARNESS_NAME = "openai"


def _prompt_for(task: dict) -> str:
    return str(task.get("input", task.get("prompt", "")))


class HarnessAdapter:
    """Faithful OpenAI adapter interface (stub)."""

    name: str = HARNESS_NAME
    harness_version: str = HARNESS_VERSION

    def run(self, task: dict, env: dict) -> tuple[str, str]:
        """Run one task via the OpenAI harness.

        Args:
            task: mapping with at least ``task_id`` and ``input`` (or ``prompt``).
            env: mapping with ``model`` plus optional provider overrides
                (``provider_kind``, ``base_url``, ``api_key_env``).

        Returns:
            ``(trace_id, harness_version)`` per PRD section 3.

        Raises:
            NotImplementedError: always for now, because the ``openai`` SDK
                is not installed. Use ``adapters.model_client.complete``
                directly for real calls until the SDK is wired.
        """
        model = str(env.get("model", "ollama/qwen2.5-coder:1.5b"))
        # Validate model/provider wiring with the real client (no network call).
        model_client.resolve(
            model,
            provider_kind=str(env.get("provider_kind", "")),
            base_url=str(env.get("base_url", "")),
            api_key_env=str(env.get("api_key_env", "")),
        )
        _prompt_for(task)  # prompt construction is real; SDK call is stubbed
        trace_id = str(uuid.uuid4())
        try:
            import openai  # type: ignore # noqa: F401
        except ImportError as e:
            raise NotImplementedError(
                "openai harness not wired: pip install openai, or call "
                "adapters.model_client.complete(model, prompt, ...) directly "
                "(stdlib, no new deps). "
                f"task_id={task.get('task_id', '?')}"
            ) from e
        # Real implementation would call the SDK and return its trace_id.
        return (trace_id, HARNESS_VERSION)
