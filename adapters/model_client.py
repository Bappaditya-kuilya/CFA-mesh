"""Minimal model client: Ollama local or Groq/OpenAI-compatible via API key. Stdlib only."""
import json
import os
import time
import urllib.error
import urllib.request

SMALL_MODELS = {
    "ollama/qwen2.5-coder:1.5b": {"kind": "ollama", "ctx": 32768},
    "ollama/qwen2.5-coder:3b": {"kind": "ollama", "ctx": 32768},
    "ollama/gemma2:2b": {"kind": "ollama", "ctx": 8192},
    "groq/qwen-qwq-32b": {"kind": "groq", "ctx": 32768},
    "groq/llama-3.1-8b-instant": {"kind": "groq", "ctx": 8192},
    "google/gemini-3.5-flash-lite": {"kind": "google", "ctx": 32768},
    "google/gemini-2.5-flash": {"kind": "google", "ctx": 32768},
    "groq/gemma2-9b-it": {"kind": "groq", "ctx": 8192},
}

PROVIDERS = {
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key_env": ""},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"},
    "google": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "api_key_env": "GOOGLE_API_KEY"},
}


def load_dotenv(path: str = "") -> None:
    """Parse KEY=VALUE lines from a .env file (stdlib only).

    Skips blank lines and comments. Never overrides existing env vars.
    Missing file is a silent no-op.
    """
    if not path:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    try:
        with open(path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                    val = val[1:-1]
                if not key or key in os.environ:
                    continue
                os.environ[key] = val
    except OSError:
        pass


def resolve(model: str, provider_kind: str = "", base_url: str = "", api_key_env: str = "") -> dict:
    """Resolve model -> {kind, base_url, api_key_present}. Never returns the key itself."""
    load_dotenv()
    if model in SMALL_MODELS:
        kind = SMALL_MODELS[model]["kind"]
    elif provider_kind:
        kind = provider_kind
        if kind not in PROVIDERS:
            raise RuntimeError(
                f"unknown provider_kind {kind!r} for {model!r}; valid kinds: {sorted(PROVIDERS)}")
    elif "/" in model:
        prefix = model.split("/")[0]
        if prefix in PROVIDERS:
            kind = prefix
        else:
            raise RuntimeError(
                f"unknown model prefix {prefix!r} in {model!r}; valid kinds: {sorted(PROVIDERS)}")
    else:
        kind = "ollama"
    defaults = PROVIDERS.get(kind, PROVIDERS["ollama"])
    url = base_url or defaults["base_url"]
    key_env = api_key_env if api_key_env != "" else defaults["api_key_env"]
    key = os.environ.get(key_env, "") if key_env else ""
    if kind in ("groq", "google") and not key:
        raise RuntimeError(f"missing API key: set {key_env} for {model}")
    return {"model": model, "kind": kind, "base_url": url, "api_key_present": bool(key)}


def complete_with_usage(model: str, prompt: str, provider_kind: str = "", base_url: str = "",
                        api_key_env: str = "", temperature: float = 0.0, seed: int = 7,
                        max_tokens: int = 512, timeout: int = 120) -> dict:
    """POST chat completion. Key sent only via Authorization header, never logged.

    Returns {text, tokens_in, tokens_out, elapsed_s}; token counts come from
    the usage payload when present, else zeros.
    """
    cfg = resolve(model, provider_kind, base_url, api_key_env)
    key_env = api_key_env or PROVIDERS[cfg["kind"]]["api_key_env"]
    key = os.environ.get(key_env, "") if key_env else ""
    short = model.split("/", 1)[1] if "/" in model else model
    payload = {
        "model": short,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if cfg["kind"] in ("ollama", "groq"):
        payload["seed"] = seed  # google OpenAI-compat rejects seed; record it in experiments instead
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {key}"} if key else {})},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        try:
            snippet = e.read().decode("utf-8", "replace")[:500]
        except Exception:
            snippet = ""
        raise RuntimeError(
            f"chat completion failed for {model}: HTTP {e.code} {e.reason}; body: {snippet}") from e
    elapsed = time.monotonic() - t0
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"malformed chat-completion response for {model}: "
            f"missing choices[0].message.content ({e})") from e
    usage = data.get("usage") or {}
    try:
        tokens_in = int(usage.get("prompt_tokens", 0) or 0)
    except (TypeError, ValueError):
        tokens_in = 0
    try:
        tokens_out = int(usage.get("completion_tokens", 0) or 0)
    except (TypeError, ValueError):
        tokens_out = 0
    return {"text": text, "tokens_in": tokens_in,
            "tokens_out": tokens_out, "elapsed_s": elapsed}


def complete(model: str, prompt: str, provider_kind: str = "", base_url: str = "",
             api_key_env: str = "", temperature: float = 0.0, seed: int = 7,
             max_tokens: int = 512, timeout: int = 120) -> str:
    """POST chat completion. Back-compat thin wrapper returning text only."""
    return complete_with_usage(model, prompt, provider_kind, base_url,
                               api_key_env, temperature, seed,
                               max_tokens, timeout)["text"]
