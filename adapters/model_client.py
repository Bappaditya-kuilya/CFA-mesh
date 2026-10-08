"""Minimal model client: Ollama local or Groq/OpenAI-compatible via API key. Stdlib only."""
import json
import os
import urllib.request

SMALL_MODELS = {
    "ollama/qwen2.5-coder:1.5b": {"kind": "ollama", "ctx": 32768},
    "ollama/qwen2.5-coder:3b": {"kind": "ollama", "ctx": 32768},
    "ollama/gemma2:2b": {"kind": "ollama", "ctx": 8192},
    "groq/qwen-2.5-coder-32b": {"kind": "groq", "ctx": 32768},
    "groq/gemma2-9b-it": {"kind": "groq", "ctx": 8192},
}

PROVIDERS = {
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key_env": ""},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"},
}


def resolve(model: str, provider_kind: str = "", base_url: str = "", api_key_env: str = "") -> dict:
    """Resolve model -> {kind, base_url, api_key_present}. Never returns the key itself."""
    info = SMALL_MODELS.get(model, {"kind": provider_kind or "ollama"})
    kind = info["kind"]
    defaults = PROVIDERS.get(kind, PROVIDERS["ollama"])
    url = base_url or defaults["base_url"]
    key_env = api_key_env if api_key_env != "" else defaults["api_key_env"]
    key = os.environ.get(key_env, "") if key_env else ""
    if kind in ("groq",) and not key:
        raise RuntimeError(f"missing API key: set {key_env} for {model}")
    return {"model": model, "kind": kind, "base_url": url, "api_key_present": bool(key)}


def complete(model: str, prompt: str, provider_kind: str = "", base_url: str = "",
             api_key_env: str = "", temperature: float = 0.0, seed: int = 7,
             max_tokens: int = 512) -> str:
    """POST chat completion. Key sent only via Authorization header, never logged."""
    cfg = resolve(model, provider_kind, base_url, api_key_env)
    key_env = api_key_env or PROVIDERS[cfg["kind"]]["api_key_env"]
    key = os.environ.get(key_env, "") if key_env else ""
    short = model.split("/", 1)[1] if "/" in model else model
    body = json.dumps({
        "model": short,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {key}"} if key else {})},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"]
