"""DeepSeek chat client, shared by the pygame human-AI UI and the attributor.

Hardened 2026-09-05 for use inside a timed human session (the attributor now
runs LIVE, see work_plan_v1.md item 5):

* **Retry with exponential backoff** on transport errors, timeouts, HTTP 429
  and 5xx.  Other 4xx (bad key, bad request) fail immediately -- retrying
  those wastes the participant's clock.
* **Response cache** keyed by a hash of (model, temperature, messages).  The
  same prompt returns the same answer without a network round-trip, which is
  what makes an attribution run reproducible after the fact.  Directory from
  ``DEEPSEEK_CACHE_DIR`` (default ``outputs/llm_cache``); ``DEEPSEEK_CACHE=0``
  disables it.
* **Model provenance.**  DeepSeek exposes floating aliases (``deepseek-chat``)
  rather than dated snapshots, so the model cannot be pinned by name.  What
  we CAN do is record what the server says it served -- the ``model`` field
  and ``system_fingerprint`` of every response -- so a later reader can tell
  whether two sessions were answered by the same thing.  ``chat_once_detailed``
  returns that; ``chat_once`` keeps the old text-only signature for the UI.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.2          # the chat UI's historical setting
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 1.5
RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}


class DeepSeekChatError(RuntimeError):
    """Raised when the chat service is not configured or returns an error."""


@dataclass
class ChatResult:
    text: str
    model_requested: str
    model_returned: str | None
    system_fingerprint: str | None
    prompt_hash: str
    temperature: float
    cached: bool
    attempts: int


def prompt_hash(model: str, temperature: float, messages: list[dict[str, str]]) -> str:
    payload = json.dumps(
        {"model": model, "temperature": temperature, "messages": messages},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_dir() -> Path | None:
    if os.getenv("DEEPSEEK_CACHE", "1") in ("0", "false", "False", ""):
        return None
    return Path(os.getenv("DEEPSEEK_CACHE_DIR", "outputs/llm_cache"))


def _post_once(request: urllib.request.Request, timeout: float) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        error = DeepSeekChatError(f"DeepSeek HTTP {exc.code}: {detail}")
        error.retryable = exc.code in RETRYABLE_HTTP  # type: ignore[attr-defined]
        raise error from exc
    except urllib.error.URLError as exc:
        error = DeepSeekChatError(f"DeepSeek request failed: {exc.reason}")
        error.retryable = True  # type: ignore[attr-defined]
        raise error from exc
    except TimeoutError as exc:
        error = DeepSeekChatError("DeepSeek request timed out.")
        error.retryable = True  # type: ignore[attr-defined]
        raise error from exc


def chat_once_detailed(
    messages: list[dict[str, str]],
    timeout: float = 30.0,
    *,
    temperature: float | None = None,
    max_attempts: int | None = None,
    max_tokens: int | None = None,
    use_cache: bool = True,
    _sleep=time.sleep,
) -> ChatResult:
    """Send a chat completion and return the text plus its provenance."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
    if temperature is None:
        temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", DEFAULT_TEMPERATURE))
    attempts_allowed = max_attempts or int(
        os.getenv("DEEPSEEK_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS)
    )
    digest = prompt_hash(model, temperature, messages)

    cache_dir = _cache_dir() if use_cache else None
    cache_path = cache_dir / f"{digest}.json" if cache_dir else None
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return ChatResult(
            text=cached["text"],
            model_requested=model,
            model_returned=cached.get("model_returned"),
            system_fingerprint=cached.get("system_fingerprint"),
            prompt_hash=digest,
            temperature=temperature,
            cached=True,
            attempts=0,
        )

    if not api_key:
        raise DeepSeekChatError(
            "DEEPSEEK_API_KEY is not set. Set it before using Chat."
        )

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    request = urllib.request.Request(
        os.getenv("DEEPSEEK_API_URL", DEFAULT_API_URL),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error: DeepSeekChatError | None = None
    data: dict | None = None
    attempts = 0
    for attempt in range(1, attempts_allowed + 1):
        attempts = attempt
        try:
            data = _post_once(request, timeout)
            break
        except DeepSeekChatError as exc:
            last_error = exc
            if not getattr(exc, "retryable", False) or attempt == attempts_allowed:
                raise
            _sleep(DEFAULT_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    assert data is not None or last_error is not None
    if data is None:
        raise last_error  # type: ignore[misc]

    try:
        text = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise DeepSeekChatError(f"Unexpected DeepSeek response: {data!r}") from exc

    result = ChatResult(
        text=text,
        model_requested=model,
        model_returned=data.get("model"),
        system_fingerprint=data.get("system_fingerprint"),
        prompt_hash=digest,
        temperature=temperature,
        cached=False,
        attempts=attempts,
    )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "text": result.text,
                    "model_requested": result.model_requested,
                    "model_returned": result.model_returned,
                    "system_fingerprint": result.system_fingerprint,
                    "temperature": temperature,
                    "messages": messages,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return result


def chat_once(messages: list[dict[str, str]], timeout: float = 30.0) -> str:
    """Send a chat completion request and return the assistant text.

    Text-only wrapper kept for the pygame chat UI.  The attributor uses
    ``chat_once_detailed`` so it can record what model actually answered.
    """
    return chat_once_detailed(messages, timeout).text
