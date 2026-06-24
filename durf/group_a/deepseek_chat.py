"""Small DeepSeek chat client used by the pygame human-AI UI."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"


class DeepSeekChatError(RuntimeError):
    """Raised when the chat service is not configured or returns an error."""


def chat_once(messages: list[dict[str, str]], timeout: float = 30.0) -> str:
    """Send a chat completion request and return the assistant text."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise DeepSeekChatError(
            "DEEPSEEK_API_KEY is not set. Set it before using Chat."
        )

    payload = {
        "model": os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
        "messages": messages,
        "temperature": 0.2,
        "stream": False,
    }
    request = urllib.request.Request(
        os.getenv("DEEPSEEK_API_URL", DEFAULT_API_URL),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise DeepSeekChatError(f"DeepSeek HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise DeepSeekChatError(f"DeepSeek request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise DeepSeekChatError("DeepSeek request timed out.") from exc

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise DeepSeekChatError(f"Unexpected DeepSeek response: {data!r}") from exc
