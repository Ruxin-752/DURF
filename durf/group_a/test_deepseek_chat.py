"""The DeepSeek client is about to sit inside a timed human session and to be
the thing that decides whether a label came from the LLM at all.  These tests
pin the behaviours that make that acceptable: retries only where retrying
helps, a cache that makes a run reproducible, and provenance on every answer.
No test touches the network."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from durf.group_a import deepseek_chat as dc

MESSAGES = [{"role": "user", "content": "hello"}]


def _ok(model="deepseek-chat-2026-01-01", fp="fp_abc", text="hi"):
    return {
        "model": model,
        "system_fingerprint": fp,
        "choices": [{"message": {"content": text}}],
    }


def _http_error(code):
    return urllib.error.HTTPError("u", code, "msg", {}, None)


class _Env:
    def __init__(self, **extra):
        self.extra = extra

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        env = {
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_CACHE_DIR": self.tmp.name,
            "DEEPSEEK_MODEL": "deepseek-chat",
            **self.extra,
        }
        self.patcher = mock.patch.dict(os.environ, env, clear=False)
        self.patcher.start()
        return Path(self.tmp.name)

    def __exit__(self, *exc):
        self.patcher.stop()
        self.tmp.cleanup()


class RetryTest(unittest.TestCase):
    def test_retries_on_5xx_then_succeeds(self) -> None:
        with _Env():
            with mock.patch.object(dc, "urllib") as fake_urllib:
                fake_urllib.request.Request = mock.MagicMock()
                fake_urllib.error = urllib.error
                responses = [_http_error(503), _http_error(502), _ok()]

                def urlopen(request, timeout):
                    item = responses.pop(0)
                    if isinstance(item, Exception):
                        item.read = lambda: b"busy"
                        raise item
                    handle = mock.MagicMock()
                    handle.__enter__.return_value.read.return_value = json.dumps(item).encode()
                    return handle

                fake_urllib.request.urlopen = urlopen
                sleeps = []
                result = dc.chat_once_detailed(
                    MESSAGES, use_cache=False, _sleep=sleeps.append
                )
        self.assertEqual(result.text, "hi")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(len(sleeps), 2)
        self.assertLess(sleeps[0], sleeps[1])  # exponential backoff

    def test_does_not_retry_a_bad_request(self) -> None:
        """A 400/401 will not fix itself; retrying it only burns the
        participant's clock."""
        with _Env():
            with mock.patch.object(dc, "urllib") as fake_urllib:
                fake_urllib.request.Request = mock.MagicMock()
                fake_urllib.error = urllib.error
                calls = []

                def urlopen(request, timeout):
                    calls.append(1)
                    err = _http_error(401)
                    err.read = lambda: b"bad key"
                    raise err

                fake_urllib.request.urlopen = urlopen
                with self.assertRaises(dc.DeepSeekChatError):
                    dc.chat_once_detailed(MESSAGES, use_cache=False, _sleep=lambda s: None)
        self.assertEqual(len(calls), 1)

    def test_gives_up_after_max_attempts(self) -> None:
        with _Env():
            with mock.patch.object(dc, "urllib") as fake_urllib:
                fake_urllib.request.Request = mock.MagicMock()
                fake_urllib.error = urllib.error
                calls = []

                def urlopen(request, timeout):
                    calls.append(1)
                    raise urllib.error.URLError("down")

                fake_urllib.request.urlopen = urlopen
                with self.assertRaises(dc.DeepSeekChatError):
                    dc.chat_once_detailed(
                        MESSAGES, use_cache=False, max_attempts=3, _sleep=lambda s: None
                    )
        self.assertEqual(len(calls), 3)


class CacheAndProvenanceTest(unittest.TestCase):
    def _serve(self, fake_urllib, payload, counter):
        fake_urllib.request.Request = mock.MagicMock()
        fake_urllib.error = urllib.error

        def urlopen(request, timeout):
            counter.append(1)
            handle = mock.MagicMock()
            handle.__enter__.return_value.read.return_value = json.dumps(payload).encode()
            return handle

        fake_urllib.request.urlopen = urlopen

    def test_second_identical_call_is_served_from_cache(self) -> None:
        """Reproducibility: the same prompt must give the same answer without
        asking the server again."""
        with _Env() as cache_dir:
            with mock.patch.object(dc, "urllib") as fake_urllib:
                calls = []
                self._serve(fake_urllib, _ok(text="first"), calls)
                first = dc.chat_once_detailed(MESSAGES, temperature=0.0)
                second = dc.chat_once_detailed(MESSAGES, temperature=0.0)
            self.assertEqual(len(calls), 1)
            self.assertFalse(first.cached)
            self.assertTrue(second.cached)
            self.assertEqual(first.text, second.text)
            self.assertEqual(first.prompt_hash, second.prompt_hash)
            self.assertTrue(list(cache_dir.glob("*.json")))

    def test_cache_key_separates_temperature_and_messages(self) -> None:
        self.assertNotEqual(
            dc.prompt_hash("m", 0.0, MESSAGES), dc.prompt_hash("m", 0.2, MESSAGES)
        )
        self.assertNotEqual(
            dc.prompt_hash("m", 0.0, MESSAGES),
            dc.prompt_hash("m", 0.0, [{"role": "user", "content": "other"}]),
        )

    def test_records_what_model_actually_answered(self) -> None:
        """The alias is floating; the server's own model string is the only
        thing that tells two sessions apart later."""
        with _Env():
            with mock.patch.object(dc, "urllib") as fake_urllib:
                self._serve(fake_urllib, _ok(model="deepseek-chat-2026-03-01", fp="fp_9"), [])
                result = dc.chat_once_detailed(MESSAGES, use_cache=False)
        self.assertEqual(result.model_requested, "deepseek-chat")
        self.assertEqual(result.model_returned, "deepseek-chat-2026-03-01")
        self.assertEqual(result.system_fingerprint, "fp_9")

    def test_missing_api_key_is_a_clear_error_not_a_retry_storm(self) -> None:
        with _Env(DEEPSEEK_API_KEY=""):
            with self.assertRaises(dc.DeepSeekChatError):
                dc.chat_once_detailed(MESSAGES, use_cache=False)

    def test_text_only_wrapper_still_works_for_the_ui(self) -> None:
        with _Env():
            with mock.patch.object(dc, "urllib") as fake_urllib:
                self._serve(fake_urllib, _ok(text="reply"), [])
                self.assertEqual(dc.chat_once(MESSAGES), "reply")


if __name__ == "__main__":
    unittest.main()
