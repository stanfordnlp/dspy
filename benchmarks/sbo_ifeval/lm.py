"""Thin litellm wrapper for direct LM calls — no DSPy, no adapter overhead."""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)

_call_counter = 0
_counter_lock = threading.Lock()


def _next_call_id() -> int:
    global _call_counter
    with _counter_lock:
        _call_counter += 1
        return _call_counter


def _print_io(call_id: int, label: str, model: str, messages: list[dict], response: str) -> None:
    w = 70
    print(f"\n{'─' * w}")
    print(f"  LM call #{call_id}  [{label}]  model={model}")
    print(f"{'─' * w}")
    for msg in messages:
        role = msg.get("role", "?").upper()
        content = msg.get("content", "")
        print(f"[{role}]")
        print(content)
        print()
    print("[RESPONSE]")
    print(response)
    print(f"{'─' * w}")


class LMClient:
    """Direct litellm wrapper. Task calls go straight to the model — no format overhead."""

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        print_io: bool = False,
        label: str = "lm",
        **kwargs: Any,
    ):
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.print_io = print_io
        self.label = label
        self._extra = kwargs

    def _call(self, messages: list[dict[str, str]], temperature: float | None = None, **kwargs: Any) -> str:
        import litellm

        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": self.max_tokens,
            **self._extra,
            **kwargs,
        }
        if self.api_base:
            params["api_base"] = self.api_base

        response = litellm.completion(**params)
        text = response.choices[0].message.content or ""
        if self.print_io:
            _print_io(_next_call_id(), self.label, self.model, messages, text)
        return text

    def task(self, system_prompt: str, user_message: str, temperature: float | None = None) -> str:
        """Direct task call: system + user → raw text. No adapter, no delimiters."""
        return self._call(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
        )

    def json(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
        retries: int = 2,
    ) -> dict[str, Any]:
        """Call LM expecting JSON output. Strips markdown fences; retries on parse failure."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        for attempt in range(1 + retries):
            text = self._call(messages, temperature=temperature)
            stripped = text.strip()
            # Strip ```json ... ``` fences
            if stripped.startswith("```"):
                lines = stripped.splitlines()
                stripped = "\n".join(lines[1:]).rstrip("`").strip()
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", stripped, re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group())
                    except json.JSONDecodeError:
                        pass
            if attempt < retries:
                logger.warning(
                    "JSON parse failed (attempt %d/%d), retrying. response: %r",
                    attempt + 1, 1 + retries, text[:200],
                )
            else:
                logger.warning("JSON parse failed after %d attempts, response: %r", 1 + retries, text[:300])
        return {}
