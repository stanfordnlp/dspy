"""Trackio callback for DSPy.

Trackio (https://github.com/gradio-app/trackio) is a lightweight,
local-first experiment tracker with a wandb-compatible API and a
first-class Trace primitive for conversational data.

This module provides ``TrackioCallback``, a ``BaseCallback`` subclass
that emits one ``trackio.Trace`` per LM invocation. Drop it into
``dspy.configure(callbacks=[...])`` and every ``Predict`` /
``ChainOfThought`` / ``ReAct`` call shows up on the Trackio dashboard
with its real prompt messages and the model's response.
"""

from __future__ import annotations

import logging
from typing import Any

from dspy.utils.callback import BaseCallback


logger = logging.getLogger(__name__)


def _coerce_messages(prompt: str | None, messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if messages:
        return [dict(m) for m in messages]
    if prompt:
        return [{"role": "user", "content": str(prompt)}]
    return []


def _completion_to_str(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("text") or completion.get("content") or completion)
    return str(completion)


class TrackioCallback(BaseCallback):
    """Logs each DSPy LM call as a ``trackio.Trace``.

    Args:
        project: Trackio project name. Defaults to ``"dspy"``.
        name: Optional Trackio run name. Forwarded to ``trackio.init``.
        space_id: Optional Hugging Face Space id (``"username/space"``)
            to host the dashboard. If unset, runs are kept locally.
        log_key: Top-level key under which traces are logged. Defaults to
            ``"lm_call"``.
        init_kwargs: Additional kwargs forwarded to ``trackio.init``.
        auto_init: If ``True`` (default), call ``trackio.init`` if no run
            is currently active. Set to ``False`` if you initialize Trackio
            yourself.

    Example:
        >>> import dspy
        >>> from dspy.utils.trackio import TrackioCallback
        >>> dspy.configure(callbacks=[TrackioCallback(project="my-app")])
        >>> cot = dspy.ChainOfThought("question -> answer")
        >>> cot(question="What is 2 + 2?")
    """

    def __init__(
        self,
        project: str = "dspy",
        name: str | None = None,
        space_id: str | None = None,
        log_key: str = "lm_call",
        init_kwargs: dict[str, Any] | None = None,
        auto_init: bool = True,
    ):
        try:
            import trackio
        except ImportError as e:
            raise ImportError(
                "trackio is not installed. Install it with `pip install trackio`."
            ) from e

        self._trackio = trackio
        self._project = project
        self._name = name
        self._space_id = space_id
        self._log_key = log_key
        self._init_kwargs = dict(init_kwargs or {})
        self._auto_init = auto_init

        self._initialized = False
        self._step = 0
        self._call_inputs: dict[str, dict[str, Any]] = {}

    def _ensure_init(self) -> None:
        if self._initialized or not self._auto_init:
            self._initialized = True
            return
        kwargs: dict[str, Any] = {"project": self._project, **self._init_kwargs}
        if self._name is not None:
            kwargs.setdefault("name", self._name)
        if self._space_id is not None:
            kwargs.setdefault("space_id", self._space_id)
        try:
            self._trackio.init(**kwargs)
        except Exception as e:
            logger.warning(f"trackio.init failed; subsequent logs will be no-ops: {e}")
        self._initialized = True

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        self._call_inputs[call_id] = inputs

    def on_lm_end(
        self,
        call_id: str,
        outputs: list[dict[str, Any] | str] | None,
        exception: Exception | None = None,
    ):
        inputs = self._call_inputs.pop(call_id, {})
        if exception is not None:
            return

        prompt = inputs.get("prompt") if isinstance(inputs, dict) else None
        messages = inputs.get("messages") if isinstance(inputs, dict) else None
        trace_messages = _coerce_messages(prompt, messages)

        completion_text = ""
        if outputs:
            first = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            completion_text = _completion_to_str(first)
        if completion_text:
            trace_messages.append({"role": "assistant", "content": completion_text})

        metadata: dict[str, Any] = {"call_id": call_id}
        if isinstance(inputs, dict):
            for k in ("model", "model_type", "temperature", "max_tokens"):
                if k in inputs:
                    metadata[k] = inputs[k]
        if outputs and isinstance(outputs, (list, tuple)) and len(outputs) > 1:
            metadata["num_completions"] = len(outputs)

        try:
            self._ensure_init()
            trace = self._trackio.Trace(messages=trace_messages, metadata=metadata)
            self._trackio.log({self._log_key: trace}, step=self._step)
            self._step += 1
        except Exception as e:
            logger.warning(f"Failed to log Trackio trace for LM call {call_id}: {e}")
