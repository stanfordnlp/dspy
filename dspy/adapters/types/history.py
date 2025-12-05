import warnings
from typing import Any, Literal

import pydantic


class History(pydantic.BaseModel):
    """Class representing conversation history for DSPy modules.

    History allows you to pass previous conversation turns or context to a module.
    Use factory methods to create History objects - DSPy will handle formatting automatically.

    **Chat-style history** (LM messages):
        ```python
        history = dspy.History.from_raw([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ])
        ```

    **Signature-matched history** (previous input/output pairs):
        ```python
        history = dspy.History.from_signature_pairs([
            {"question": "What is 2+2?", "answer": "4"},
        ])
        ```

    **Few-shot demonstrations**:
        ```python
        history = dspy.History.from_demos([
            {"input_fields": {"question": "2+2?"}, "output_fields": {"answer": "4"}},
        ])
        ```

    **Arbitrary context** (key-value pairs as user messages):
        ```python
        history = dspy.History.from_kv([
            {"thought": "I need to search", "tool": "search", "result": "Found it"},
        ])
        ```

    You can also pass `History(messages=[...])` directly - DSPy will infer the format
    from the message structure when possible.

    Example:
        ```python
        import dspy

        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        class QA(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        predict = dspy.Predict(QA)

        # First turn
        result = predict(question="What is the capital of France?")

        # Build history from previous turn
        history = dspy.History.from_signature_pairs([
            {"question": "What is the capital of France?", **result}
        ])

        # Follow-up with context
        result = predict(question="What about Germany?", history=history)
        ```
    """

    messages: list[dict[str, Any]]
    mode: Literal["signature", "demo", "flat", "raw"] = "flat"
    """Advanced: Override the message format mode.

    In most cases, use factory methods (from_raw, from_demos, from_signature_pairs,
    from_kv) instead of setting this directly. DSPy can also infer the mode from
    message structure for raw and demo formats.

    Modes:
    - "raw": LM-style messages with role/content
    - "demo": Few-shot examples with input_fields/output_fields
    - "signature": Dict keys match signature fields → user/assistant pairs
    - "flat": Arbitrary key-value pairs → single user messages (default)
    """

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    @staticmethod
    def _infer_mode_from_msg(msg: dict) -> str:
        """Infer the mode from a message's structure.

        Detection rules (conservative):
        - Raw: has "role" key and ONLY LM-like keys (role, content, tool_calls, tool_call_id, name)
        - Demo: keys are ONLY "input_fields" and/or "output_fields"
        - Flat: everything else (signature mode must be explicit)
        """
        keys = set(msg.keys())
        lm_keys = {"role", "content", "tool_calls", "tool_call_id", "name"}

        if "role" in keys and keys <= lm_keys:
            return "raw"

        if keys <= {"input_fields", "output_fields"} and keys:
            return "demo"

        return "flat"

    def _validate_msg_for_mode(self, msg: dict, mode: str) -> None:
        """Validate a message conforms to the expected mode structure."""
        if mode == "raw":
            if not isinstance(msg.get("role"), str):
                raise ValueError(f"Raw mode: 'role' must be a string: {msg}")
            content = msg.get("content")
            if content is not None and not isinstance(content, (str, list)):
                raise ValueError(f"Raw mode: 'content' must be a string, list, or None: {msg}")

        elif mode == "demo":
            if "input_fields" in msg and not isinstance(msg["input_fields"], dict):
                raise ValueError(f"Demo mode: 'input_fields' must be a dict: {msg}")
            if "output_fields" in msg and not isinstance(msg["output_fields"], dict):
                raise ValueError(f"Demo mode: 'output_fields' must be a dict: {msg}")

        elif mode == "signature":
            if not isinstance(msg, dict) or not msg:
                raise ValueError(f"Signature mode: messages must be non-empty dicts: {msg}")

    def _warn_if_likely_wrong_mode(self, msg: dict, stacklevel: int = 2) -> None:
        """Warn if a flat-mode message looks like it was intended for another mode."""
        keys = set(msg.keys())

        if "role" in keys:
            warnings.warn(
                f"History message has 'role' key but is in flat mode. "
                f"Did you mean to use mode='raw'? Message keys: {sorted(keys)}",
                UserWarning,
                stacklevel=stacklevel,
            )
        elif keys & {"input_fields", "output_fields"}:
            warnings.warn(
                f"History message has 'input_fields'/'output_fields' but is in flat mode. "
                f"Did you mean to use mode='demo'? Message keys: {sorted(keys)}",
                UserWarning,
                stacklevel=stacklevel,
            )

    @pydantic.model_validator(mode="after")
    def _validate_messages(self) -> "History":
        if not self.messages:
            return self

        # Only infer if mode is the default "flat" and messages clearly match another mode
        if self.mode == "flat":
            inferred = self._infer_mode_from_msg(self.messages[0])
            if inferred in {"raw", "demo"}:
                object.__setattr__(self, "mode", inferred)

        for msg in self.messages:
            self._validate_msg_for_mode(msg, self.mode)
            if self.mode == "flat":
                # stacklevel=6: warn -> _warn_if_likely_wrong_mode -> _validate_messages -> validator -> __init__ -> caller
                self._warn_if_likely_wrong_mode(msg, stacklevel=6)

        return self

    def with_messages(self, messages: list[dict[str, Any]]) -> "History":
        """Return a new History with additional messages appended."""
        return History(messages=[*self.messages, *messages], mode=self.mode)

    @classmethod
    def from_raw(cls, messages: list[dict[str, Any]]) -> "History":
        """Create History from LM-style messages with role/content.

        Use this for chat-style conversation history or ReAct trajectories
        that are already formatted as LM messages.

        Example:
            ```python
            history = dspy.History.from_raw([
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ])
            ```
        """
        return cls(messages=messages, mode="raw")

    @classmethod
    def from_demos(cls, examples: list[dict[str, Any]]) -> "History":
        """Create History from few-shot demonstration examples.

        Each example should have 'input_fields' and/or 'output_fields' keys
        containing the respective field dictionaries.

        Example:
            ```python
            history = dspy.History.from_demos([
                {"input_fields": {"question": "2+2?"}, "output_fields": {"answer": "4"}},
            ])
            ```
        """
        return cls(messages=examples, mode="demo")

    @classmethod
    def from_signature_pairs(cls, messages: list[dict[str, Any]]) -> "History":
        """Create History from signature-matched field pairs.

        Each message dict should have keys matching the signature's input/output
        fields. Each dict becomes a user/assistant message pair.

        Example:
            ```python
            history = dspy.History.from_signature_pairs([
                {"question": "What is 2+2?", "answer": "4"},
            ])
            ```
        """
        return cls(messages=messages, mode="signature")

    @classmethod
    def from_kv(cls, messages: list[dict[str, Any]]) -> "History":
        """Create History from arbitrary key-value context.

        Each dict becomes a single user message containing all key-value pairs.
        Use this when you want to pass context that should NOT be split into
        user/assistant turns.

        Example:
            ```python
            history = dspy.History.from_kv([
                {"thought": "I need to search", "tool": "search", "result": "Found it"},
            ])
            ```
        """
        return cls(messages=messages, mode="flat")
