from typing import Any, Literal

import pydantic


class History(pydantic.BaseModel):
    """Class representing conversation history.

    History supports four message formats, with one mode per History instance:

    1. **Raw mode**: Direct LM messages with `{"role": "...", "content": "..."}`.
       Used for ReAct trajectories and native tool calling.
       ```python
       history = dspy.History.from_raw([
           {"role": "user", "content": "Hello"},
           {"role": "assistant", "content": "Hi there!"},
       ])
       ```

    2. **Demo mode**: Nested `{"input_fields": {...}, "output_fields": {...}}` pairs.
       Used for few-shot demonstrations with explicit input/output separation.
       ```python
       history = dspy.History.from_demo([
           {"input_fields": {"question": "2+2?"}, "output_fields": {"answer": "4"}},
       ])
       ```

    3. **Flat mode** (default): Arbitrary key-value pairs in a single user message.
       ```python
       history = dspy.History(messages=[
           {"thought": "I need to search", "tool_name": "search", "observation": "Found it"},
       ])
       ```

    4. **Signature mode**: Dict keys match signature fields → user/assistant pairs.
       Must be explicitly set.
       ```python
       history = dspy.History.from_signature([
           {"question": "What is 2+2?", "answer": "4"},
       ])
       ```

    Example:
        ```python
        import dspy

        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        history = dspy.History.from_signature([
            {"question": "What is the capital of France?", "answer": "Paris"},
        ])

        predict = dspy.Predict(MySignature)
        outputs = predict(question="What is the capital of France?", history=history)
        ```

    Example of capturing the conversation history:
        ```python
        import dspy

        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        predict = dspy.Predict(MySignature)
        outputs = predict(question="What is the capital of France?")
        history = dspy.History(messages=[{"question": "What is the capital of France?", **outputs}])
        outputs_with_history = predict(question="Are you sure?", history=history)
        ```
    """

    messages: list[dict[str, Any]]
    mode: Literal["signature", "demo", "flat", "raw"] = "flat"

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
            if content is not None and not isinstance(content, str):
                raise ValueError(f"Raw mode: 'content' must be a string or None: {msg}")

        elif mode == "demo":
            if "input_fields" in msg and not isinstance(msg["input_fields"], dict):
                raise ValueError(f"Demo mode: 'input_fields' must be a dict: {msg}")
            if "output_fields" in msg and not isinstance(msg["output_fields"], dict):
                raise ValueError(f"Demo mode: 'output_fields' must be a dict: {msg}")

        elif mode == "signature":
            if not isinstance(msg, dict) or not msg:
                raise ValueError(f"Signature mode: messages must be non-empty dicts: {msg}")

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

        return self

    def with_messages(self, messages: list[dict[str, Any]]) -> "History":
        """Return a new History with additional messages appended."""
        return History(messages=[*self.messages, *messages], mode=self.mode)

    @classmethod
    def from_demo(cls, messages: list[dict[str, Any]]) -> "History":
        """Create a History with demo mode.

        Demo mode expects messages with "input_fields" and/or "output_fields" keys.
        """
        return cls(messages=messages, mode="demo")

    @classmethod
    def from_raw(cls, messages: list[dict[str, Any]]) -> "History":
        """Create a History with raw mode.

        Raw mode expects direct LM messages with "role" and "content" keys.
        """
        return cls(messages=messages, mode="raw")

    @classmethod
    def from_signature(cls, messages: list[dict[str, Any]]) -> "History":
        """Create a History with signature mode.

        Signature mode expects dicts with keys matching the signature's fields.
        """
        return cls(messages=messages, mode="signature")
