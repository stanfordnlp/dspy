from typing import Any, Literal

import pydantic


class History(pydantic.BaseModel):
    """Class representing the conversation history.

    History supports four message formats:
    
    1. **Signature mode**: Dict keys match signature input/output fields → user/assistant pairs.
       Must be explicitly set via mode="signature".
       ```python
       history = dspy.History(messages=[
           {"question": "What is 2+2?", "answer": "4"},
       ], mode="signature")
       ```
    
    2. **KV mode**: Nested `{"input_fields": {...}, "output_fields": {...}}` → user/assistant pairs.
       ```python
       history = dspy.History.from_kv([
           {"input_fields": {"thought": "...", "tool_name": "search"}, "output_fields": {"observation": "..."}},
       ])
       ```
    
    3. **Dict mode** (default): Arbitrary serializable key-value pairs → all in single user message.
       ```python
       history = dspy.History(messages=[
           {"thought": "I need to search", "tool_name": "search", "observation": "Results found"},
       ])
       ```
    
    4. **Raw mode**: Direct LM messages with `{"role": "user", "content": "..."}` → passed through.
       ```python
       history = dspy.History.from_raw([
           {"role": "user", "content": "Hello"},
           {"role": "assistant", "content": "Hi there!"},
       ])
       ```

    The mode is auto-detected from the first message if not explicitly provided.

    Example:
        ```python
        import dspy

        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        history = dspy.History(
            messages=[
                {"question": "What is the capital of France?", "answer": "Paris"},
                {"question": "What is the capital of Germany?", "answer": "Berlin"},
            ]
        )

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
    mode: Literal["signature", "kv", "dict", "raw"] | None = None

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    def _detect_mode(self, msg: dict) -> str:
        """Detect the mode for a message based on its structure.
        
        Detection rules:
        - Raw: has "role" and "content" keys, but NOT "input_fields"/"output_fields"
        - KV: keys are ONLY "input_fields" and/or "output_fields"
        - Signature: must be explicitly set (requires matching against signature fields)
        - Dict: everything else (default) - arbitrary kv pairs go into user message
        """
        if self.mode:
            return self.mode

        keys = set(msg.keys())

        if {"role", "content"} <= keys and not ({"input_fields", "output_fields"} & keys):
            return "raw"

        if keys <= {"input_fields", "output_fields"} and keys:
            return "kv"

        return "dict"

    @pydantic.model_validator(mode="after")
    def _validate_messages(self) -> "History":
        for msg in self.messages:
            detected = self._detect_mode(msg)

            if detected == "raw":
                if not isinstance(msg.get("role"), str):
                    raise ValueError(f"'role' must be a string: {msg}")
                # content can be None for tool call messages, or string otherwise
                content = msg.get("content")
                if content is not None and not isinstance(content, str):
                    raise ValueError(f"'content' must be a string or None: {msg}")

            elif detected == "kv":
                if "input_fields" in msg and not isinstance(msg["input_fields"], dict):
                    raise ValueError(f"'input_fields' must be a dict: {msg}")
                if "output_fields" in msg and not isinstance(msg["output_fields"], dict):
                    raise ValueError(f"'output_fields' must be a dict: {msg}")

        return self

    def with_messages(self, messages: list[dict[str, Any]]) -> "History":
        """Return a new History with additional messages appended.
        
        Args:
            messages: List of messages to append.
            
        Returns:
            A new History instance with the messages appended.
        """
        return History(messages=[*self.messages, *messages], mode=self.mode)

    @classmethod
    def from_kv(cls, messages: list[dict[str, Any]]) -> "History":
        """Create a History instance with KV mode.
        
        KV mode expects messages with "input_fields" and/or "output_fields" keys,
        each containing a dict of field names to values.
        
        Args:
            messages: List of dicts with "input_fields" and/or "output_fields" keys.
            
        Returns:
            A History instance with mode="kv".
        """
        return cls(messages=messages, mode="kv")

    @classmethod
    def from_raw(cls, messages: list[dict[str, Any]]) -> "History":
        """Create a History instance with raw mode.
        
        Raw mode expects direct LM messages with "role" and "content" keys.
        
        Args:
            messages: List of dicts with "role" and "content" keys.
            
        Returns:
            A History instance with mode="raw".
        """
        return cls(messages=messages, mode="raw")
