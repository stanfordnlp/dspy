"""History type for conversation history."""

from typing import List, Dict, Any


class History:
    """Class representing the conversation history.

    The conversation history is a list of messages, each message entity should have keys from the associated signature.
    For example, if you have the following signature:

    ```
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()
    ```

    Then the history should be a list of dictionaries with keys "question" and "answer".

    Example:
        ```
        import dspy

        dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

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
        ```
        import dspy

        dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

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
    
    def __init__(self, messages: List[Dict[str, Any]] = None):
        self.messages = messages or []
        # Validate that messages is a list of dictionaries
        if not isinstance(self.messages, list):
            raise ValueError("messages must be a list")
        for msg in self.messages:
            if not isinstance(msg, dict):
                raise ValueError("Each message must be a dictionary")
    
    def add_message(self, role: str, content: str):
        """Add a message to the history."""
        self.messages.append({"role": role, "content": content})
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in the history."""
        return self.messages
    
    def clear(self):
        """Clear the conversation history."""
        self.messages = []
    
    def __len__(self) -> int:
        return len(self.messages)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.messages[index]
    
    def __iter__(self):
        return iter(self.messages)
    
    def __repr__(self) -> str:
        return f"History(messages={self.messages})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, History):
            return False
        return self.messages == other.messages
    
    def __hash__(self) -> int:
        return hash(tuple(sorted(str(msg) for msg in self.messages)))
    
    def copy(self) -> "History":
        """Create a copy of the history."""
        return History(messages=self.messages.copy())
    
    def extend(self, other: "History"):
        """Extend this history with messages from another history."""
        if isinstance(other, History):
            self.messages.extend(other.messages)
        elif isinstance(other, list):
            self.messages.extend(other)
        else:
            raise ValueError("Can only extend with History or list of messages")
    
    def append(self, message: Dict[str, Any]):
        """Append a single message to the history."""
        if not isinstance(message, dict):
            raise ValueError("Message must be a dictionary")
        self.messages.append(message)
    
    def insert(self, index: int, message: Dict[str, Any]):
        """Insert a message at a specific index."""
        if not isinstance(message, dict):
            raise ValueError("Message must be a dictionary")
        self.messages.insert(index, message)
    
    def pop(self, index: int = -1) -> Dict[str, Any]:
        """Remove and return a message at the specified index."""
        return self.messages.pop(index)
    
    def remove(self, message: Dict[str, Any]):
        """Remove a specific message from the history."""
        self.messages.remove(message)
    
    def reverse(self):
        """Reverse the order of messages in the history."""
        self.messages.reverse()
    
    def sort(self, key=None, reverse=False):
        """Sort the messages in the history."""
        self.messages.sort(key=key, reverse=reverse)
    
    def count(self, message: Dict[str, Any]) -> int:
        """Count occurrences of a specific message."""
        return self.messages.count(message)
    
    def index(self, message: Dict[str, Any]) -> int:
        """Find the index of a specific message."""
        return self.messages.index(message)
    
    def __contains__(self, message: Dict[str, Any]) -> bool:
        """Check if a message is in the history."""
        return message in self.messages