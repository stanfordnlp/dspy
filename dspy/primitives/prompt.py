from typing import Any, Optional

from pydantic import BaseModel


class Prompt(BaseModel):
    content: Optional[str]
    messages: Optional[list[dict[str, Any]]]

    @property
    def is_valid(self):
        return self.content is not None or self.messages is not None

    def get_messages(self):

        if not self.is_valid:
            raise ValueError("Prompt has not been initialized with a content or messages")

        if self.messages is not None:
            return self.messages

        return [{"role": "user", "content": self.content}]
