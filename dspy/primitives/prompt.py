from enum import Enum
from typing import Union

from pydantic import BaseModel


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ContentType(Enum):
    TEXT = "text"
    IMAGE_URL = "image_url"

class TextContent(BaseModel):
    type: ContentType
    text: str

class URL(BaseModel):
    url: str

class ImageContent(BaseModel):
    type: ContentType
    image_url: URL

class Message(BaseModel):
    role: Role
    content: Union[str, ImageContent, TextContent]

    @classmethod
    def from_dict(cls, data: dict):
        return cls.model_validate(data)

class Prompt(BaseModel):
    messages: list[Message]

    @classmethod
    def from_str(cls, string: str):
        return Prompt(messages=[Message.from_dict({"role": "user", "content": string})])

    @property
    def messages_data(self) -> list[dict]:
        return [message.model_dump() for message in self.messages]

    def to_str(self) -> str:
        return "\n\n---\n\n".join([message.content for message in self.messages])
