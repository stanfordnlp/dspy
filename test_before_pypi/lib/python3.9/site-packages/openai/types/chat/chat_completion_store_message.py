# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.


from .chat_completion_message import ChatCompletionMessage

__all__ = ["ChatCompletionStoreMessage"]


class ChatCompletionStoreMessage(ChatCompletionMessage):
    id: str
    """The identifier of the chat message."""
