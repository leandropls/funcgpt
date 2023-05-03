from typing import Literal, TypedDict


class Message(TypedDict):
    """
    A message sent or received by the assistant.

    :param role: The role of the message sender or receiver. Can be "system",
                 "user" or "assistant".
    :param content: The text content of the message.
    """

    role: Literal["system", "user", "assistant"]
    content: str
