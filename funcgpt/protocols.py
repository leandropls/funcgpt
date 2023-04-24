from typing import Iterator, Protocol


class GPTAnswerProtocol(Protocol):
    def __call__(self, message: str) -> str:
        ...


class GPTStreamProtocol(Protocol):
    def __call__(self, message: str) -> Iterator[str]:
        ...
