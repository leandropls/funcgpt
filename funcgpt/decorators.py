from typing import Callable, Literal, TypeVar, overload

from funcgpt.protocols import GPTAnswerProtocol, GPTStreamProtocol
from funcgpt.wrapper import create_generic_wrapper

T = TypeVar("T", bound=GPTAnswerProtocol | GPTStreamProtocol)

__all__ = ["gpt"]


@overload
def gpt(f: T) -> T:
    ...


@overload
def gpt(
    model: Literal["gpt-3.5-turbo", "gpt-4"] = "gpt-3.5-turbo",
    temperature: int = 0,
    max_tokens: int | None = None,
) -> Callable[[T], T]:
    ...


def gpt(
    *args,
    model: Literal["gpt-3.5-turbo", "gpt-4"] = "gpt-3.5-turbo",
    temperature: int = 0,
    max_tokens: int | None = None,
):
    if len(args) == 1 and callable(f := args[0]):
        return create_generic_wrapper(
            f=f, model=model, temperature=temperature, max_tokens=max_tokens
        )
    else:
        return lambda f_: create_generic_wrapper(
            f_, model=model, temperature=temperature, max_tokens=max_tokens
        )
