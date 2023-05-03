from __future__ import annotations

from functools import wraps
from inspect import signature
from textwrap import dedent
from typing import Callable, Iterator, Literal

from funcgpt.gpt import answer, stream
from funcgpt.tokentools import get_token_count

MAX_TOKENS = {
    "gpt-3.5-turbo": 4096,
    "gpt-4": 8192,
}

DEFAULT_PROMPT_TOKENS_SHARE = 0.875  # 7/8


def create_generic_wrapper(
    f: Callable,
    model: Literal["gpt-3.5-turbo", "gpt-4"],
    temperature: int,
    max_tokens: int | None = None,
) -> Callable[[str], str | Iterator[str]]:
    """
    Create a generic wrapper for a callable (typically a function) to generate response
    based on the callable's signature and docstring.

    :param f: The callable to be wrapped.
    :param model: The OpenAI model to use for generating responses. Must be one of:
                  "gpt-3.5-turbo" or "gpt-4".
    :param temperature: The temperature value to use for generating the responses.
                        Lower values make the responses more focused and deterministic,
                        while higher values make them more diverse.
    :param max_tokens: The maximum number of tokens for the input prompt. If not provided,
                        the maximum number of tokens will be determined based on the model
                        and the default prompt tokens share.

    :return: A wrapped callable that generates responses according to the specifications
             and instructions provided in the function docstring.

    :raises ValueError: If the callable does not have a docstring or if the return
                        annotation is not of the types str, Iterator[str] or bool.
    """
    # Extract the callable's docstring
    fdoc: str | None = f.__doc__
    if fdoc is None:
        raise ValueError("Function must have a docstring")

    # Extract the callable's return annotation
    fsig = signature(f)
    return_annotation = fsig.return_annotation

    # Determine the engine to use for generating responses
    if return_annotation is str:
        engine = answer
    elif return_annotation is Iterator[str]:
        engine = stream
    elif return_annotation is bool:
        # Wrap the answer function with special handling for boolean return values
        engine = (
            lambda *args, **kwargs: "true"
            in answer(*args, **kwargs, stop=["true", "false"]).lower()
        )
    else:
        raise ValueError("Function must have a return annotation of str, Iterator[str], or bool")

    # Create the instructions for the GPT engine
    instructions = "You should answer to inputs according to the following specification:\n\n"
    instructions += dedent(fdoc).strip()

    if return_annotation is bool:
        instructions += (
            "\n\nAnswer with either true or false without including any other text. "
            "If no definitive answer can be given, answer false."
        )
    else:
        instructions += "\n\nAnswer with only what was requested without including any other text."

    # Determine the system role based on the model
    if model == "gpt-3.5-turbo":
        systemRole: Literal["system", "user"] = "user"
    else:
        systemRole = "system"

    # Determine the maximum number of tokens for the input prompt
    if max_tokens is None:
        max_tokens = int(MAX_TOKENS[model] * DEFAULT_PROMPT_TOKENS_SHARE)

    # Create the wrapper function
    @wraps(f)
    def wrapper(message: str) -> str | Iterator[str]:
        messages = [
            {"role": systemRole, "content": instructions},
            {"role": "user", "content": message},
        ]
        message_tokens_count = get_token_count(messages=messages, model=model)
        if message_tokens_count > max_tokens:
            raise ValueError(
                f"Message exceeds maximum number of tokens ({message_tokens_count} > {max_tokens})"
            )
        return engine(
            model=model,
            messages=messages,
            temperature=temperature,
        )

    # Return the wrapper function
    return wrapper
