import re
from codecs import getreader
from json import dumps, loads
from typing import IO, Iterator, Literal, cast
from urllib.request import Request, urlopen

from funcgpt.credentials import OPENAI_API_KEY, OPENAI_ORG_ID
from funcgpt.message import Message

DEFAULT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

BASE_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}",
}

if OPENAI_ORG_ID is not None:
    BASE_HEADERS["OpenAI-Organization"] = OPENAI_ORG_ID

match_data = re.compile(r"^data: (.*)(?:\r\n|\r|\n)$").match

match_empty_line = re.compile(r"^(?:\r\n|\r|\n)$").match

utf8reader = getreader("utf-8")


def stream(
    model: Literal["gpt-3.5-turbo", "gpt-4"],
    messages: list[Message],
    temperature: int = 0,
    stop: str | list[str] | None = None,
    chat_completions_url: str = DEFAULT_COMPLETIONS_URL,
) -> Iterator[str]:
    """
    Stream the chat completions for a given model, messages and other parameters.

    :param model: The identifier of the AI model to be used. Can be either "gpt-3.5-turbo" or "gpt-4".
    :param messages: A list of dictionaries representing the message history to be passed to the model.
    :param temperature: Controls randomness in the AI response. Higher values result in more random
                        completions, while lower values steer the model towards more focused responses.
    :param stop: A string or list of strings to specify sequence(s) at which the API will stop
                 generating further tokens. Default is None.
    :param chat_completions_url: The URL endpoint for fetching chat completions. The default value
                                 is DEFAULT_COMPLETIONS_URL.

    :return: An iterator over the formatted chat completions.

    :raises ValueError: Raised if the provided temperature is less than 0.

    .. note:: If the model does not return a completion, the iterator will ignore the response and
              not yield anything.
    """
    # Prepare request payload with model, messages, and temperature
    requestArguments = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }

    # Add "stop" parameter if provided
    if stop is not None:
        requestArguments["stop"] = stop

    request = Request(
        url=chat_completions_url,
        data=dumps(requestArguments).encode(),
        headers=BASE_HEADERS,
    )

    with urlopen(request) as f:  # type: IO[bytes]
        fstr = cast(IO[str], utf8reader(f))
        data = ""

        while True:
            line = fstr.readline()

            # If data is empty, look for a valid data line
            if not data:
                if (dataMatch := match_data(line)) is not None:
                    data = dataMatch.group(1)
                continue

            # If it's not an empty line, reset the data and continue
            if match_empty_line(line) is None:
                data = ""
                continue

            # If data received is "[DONE]", break the loop
            if data == "[DONE]":
                break

            # Process and yield the completion
            body = loads(data)
            data = ""
            delta = body["choices"][0]["delta"]

            if "content" not in delta:
                continue

            deltaContent = delta["content"]

            yield deltaContent


def answer(
    model: Literal["gpt-3.5-turbo", "gpt-4"],
    messages: list[Message],
    temperature: float = 0,
    stop: str | list[str] | None = None,
    chat_completions_url: str = DEFAULT_COMPLETIONS_URL,
) -> str:
    """
    Query an OpenAI model with a list of messages and get the model's answer.

    :param model: The identifier of the AI model to be used. Can be either "gpt-3.5-turbo" or "gpt-4".
    :param messages: A list of Message TypedDicts.
    :param temperature: Controls randomness in the model's response. Higher values (e.g., 1) will
                        generate more random answers, while lower values (e.g., 0) make the model
                        more deterministic. Default is 0.
    :param stop: A string or list of strings to specify sequence(s) at which the API will stop
                 generating further tokens. Default is None.
    :param chat_completions_url: The URL endpoint for fetching chat completions. The default value
                                 is DEFAULT_COMPLETIONS_URL.
    :raises OverflowError: Raised if the model's response exceeds the maximum length.
    :return: The assistant's generated response as a string.
    """
    # Prepare request payload with model, messages, and temperature
    requestArguments = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    # Add "stop" parameter if provided
    if stop is not None:
        requestArguments["stop"] = stop

    # Prepare the HTTP request
    request = Request(
        url=chat_completions_url,
        data=dumps(requestArguments).encode(),
        headers=BASE_HEADERS,
    )

    # Send the request to OpenAI's chat completions API
    response = urlopen(request)

    # Parse the response JSON
    body = loads(response.read())

    # Get the first choice
    choice = body["choices"][0]

    if choice["finish_reason"] == "length":
        raise OverflowError("The model's response exceeded the maximum length.")

    # Extract and return the assistant's response
    return choice["message"]["content"]
