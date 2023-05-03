import bz2
from pathlib import Path
from typing import Literal, TypedDict
from unittest.mock import patch
from urllib.parse import urlparse

from tiktoken import Encoding as TikTokenEncoding
from tiktoken_ext.openai_public import cl100k_base

from funcgpt.message import Message

__all__ = ["serialize_to_gpt", "get_token_count"]

MSG_SEP = {
    "gpt-3.5-turbo": "\n",
    "gpt-4": "",
}

IM_SEP = {
    "gpt-3.5-turbo": "\n",
    "gpt-4": "<|im_sep|>",
}

IM_START = {
    "gpt-3.5-turbo": "<|im_start|>",
    "gpt-4": "<|im_start|>",
}

IM_END = {
    "gpt-3.5-turbo": "<|im_end|>",
    "gpt-4": "<|im_end|>",
}


def read_file_cached(blobpath: str) -> bytes:
    """
    Reads a cached file given a URL path to the original file.

    The cache is stored in the "data" directory, and the cached files are
    compressed using the bz2 format. If the cached file is not found, this
    function raises a FileNotFoundError.

    :param blobpath: The URL path of the original file.
                     Example: "https://example.com/data/file.txt"
    :return: The content of the cached file as bytes.
    :raises FileNotFoundError: If the cached file is not found.
    """
    # Get the cache directory path
    cache_dir = Path(__file__).parent / "data"

    # Generate the cache key by extracting the file name from the URL and adding ".bz2" extension
    cache_key = Path(urlparse(blobpath).path).name + ".bz2"

    # Construct the full cache file path
    cache_path = cache_dir / cache_key

    # Read and return the content of the bz2-compressed cached file
    with bz2.open(cache_path, "rb") as f:
        return f.read()


class EncodingParameters(TypedDict):
    name: str
    pat_str: str
    mergeable_ranks: dict[bytes, int]
    special_tokens: dict[str, int]


@patch("tiktoken.load.read_file_cached", read_file_cached)
def get_cl100k_im_encoding() -> TikTokenEncoding:
    """
    Generate a tiktoken Encoding for cl100k_im with appropriate special tokens.

    :return: the tiktoken Encoding instance
    """
    # Obtain base encoding parameters for cl100k
    parameters: EncodingParameters = cl100k_base()

    # Return an encoding instance with the specific cl100k_im parameters and special tokens
    return TikTokenEncoding(
        name="cl100k_im",
        pat_str=parameters["pat_str"],
        mergeable_ranks=parameters["mergeable_ranks"],
        special_tokens={
            **parameters["special_tokens"],
            IM_START["gpt-4"]: 100264,
            IM_END["gpt-4"]: 100265,
            IM_SEP["gpt-4"]: 100266,
        },
    )


CL100K_IM_ENCODING = get_cl100k_im_encoding()


def serialize_to_gpt(
    messages: list[Message],
    model: Literal["gpt-3.5-turbo", "gpt-4"],
) -> list[int]:
    """
    Serialize the given list of messages to a format that can be consumed by the specified GPT model by
    converting them into their respective encoding indices.

    :param messages: A list of message dictionaries, where each dictionary contains a "role" (either
                     "system", "user", or "assistant") and a "content" (string).
    :param model: The GPT model to be used ("gpt-3.5-turbo" or "gpt-4").
    :return: A list of integer encoding indices that represent the serialized messages.
    """
    # Constants for the start, end, and separator placeholders
    imStart = IM_START[model]
    imEnd = IM_END[model]
    imSep = IM_SEP[model]
    msgSep = MSG_SEP[model]

    pieces = []
    for message in messages:
        # Format each message into its corresponding encoded format
        pieces.append(f"{imStart}{message['role']}{imSep}{message['content']}{imEnd}")
    # Add the final assistant prompt
    pieces.append(f"{imStart}assistant{imSep}")
    # Join the encoded message pieces with a separator
    serialized = msgSep.join(pieces)

    return CL100K_IM_ENCODING.encode(serialized, allowed_special="all")


def get_token_count(
    messages: list[Message],
    model: Literal["gpt-3.5-turbo", "gpt-4"],
) -> int:
    """
    Determine the number of tokens for a given list of messages.

    :param messages: A list of message dictionaries, where each dictionary contains a "role" (either
                     "system", "user", or "assistant") and a "content" (string).
    :param model: The GPT model to be used ("gpt-3.5-turbo" or "gpt-4").
    :return: The total number of tokens in the serialized messages.
    """
    return len(serialize_to_gpt(messages, model))
