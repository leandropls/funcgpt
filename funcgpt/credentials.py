from __future__ import annotations

from os import getenv

if (OPENAI_API_KEY := getenv("OPENAI_API_KEY")) is None:
    raise RuntimeError("OPENAI_API_KEY environment variable not set")

OPENAI_ORG_ID = getenv("OPENAI_ORG_ID")
