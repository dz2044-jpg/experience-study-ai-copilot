"""Helpers for optional OpenAI client loading in local/offline environments."""

import os
from typing import Any, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - depends on environment
    load_dotenv = None  # type: ignore[assignment]

if load_dotenv is not None:
    load_dotenv()

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on environment
    OpenAI = None  # type: ignore[assignment]


def build_openai_client() -> Optional[Any]:
    """Return an OpenAI client when both the SDK and API key are available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if OpenAI is None or not api_key:
        return None
    return OpenAI(api_key=api_key)
