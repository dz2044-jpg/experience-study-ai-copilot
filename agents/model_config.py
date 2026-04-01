"""Centralized OpenAI model defaults and env override resolution."""

import os
from typing import Optional

DEFAULT_ROUTER_MODEL = "gpt-5-nano"
DEFAULT_STEWARD_MODEL = "gpt-5-mini"
DEFAULT_ACTUARY_MODEL = "gpt-5"


def _resolve_model(explicit_model: Optional[str], env_var: str, default_model: str) -> str:
    """Resolve a model from an explicit value, env override, or repository default."""
    if explicit_model and explicit_model.strip():
        return explicit_model.strip()

    env_value = os.getenv(env_var, "").strip()
    if env_value:
        return env_value

    return default_model


def resolve_router_model(explicit_model: Optional[str] = None) -> str:
    """Resolve the orchestrator classifier model."""
    return _resolve_model(explicit_model, "OPENAI_ROUTER_MODEL", DEFAULT_ROUTER_MODEL)


def resolve_steward_model(explicit_model: Optional[str] = None) -> str:
    """Resolve the data steward tool-calling model."""
    return _resolve_model(explicit_model, "OPENAI_STEWARD_MODEL", DEFAULT_STEWARD_MODEL)


def resolve_actuary_model(explicit_model: Optional[str] = None) -> str:
    """Resolve the actuary tool-calling model."""
    return _resolve_model(explicit_model, "OPENAI_ACTUARY_MODEL", DEFAULT_ACTUARY_MODEL)
