from agents.agent_actuary import ActuaryAgent
from agents.agent_analyst import AnalystAgent
from agents.agent_steward import DataStewardAgent
from agents.model_config import (
    DEFAULT_ACTUARY_MODEL,
    DEFAULT_ROUTER_MODEL,
    DEFAULT_STEWARD_MODEL,
    resolve_actuary_model,
    resolve_router_model,
    resolve_steward_model,
)
from agents.orchestrator import StudyOrchestrator


def test_model_resolution_uses_repo_defaults_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("OPENAI_ROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_STEWARD_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_ACTUARY_MODEL", raising=False)

    assert resolve_router_model() == DEFAULT_ROUTER_MODEL
    assert resolve_steward_model() == DEFAULT_STEWARD_MODEL
    assert resolve_actuary_model() == DEFAULT_ACTUARY_MODEL


def test_model_resolution_uses_env_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_ROUTER_MODEL", "router-test-model")
    monkeypatch.setenv("OPENAI_STEWARD_MODEL", "steward-test-model")
    monkeypatch.setenv("OPENAI_ACTUARY_MODEL", "actuary-test-model")

    assert resolve_router_model() == "router-test-model"
    assert resolve_steward_model() == "steward-test-model"
    assert resolve_actuary_model() == "actuary-test-model"


def test_agent_defaults_use_supported_chat_model_configuration(monkeypatch):
    monkeypatch.delenv("OPENAI_ROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_STEWARD_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_ACTUARY_MODEL", raising=False)

    orchestrator = StudyOrchestrator()
    steward = DataStewardAgent()
    actuary = ActuaryAgent()
    analyst = AnalystAgent()

    assert orchestrator.classifier_model == "gpt-5-nano"
    assert steward.model == "gpt-5-mini"
    assert actuary.model == "gpt-5"
    assert analyst.model is None
    assert "gpt-5.3-codex" not in {
        orchestrator.classifier_model,
        steward.model,
        actuary.model,
    }
