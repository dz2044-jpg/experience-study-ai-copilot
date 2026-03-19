import os

from agents.orchestrator import StudyOrchestrator


def test_continue_after_data_prep_routes_to_pending_analysis(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    orchestrator = StudyOrchestrator()
    calls = []

    monkeypatch.setattr(
        orchestrator.data_steward,
        "run",
        lambda message: calls.append(("steward", message)) or "steward-ok",
    )
    monkeypatch.setattr(
        orchestrator.actuary,
        "run",
        lambda message: calls.append(("actuary", message)) or "actuary-ok",
    )

    first = orchestrator.process_query("profile the inforce dataset")
    second = orchestrator.process_query("continue")

    assert first == "steward-ok"
    assert second == "actuary-ok"
    assert calls[0] == ("steward", "profile the inforce dataset")
    assert calls[1][0] == "actuary"
    assert "1-way dimensional sweep" in calls[1][1]


def test_continue_after_analysis_routes_to_pending_visualization(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    orchestrator = StudyOrchestrator()
    calls = []

    monkeypatch.setattr(
        orchestrator.actuary,
        "run",
        lambda message: calls.append(("actuary", message)) or "actuary-ok",
    )
    monkeypatch.setattr(
        orchestrator.analyst_agent,
        "run",
        lambda message: calls.append(("analyst", message)) or "analyst-ok",
    )

    first = orchestrator.process_query("run a 1-way sweep")
    second = orchestrator.process_query("proceed")

    assert first == "actuary-ok"
    assert second == "analyst-ok"
    assert calls[0] == ("actuary", "run a 1-way sweep")
    assert calls[1] == (
        "analyst",
        "Create a scatter report using the amount metric from the latest sweep summary.",
    )
