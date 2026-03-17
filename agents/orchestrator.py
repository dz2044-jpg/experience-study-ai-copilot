"""Supervisor orchestrator for routing Experience Study AI Copilot requests."""

import json
import os
import sys
import re
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.agent_actuary import ActuaryAgent
from agents.agent_analyst import AnalystAgent
from agents.agent_steward import DataStewardAgent

Intent = Literal["GENERAL", "DATA_PREP", "ANALYSIS", "VISUALIZE"]


class StudyOrchestrator:
    """Routes user requests to Data Steward, Lead Actuary, Analyst, or general response."""

    def __init__(self, classifier_model: str = "gpt-5-nano") -> None:
        self.classifier_model = classifier_model
        self.client: OpenAI = client

        self.data_steward = DataStewardAgent()
        self.actuary = ActuaryAgent()
        self.analyst_agent = AnalystAgent()

    def _heuristic_classify(self, user_query: str) -> Intent:
        """Fast deterministic fallback classifier when API key is unavailable."""
        q = user_query.lower()
        prep_hits = [
            "read",
            "profile",
            "validate",
            "check",
            "band",
            "regroup",
            "clean",
            "prepare",
            "column",
            "feature",
            "bucket",
            "dataset",
            "inforce",
        ]
        analysis_hits = [
            "a/e",
            "sweep",
            "ci",
            "confidence interval",
            "analysis",
            "worst-performing",
            "ratio",
            "cohort",
            "mortality",
            "exposure",
            "calculate",
            "run",
            "trend",
        ]
        visualize_hits = [
            "chart",
            "plot",
            "visual",
            "visualize",
            "treemap",
            "scatter",
            "graph",
            "report",
            "heatmap",
        ]
        has_prep = any(k in q for k in prep_hits)
        has_analysis = any(k in q for k in analysis_hits)
        has_visualize = any(k in q for k in visualize_hits)

        if has_visualize:
            return "VISUALIZE"
        if has_prep and has_analysis:
            # NO CHAINING: classify only the first logical step.
            return "DATA_PREP"
        if has_prep:
            return "DATA_PREP"
        if has_analysis:
            return "ANALYSIS"
        return "GENERAL"

    def _classify_intent(self, user_query: str) -> Intent:
        """Classify request into GENERAL, DATA_PREP, ANALYSIS, or VISUALIZE."""
        if not os.getenv("OPENAI_API_KEY"):
            return self._heuristic_classify(user_query)

        prompt = (
            "You are the Master Orchestrator for an Actuarial AI Copilot. "
            "Classify the user's request into exactly one label: GENERAL, DATA_PREP, ANALYSIS, or VISUALIZE.\n\n"
            "Definitions & Routing Boundaries:\n"
            "- GENERAL: Questions about the tool, actuarial concepts, or casual conversation.\n"
            "- DATA_PREP: Data profiling, validation, or feature engineering (banding/grouping). Routes to Data Steward.\n"
            "- ANALYSIS: Running A/E calculations, dimensional sweeps, or statistical credibility checks. Routes to Lead Actuary.\n"
            "- VISUALIZE: Generating charts, treemaps, scatter plots, or visual reports. Routes to Analyst Agent.\n\n"
            "STRICT ORCHESTRATOR GUARDRAILS:\n"
            "1. NO CHAINING: If the user asks for multiple steps (e.g., 'Group the data and then run a sweep'), you MUST classify ONLY the first logical step. We must execute one agent at a time.\n"
            "2. NO MOONLIGHTING: Do not route math/sweeps to DATA_PREP. Do not route column manipulation to ANALYSIS.\n\n"
            "Return JSON only, e.g., {\"intent\":\"DATA_PREP\"}."
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.classifier_model,
                messages=[
                    {"role": "system", "content": "You are a precise intent classifier."},
                    {"role": "user", "content": f"{prompt}\n\nUser request:\n{user_query}"},
                ],
                temperature=0,
            )
        except Exception:
            return self._heuristic_classify(user_query)
        content = completion.choices[0].message.content or "{}"

        try:
            parsed = json.loads(content)
            intent = parsed.get("intent", "").strip().upper()
            if intent in {"GENERAL", "DATA_PREP", "ANALYSIS", "VISUALIZE"}:
                return intent  # type: ignore[return-value]
        except json.JSONDecodeError:
            # If model returned non-JSON text, recover by keyword extraction.
            for label in ("GENERAL", "DATA_PREP", "ANALYSIS", "VISUALIZE"):
                if re.search(rf"\b{label}\b", content.upper()):
                    return label  # type: ignore[return-value]

        return self._heuristic_classify(user_query)

    def process_query(self, user_query: str) -> str:
        """Route a query to the appropriate specialized agent(s)."""
        intent = self._classify_intent(user_query)

        if intent == "GENERAL":
            # Secondary guardrail routing for ambiguous natural phrasing.
            q = user_query.lower()
            if any(
                k in q
                for k in (
                    "read",
                    "profile",
                    "validate",
                    "band",
                    "regroup",
                    "feature",
                    "column",
                    "bucket",
                    "dataset",
                    "inforce",
                )
            ):
                return self.data_steward.run(user_query)
            if any(k in q for k in ("a/e", "sweep", "ci", "confidence", "mortality", "ratio", "cohort", "trend")):
                return self.actuary.run(user_query)
            if any(k in q for k in ("chart", "plot", "visual", "treemap", "scatter", "graph", "heatmap")):
                return self.analyst_agent.run(user_query)
            return (
                "Tell me which step you want first: DATA_PREP, ANALYSIS, or VISUALIZE. "
                "Example: 'Run a 2-way sweep with min_mac=2'."
            )

        if intent == "DATA_PREP":
            return self.data_steward.run(user_query)

        if intent == "ANALYSIS":
            return self.actuary.run(user_query)

        if intent == "VISUALIZE":
            return self.analyst_agent.run(user_query)
        return self._heuristic_classify(user_query)


if __name__ == "__main__":
    orchestrator = StudyOrchestrator()

    test_query = (
        "Please group the Face Amounts into 4 equal-width bands. Once that is done, "
        "run a 1-way sweep to tell me which of those new Face Amount bands has the "
        "worst A/E ratio by amount."
    )

    print("=== Study Orchestrator Handoff Test ===")
    print(orchestrator.process_query(test_query))
