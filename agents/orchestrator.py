"""Supervisor orchestrator for routing Experience Study AI Copilot requests."""

import json
import os
import sys
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

Intent = Literal["GENERAL", "DATA_PREP", "ANALYSIS", "VISUALIZE", "CONTINUE"]

# Tracks the last agent used so we can route continuations back to them
_LAST_ACTIVE_AGENT = None


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
        continuation_tokens = {
            "yes",
            "y",
            "ok",
            "okay",
            "proceed",
            "continue",
            "go ahead",
            "do it",
            "option 1",
            "option 2",
            "option 3",
        }
        if q.strip() in continuation_tokens:
            return "CONTINUE"

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
        """Classify request into GENERAL, DATA_PREP, ANALYSIS, VISUALIZE, or CONTINUE."""
        if not os.getenv("OPENAI_API_KEY"):
            return self._heuristic_classify(user_query)

        prompt = (
            "You are the Master Orchestrator for an Actuarial AI Copilot. "
            "Classify the user's request into exactly one label: GENERAL, DATA_PREP, ANALYSIS, VISUALIZE, or CONTINUE.\n\n"
            "Definitions & Routing Boundaries:\n"
            "- GENERAL: Questions about the tool, actuarial concepts, or casual conversation.\n"
            "- DATA_PREP: Data profiling, validation, or feature engineering. Routes to Data Steward.\n"
            "- ANALYSIS: Running A/E calculations or dimensional sweeps. Routes to Lead Actuary.\n"
            "- VISUALIZE: Generating charts, treemaps, or visual reports. Routes to Analyst Agent.\n"
            "- CONTINUE: The user is answering a direct question, confirming an action (e.g., 'proceed', 'yes', 'option 1'), or providing a short reply to the previous output.\n\n"
            "Return RAW JSON ONLY. Do not use markdown formatting. Do not wrap your response in ```json or ``` backticks. Example: {\"intent\": \"GENERAL\"}"
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
        raw_response = completion.choices[0].message.content or "{}"
        clean_response = raw_response.strip()

        # Aggressively strip markdown code blocks if the model hallucinated them.
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        elif clean_response.startswith("```"):
            clean_response = clean_response[3:]

        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]

        clean_response = clean_response.strip()

        # Safely attempt to parse cleaned JSON.
        try:
            response_json = json.loads(clean_response)
            intent = str(response_json.get("intent", "GENERAL")).strip().upper()
            if intent in {"GENERAL", "DATA_PREP", "ANALYSIS", "VISUALIZE", "CONTINUE"}:
                return intent  # type: ignore[return-value]
        except json.JSONDecodeError as exc:
            # Keep conversation flowing even if classifier returns malformed JSON.
            print(f"[Orchestrator Debug] JSON parsing failed: {exc}. Falling back to GENERAL.")
            return "GENERAL"

        return self._heuristic_classify(user_query)

    def process_query(self, user_query: str) -> str:
        """Route a query to the appropriate specialized agent(s)."""
        global _LAST_ACTIVE_AGENT
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
                _LAST_ACTIVE_AGENT = "STEWARD"
                return self.data_steward.run(user_query)
            if any(k in q for k in ("a/e", "sweep", "ci", "confidence", "mortality", "ratio", "cohort", "trend")):
                _LAST_ACTIVE_AGENT = "ACTUARY"
                return self.actuary.run(user_query)
            if any(k in q for k in ("chart", "plot", "visual", "treemap", "scatter", "graph", "heatmap")):
                _LAST_ACTIVE_AGENT = "ANALYST"
                return self.analyst_agent.run(user_query)
            return (
                "I can help with data preparation, actuarial analysis, or visualization. "
                "Share what you want to do, and I will route it."
            )

        if intent == "DATA_PREP":
            _LAST_ACTIVE_AGENT = "STEWARD"
            return self.data_steward.run(user_query)

        if intent == "ANALYSIS":
            _LAST_ACTIVE_AGENT = "ACTUARY"
            return self.actuary.run(user_query)

        if intent == "VISUALIZE":
            _LAST_ACTIVE_AGENT = "ANALYST"
            return self.analyst_agent.run(user_query)

        if intent == "CONTINUE":
            # If a "continue" message actually contains a concrete new request
            # (e.g., a sweep/analysis command), prioritize that intent.
            followup_intent = self._heuristic_classify(user_query)
            if followup_intent == "DATA_PREP":
                _LAST_ACTIVE_AGENT = "STEWARD"
                return self.data_steward.run(user_query)
            if followup_intent == "ANALYSIS":
                _LAST_ACTIVE_AGENT = "ACTUARY"
                return self.actuary.run(user_query)
            if followup_intent == "VISUALIZE":
                _LAST_ACTIVE_AGENT = "ANALYST"
                return self.analyst_agent.run(user_query)

            if _LAST_ACTIVE_AGENT == "STEWARD":
                return self.data_steward.run(user_query)
            if _LAST_ACTIVE_AGENT == "ACTUARY":
                return self.actuary.run(user_query)
            if _LAST_ACTIVE_AGENT == "ANALYST":
                return self.analyst_agent.run(user_query)
            return (
                "I can route this request to Data Steward, Lead Actuary, or Analyst. "
                "Please specify whether you want data prep, analysis, or visualization."
            )

        return (
            "I can route this request to Data Steward, Lead Actuary, or Analyst. "
            "Please specify whether you want data prep, analysis, or visualization."
        )


if __name__ == "__main__":
    orchestrator = StudyOrchestrator()

    test_query = (
        "Please group the Face Amounts into 4 equal-width bands. Once that is done, "
        "run a 1-way sweep to tell me which of those new Face Amount bands has the "
        "worst A/E ratio by amount."
    )

    print("=== Study Orchestrator Handoff Test ===")
    print(orchestrator.process_query(test_query))
