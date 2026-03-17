"""Supervisor orchestrator for routing Experience Study AI Copilot requests."""

import json
import os
import re
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

Intent = Literal["DATA_PREP", "ANALYSIS", "BOTH", "VISUALIZE"]


class StudyOrchestrator:
    """Routes user requests to Data Steward, Lead Actuary, or both."""

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
            "profile",
            "validate",
            "check",
            "band",
            "regroup",
            "clean",
            "prepare",
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
        ]
        has_prep = any(k in q for k in prep_hits)
        has_analysis = any(k in q for k in analysis_hits)
        has_visualize = any(k in q for k in visualize_hits)

        if has_visualize:
            return "VISUALIZE"
        if has_prep and has_analysis:
            return "BOTH"
        if has_prep:
            return "DATA_PREP"
        return "ANALYSIS"

    def _classify_intent(self, user_query: str) -> Intent:
        """Classify request into DATA_PREP, ANALYSIS, or BOTH."""
        if not os.getenv("OPENAI_API_KEY"):
            return self._heuristic_classify(user_query)

        prompt = (
            "Classify the user's request into exactly one label: DATA_PREP, ANALYSIS, BOTH, or VISUALIZE.\n"
            "Definitions:\n"
            "- DATA_PREP: profiling, validation, feature engineering (banding/regrouping).\n"
            "- ANALYSIS: A/E sweeps, ratios, CI interpretation.\n"
            "- BOTH: requires data prep then analysis.\n"
            "- VISUALIZE: charts, plots, treemap, scatter, graphs, or visual reports.\n"
            "Return JSON only, e.g. {\"intent\":\"VISUALIZE\"}."
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
            intent = parsed.get("intent", "").upper()
            if intent in {"DATA_PREP", "ANALYSIS", "BOTH", "VISUALIZE"}:
                return intent  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass

        return self._heuristic_classify(user_query)

    def process_query(self, user_query: str) -> str:
        """Route a query to the appropriate specialized agent(s)."""
        intent = self._classify_intent(user_query)

        if intent == "DATA_PREP":
            return self.data_steward.run(user_query)

        if intent == "ANALYSIS":
            return self.actuary.run(user_query)

        if intent == "VISUALIZE":
            return self.analyst_agent.run(user_query)

        # BOTH: run steward first, then hand context to actuary.
        steward_query = user_query
        q = user_query.lower()
        if "band" in q and "face amount" in q:
            bins_match = re.search(r"\b(\d+)\b", q)
            bins = bins_match.group(1) if bins_match else "4"
            steward_query = f"Create {bins} equal-width bands for the Face_Amount column."

        steward_response = self.data_steward.run(steward_query)

        # Build a clear handoff prompt for the actuary.
        if "1-way" in q and "worst" in q and "face amount" in q:
            handoff_prompt = (
                "Please run a high-level sweep on the data. "
                "What is the worst-performing single cohort by Face Amount?"
            )
        else:
            handoff_prompt = (
                "Data preparation step is complete. Use the processed dataset and provide actuarial interpretation.\n\n"
                f"Original user request:\n{user_query}\n\n"
                f"Data Steward output:\n{steward_response}\n\n"
                "Now run the appropriate A/E sweep and provide a concise professional summary."
            )
        actuary_response = self.actuary.run(handoff_prompt)

        return (
            "### Data Steward Output\n"
            f"{steward_response}\n\n"
            "### Lead Actuary Output\n"
            f"{actuary_response}"
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
