"""Supervisor orchestrator for routing Experience Study AI Copilot requests."""

import json
import os
import re
import sys
from pathlib import Path
from typing import Literal, Optional

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.agent_actuary import ActuaryAgent
from agents.agent_analyst import AnalystAgent
from agents.openai_compat import build_openai_client
from agents.agent_steward import DataStewardAgent

Intent = Literal["GENERAL", "DATA_PREP", "ANALYSIS", "VISUALIZE", "CONTINUE"]

class StudyOrchestrator:
    """Routes user requests to Data Steward, Lead Actuary, Analyst, or general response."""

    def __init__(self, classifier_model: str = "gpt-5-nano") -> None:
        self.classifier_model = classifier_model
        self.client = build_openai_client()

        self.data_steward = DataStewardAgent()
        self.actuary = ActuaryAgent()
        self.analyst_agent = AnalystAgent()
        self.last_active_agent: Optional[Literal["STEWARD", "ACTUARY", "ANALYST"]] = None
        self.pending_analysis_prompt: Optional[str] = None
        self.pending_visualization_prompt: Optional[str] = None
        self.latest_analysis_output_path: Optional[str] = None

    @staticmethod
    def _is_continue_message(user_query: str) -> bool:
        """Detect short confirmation-style follow-ups."""
        continuation_tokens = {
            "yes",
            "y",
            "ok",
            "okay",
            "proceed",
            "continue",
            "go ahead",
            "do it",
            "confirm",
            "confirmed",
        }
        return user_query.strip().lower() in continuation_tokens

    @staticmethod
    def _default_analysis_prompt() -> str:
        """Default next-step analysis after data prep completes."""
        return (
            "Use the prepared analysis dataset to run a 1-way dimensional sweep, "
            "rank cohorts by AE_Ratio_Amount, and summarize the most adverse cohort "
            "with count and amount confidence intervals."
        )

    @staticmethod
    def _default_visualization_prompt(data_path: str) -> str:
        """Default next-step visualization after an analysis run completes."""
        return f"Create a scatter report using the amount metric from this sweep summary CSV: {data_path}"

    @staticmethod
    def _extract_csv_path(user_query: str) -> Optional[str]:
        """Extract an explicit CSV path from the user's request when present."""
        path_match = re.search(r"((?:/|data/)[\w./-]+\.csv)", user_query)
        if not path_match:
            return None
        return path_match.group(1)

    @staticmethod
    def _has_visualization_intent(query: str) -> bool:
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
        return any(k in query for k in visualize_hits)

    @staticmethod
    def _has_analysis_intent(query: str) -> bool:
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
        return any(k in query for k in analysis_hits)

    @staticmethod
    def _has_feature_engineering_intent(query: str) -> bool:
        """Detect explicit requests to engineer columns rather than reference existing band columns."""
        feature_patterns = [
            r"\bgroup\b.*\binto\b.*\bbands?\b",
            r"\bcreate\b.*\bbands?\b",
            r"\bregroup\b",
            r"\bbucket\b",
            r"\bband\b.*\bcolumn\b",
            r"\bfeature engineering\b",
        ]
        return any(re.search(pattern, query) for pattern in feature_patterns)

    @classmethod
    def _has_data_prep_intent(cls, query: str) -> bool:
        prep_hits = [
            "read",
            "profile",
            "validate",
            "check the data",
            "check data",
            "check the dataset",
            "clean",
            "prepare",
            "missing values",
            "dataset",
            "inforce",
        ]
        return any(k in query for k in prep_hits) or cls._has_feature_engineering_intent(query)

    @staticmethod
    def _has_fresh_analysis_artifact(path: Optional[str]) -> bool:
        """Return True only when the current-session sweep artifact exists on disk."""
        return bool(path and Path(path).exists())

    def _resolve_visualization_data_path(self, user_query: str) -> Optional[str]:
        """Choose the visualization input path, preferring explicit user paths then fresh session artifacts."""
        explicit_path = self._extract_csv_path(user_query)
        if explicit_path:
            return explicit_path
        if self._has_fresh_analysis_artifact(self.latest_analysis_output_path):
            return self.latest_analysis_output_path
        return None

    def _set_last_agent(self, agent_name: Literal["STEWARD", "ACTUARY", "ANALYST"]) -> None:
        """Track the last active specialist and clear stale pending work."""
        self.last_active_agent = agent_name
        if agent_name != "ACTUARY":
            self.pending_visualization_prompt = None

    def _route_to_steward(self, user_query: str) -> str:
        """Run data prep and queue a sensible follow-up analysis step."""
        response = self.data_steward.run(user_query)
        self._set_last_agent("STEWARD")
        self.pending_analysis_prompt = self._default_analysis_prompt()
        self.pending_visualization_prompt = None
        self.latest_analysis_output_path = None
        return response

    def _route_to_actuary(self, user_query: str) -> str:
        """Run analysis and queue a sensible follow-up visualization step."""
        response = self.actuary.run(user_query)
        self._set_last_agent("ACTUARY")
        self.pending_analysis_prompt = None
        if self._has_fresh_analysis_artifact(self.actuary.latest_output_path):
            self.latest_analysis_output_path = self.actuary.latest_output_path
            self.pending_visualization_prompt = self._default_visualization_prompt(
                self.latest_analysis_output_path
            )
        else:
            self.latest_analysis_output_path = None
            self.pending_visualization_prompt = None
        return response

    def _route_to_analyst(self, user_query: str) -> str:
        """Run visualization and clear pending visualization follow-ups."""
        data_path = self._resolve_visualization_data_path(user_query)
        if not data_path:
            return (
                "I do not have a fresh sweep artifact for this session. "
                "Run an actuarial sweep first, or provide an explicit sweep summary CSV path."
            )

        response = self.analyst_agent.run(user_query, data_path=data_path)
        self._set_last_agent("ANALYST")
        self.pending_analysis_prompt = None
        self.pending_visualization_prompt = None
        return response

    def _handle_continue(self) -> str:
        """Continue the next pending step using instance-level state."""
        if self.pending_analysis_prompt:
            prompt = self.pending_analysis_prompt
            self.pending_analysis_prompt = None
            return self._route_to_actuary(prompt)

        if self.pending_visualization_prompt:
            prompt = self.pending_visualization_prompt
            self.pending_visualization_prompt = None
            return self._route_to_analyst(prompt)

        if self.last_active_agent == "STEWARD":
            return self._route_to_actuary(self._default_analysis_prompt())
        if self.last_active_agent == "ACTUARY":
            if self._has_fresh_analysis_artifact(self.latest_analysis_output_path):
                return self._route_to_analyst(
                    self._default_visualization_prompt(self.latest_analysis_output_path)
                )
            return (
                "The last analysis did not produce a fresh sweep artifact to visualize. "
                "Run a new actuarial sweep first."
            )
        if self.last_active_agent == "ANALYST":
            return "The last step was already a visualization. Ask for a new analysis or chart."
        return (
            "There is no pending step to continue. Ask for data prep, analysis, or visualization."
        )

    def _heuristic_classify(self, user_query: str) -> Intent:
        """Fast deterministic fallback classifier when API key is unavailable."""
        q = user_query.lower()
        if self._is_continue_message(user_query):
            return "CONTINUE"

        has_visualize = self._has_visualization_intent(q)
        has_analysis = self._has_analysis_intent(q)
        has_prep = self._has_data_prep_intent(q)

        if has_visualize:
            return "VISUALIZE"
        if has_analysis and not has_prep:
            return "ANALYSIS"
        if has_prep and has_analysis:
            # NO CHAINING: classify only the first logical step.
            return "DATA_PREP"
        if has_prep:
            return "DATA_PREP"
        return "GENERAL"

    def _classify_intent(self, user_query: str) -> Intent:
        """Classify request into GENERAL, DATA_PREP, ANALYSIS, VISUALIZE, or CONTINUE."""
        if self._is_continue_message(user_query):
            return "CONTINUE"

        if not self.client:
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
                heuristic_intent = self._heuristic_classify(user_query)
                if intent == "DATA_PREP" and heuristic_intent == "ANALYSIS":
                    return heuristic_intent
                return intent  # type: ignore[return-value]
        except json.JSONDecodeError as exc:
            # Keep conversation flowing even if classifier returns malformed JSON.
            print(f"[Orchestrator Debug] JSON parsing failed: {exc}. Falling back to GENERAL.")
            return "GENERAL"

        return self._heuristic_classify(user_query)

    def process_query(self, user_query: str) -> str:
        """Route a query to the appropriate specialized agent(s)."""
        intent = self._classify_intent(user_query)

        if intent == "GENERAL":
            # Secondary guardrail routing for ambiguous natural phrasing.
            q = user_query.lower()
            if self._has_analysis_intent(q) and not self._has_data_prep_intent(q):
                return self._route_to_actuary(user_query)
            if self._has_data_prep_intent(q):
                return self._route_to_steward(user_query)
            if self._has_visualization_intent(q):
                return self._route_to_analyst(user_query)
            return (
                "I can help with data preparation, actuarial analysis, or visualization. "
                "Share what you want to do, and I will route it."
            )

        if intent == "DATA_PREP":
            return self._route_to_steward(user_query)

        if intent == "ANALYSIS":
            return self._route_to_actuary(user_query)

        if intent == "VISUALIZE":
            return self._route_to_analyst(user_query)

        if intent == "CONTINUE":
            return self._handle_continue()

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
