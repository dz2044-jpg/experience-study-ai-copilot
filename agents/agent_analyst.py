"""Actuarial Data Analyst Agent: generates visual A/E reports."""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.openai_compat import build_openai_client
from agents.schemas import VisualizationSchema
from tools.visualization import generate_treemap_report, generate_univariate_report


SYSTEM_PROMPT = """
You are an Expert Actuarial Data Analyst.

Your job is to take aggregated mortality A/E data and generate interactive visual
reports for the Lead Actuary.

When asked to visualize data, decide whether the user needs a standard univariate
view (scatter plot with data table) or a hierarchical risk view (treemap heatmap),
then call the appropriate visualization tool.

Be concise and professional. After running a tool, just confirm which chart was
generated, which metric was used, and that it opened in the user's browser.
""".strip()


class AnalystAgent:
    """Agent that turns aggregated A/E data into interactive visual reports."""

    def __init__(self, model: str = "gpt-5.3-codex") -> None:
        self.model = model
        self.client = build_openai_client()
        self.last_data_path_used: Optional[str] = None

        self.tool_handlers: Dict[str, Callable[..., str]] = {
            "generate_univariate_report": generate_univariate_report,
            "generate_treemap_report": generate_treemap_report,
        }

    @staticmethod
    def _normalize_dimension_name(name: str) -> str:
        """Normalize user and dataset dimension names for matching."""
        normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
        return re.sub(r"_+", "_", normalized).strip("_")

    @classmethod
    def _extract_requested_dimensions(cls, user_message: str) -> Optional[list[str]]:
        """Parse an explicit treemap slice request such as 'on Gender and Smoker'."""
        patterns = [
            r"\bon\s+(.+?)(?:[?.]|$)",
            r"\bfor\s+(.+?)(?:[?.]|$)",
        ]
        requested_segment: Optional[str] = None
        for pattern in patterns:
            match = re.search(pattern, user_message, flags=re.IGNORECASE)
            if match:
                requested_segment = match.group(1)
                break

        if not requested_segment:
            return None

        cleaned = requested_segment
        for phrase in (
            "the 2-way sweep",
            "2-way sweep",
            "pairwise sweep",
            "latest sweep summary",
            "latest sweep",
            "sweep summary",
            "treemap",
            "heatmap",
        ):
            cleaned = re.sub(rf"\b{re.escape(phrase)}\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("×", " and ").replace(" x ", " and ")

        tokens = [
            token.strip(" `.")
            for token in re.split(r",|\band\b|&", cleaned, flags=re.IGNORECASE)
            if token.strip(" `.")
        ]
        if len(tokens) < 2:
            return None
        return [cls._normalize_dimension_name(token) for token in tokens[:2]]

    @classmethod
    def _filter_treemap_source_for_dimensions(cls, data_path: str, requested_dimensions: list[str]) -> tuple[Optional[str], Optional[str]]:
        """
        Filter a pairwise sweep artifact to a requested dimension pair.

        Returns (filtered_path, error_message). When no explicit pair was requested,
        the original path is returned unchanged.
        """
        if not requested_dimensions:
            return (data_path, None)

        df = pd.read_csv(data_path)
        if "Dimensions" not in df.columns:
            return (None, f"Visualization source is missing the required `Dimensions` column: {data_path}")

        dimension_pairs: list[tuple[str, str]] = []
        mask = []
        requested_pair = tuple(sorted(requested_dimensions))
        for label in df["Dimensions"].astype(str):
            parts = [part.split("=")[0] for part in label.split(" | ") if "=" in part]
            normalized_parts = tuple(sorted(cls._normalize_dimension_name(part) for part in parts))
            dimension_pairs.append(normalized_parts)
            mask.append(normalized_parts == requested_pair)

        if not any(mask):
            available_pairs = sorted({pair for pair in dimension_pairs if len(pair) == 2})
            formatted_pairs = ", ".join(" + ".join(pair) for pair in available_pairs) if available_pairs else "none"
            requested_label = " + ".join(requested_dimensions)
            return (
                None,
                f"Unable to generate treemap: the current sweep artifact does not contain the requested pair `{requested_label}`. "
                f"Available pairs in `{data_path}`: {formatted_pairs}.",
            )

        filtered_df = df[mask].copy()
        requested_slug = "_".join(requested_dimensions)
        filtered_path = Path("data/output") / f"temp_treemap_source_{requested_slug}.csv"
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_csv(filtered_path, index=False)
        return (str(filtered_path), None)

    def _tools_spec(self) -> list[dict[str, Any]]:
        """Define visualization tools for OpenAI function calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "generate_univariate_report",
                    "description": "Generate a univariate A/E scatter plot and data table report.",
                    "parameters": VisualizationSchema.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_treemap_report",
                    "description": "Generate a hierarchical treemap heatmap of A/E risk.",
                    "parameters": VisualizationSchema.model_json_schema(),
                },
            },
        ]

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        if tool_name not in self.tool_handlers:
            return json.dumps({"error": f"Unknown tool: {tool_name}"}, indent=2)

        # Map visualization schema fields to underlying function signatures.
        metric = args.get("metric", "amount")
        data_path = args.get("data_path", "data/output/sweep_summary.csv")
        self.last_data_path_used = data_path

        try:
            if tool_name == "generate_univariate_report":
                return self.tool_handlers[tool_name](data_path=data_path, metric=metric)
            if tool_name == "generate_treemap_report":
                return self.tool_handlers[tool_name](data_path=data_path, metric=metric)
            return json.dumps({"error": f"No handler implemented for {tool_name}"}, indent=2)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            return json.dumps({"error": f"{tool_name} failed: {str(exc)}"}, indent=2)

    @staticmethod
    def _extract_csv_path(user_message: str) -> Optional[str]:
        """Extract an explicit CSV path from the request when provided."""
        path_match = re.search(r"((?:/|data/)[\w./-]+\.csv)", user_message)
        if not path_match:
            return None
        return path_match.group(1)

    def _fallback_route(self, user_message: str, data_path: Optional[str] = None) -> str:
        """Deterministic fallback when no API key is configured."""
        msg = user_message.lower()
        metric = "count" if "count" in msg else "amount"
        resolved_data_path = data_path or self._extract_csv_path(user_message) or "data/output/sweep_summary.csv"
        self.last_data_path_used = resolved_data_path

        if "treemap" in msg or "heatmap" in msg:
            requested_dimensions = self._extract_requested_dimensions(user_message) or []
            filtered_path, error_message = self._filter_treemap_source_for_dimensions(
                resolved_data_path,
                requested_dimensions,
            )
            if error_message:
                return error_message
            try:
                self.last_data_path_used = filtered_path
                return generate_treemap_report(data_path=filtered_path or resolved_data_path, metric=metric)
            except Exception as exc:
                return f"Unable to generate treemap report: {exc}"

        try:
            return generate_univariate_report(data_path=resolved_data_path, metric=metric)
        except Exception as exc:
            return f"Unable to generate univariate report: {exc}"

    def run(self, user_message: str, data_path: Optional[str] = None) -> str:
        """Handle visualization requests using OpenAI tool calling."""
        self.last_data_path_used = data_path

        if "treemap" in user_message.lower() or "heatmap" in user_message.lower():
            return self._fallback_route(user_message, data_path=data_path)

        if not self.client:
            return self._fallback_route(user_message, data_path=data_path)

        resolved_data_path = data_path or self._extract_csv_path(user_message)
        effective_message = user_message
        if resolved_data_path:
            effective_message = (
                f"{user_message}\n\n"
                f"Use this exact visualization input CSV path: {resolved_data_path}"
            )
            self.last_data_path_used = resolved_data_path

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": effective_message},
        ]

        for _ in range(4):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self._tools_spec(),
                    tool_choice="auto",
                )
            except Exception as exc:
                # Keep deterministic local usability when network/proxy is unavailable.
                return self._fallback_route(user_message, data_path=resolved_data_path)

            message = completion.choices[0].message
            tool_calls = message.tool_calls or []

            if not tool_calls:
                return message.content or "No response generated."

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                }
            )

            # Execute tools and feed results back for final natural-language summary.
            for tool_call in tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                tool_result = self._execute_tool(name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

        return "Unable to complete visualization request within tool-calling loop."


if __name__ == "__main__":
    agent = AnalystAgent()

    msg = "Please create a Treemap heatmap based on the A/E Amount metric for our recent sweep."

    print("=== AnalystAgent Visualization Test ===")
    print(agent.run(msg))
