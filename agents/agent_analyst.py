"""Actuarial Data Analyst Agent: generates visual A/E reports."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model
        self.client: OpenAI = client

        self.tool_handlers: Dict[str, Callable[..., str]] = {
            "generate_univariate_report": generate_univariate_report,
            "generate_treemap_report": generate_treemap_report,
        }

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
        data_path = args.get("data_path", "data/sweep_summary.csv")

        try:
            if tool_name == "generate_univariate_report":
                return self.tool_handlers[tool_name](data_path=data_path, metric=metric)
            if tool_name == "generate_treemap_report":
                return self.tool_handlers[tool_name](data_path=data_path, metric=metric)
            return json.dumps({"error": f"No handler implemented for {tool_name}"}, indent=2)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            return json.dumps({"error": f"{tool_name} failed: {str(exc)}"}, indent=2)

    def run(self, user_message: str) -> str:
        """Handle visualization requests using OpenAI tool calling."""
        if not os.getenv("OPENAI_API_KEY"):
            return "OPENAI_API_KEY is missing. Add it to .env before running this agent."

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
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
                # If network/proxy blocks OpenAI, surface a clear diagnostic.
                return f"Unable to contact OpenAI API for visualization: {exc}"

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

