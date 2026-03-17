"""Data Steward Agent: profiles, validates, and engineers inforce data."""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.schemas import ProfileDatasetSchema, CategoricalBandingSchema

from tools.data_steward import (
    create_categorical_bands,
    profile_dataset,
    run_actuarial_data_checks,
)


SYSTEM_PROMPT = """
You are a meticulous Data Engineer and Actuarial Data Steward.

Your job is to profile datasets to ensure actuarial integrity, including MAC/MEC/MOC logic,
and perform feature engineering (for example, age or face-amount banding) so actuaries can run
multi-dimensional sweeps.

Use tools whenever possible. Be technical, concise, and helpful.
Always confirm what data transformations were applied and where output was saved.

CRITICAL GUARDRAILS: 
1. Immutable Input: You may read from `data/input/`, but you must save all outputs strictly to `data/output/analysis_inforce.csv`.
2. Null Preservation: You shall NEVER automatically drop rows with Null/NaN values, nor impute them, unless explicitly instructed by the user. 
3. Domain Rule: In life insurance, a Null in the 'COLA' column is expected if there is no claim. Do not flag this as an error.
""".strip()


class DataStewardAgent:
    """Agent wrapper that routes user requests to deterministic steward tools."""

    def __init__(self, model: str = "gpt-5-mini") -> None:
        self.model = model
        self.client: OpenAI = client

        self.tool_handlers: Dict[str, Callable[..., str]] = {
            "profile_dataset": profile_dataset,
            "run_actuarial_data_checks": run_actuarial_data_checks,
            "create_categorical_bands": create_categorical_bands,
        }

    def _tools_spec(self) -> list[dict[str, Any]]:
        """Build OpenAI tool specs from Pydantic schemas."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "profile_dataset",
                    "description": "Profile inforce dataset structure and nulls.",
                    "parameters": ProfileDatasetSchema.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_actuarial_data_checks",
                    "description": "Run actuarial data quality checks (MAC, MEC, MOC, COLA, exposure logic).",
                    "parameters": ProfileDatasetSchema.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_categorical_bands",
                    "description": "Create categorical bands for numeric columns and save to analysis dataset.",
                    "parameters": CategoricalBandingSchema.model_json_schema(),
                },
            },
        ]

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        if tool_name not in self.tool_handlers:
            return json.dumps({"error": f"Unknown tool: {tool_name}"}, indent=2)

        if tool_name == "create_categorical_bands" and "operation" in args:
            # If FeatureEngineeringSchema is used as alias, strip non-function key.
            args = {k: v for k, v in args.items() if k != "operation"}
        if tool_name == "create_categorical_bands" and "data_path" in args and "source_path" not in args:
            # Schema alias compatibility: data_path maps to source_path.
            args["source_path"] = args.pop("data_path")
        elif "data_path" in args:
            args.pop("data_path")

        try:
            return self.tool_handlers[tool_name](**args)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            return json.dumps({"error": f"{tool_name} failed: {str(exc)}"}, indent=2)

    def _fallback_route(self, user_message: str) -> str:
        """Deterministic fallback when no API key is configured."""
        msg = user_message.lower()

        if "profile" in msg or "moc" in msg:
            data_path = "data/input/synthetic_inforce.csv"
            path_match = re.search(r"(data/[\w./-]+\.csv)", user_message)
            if path_match:
                data_path = path_match.group(1)

            profile_json = profile_dataset(data_path=data_path)
            checks_json = run_actuarial_data_checks(data_path=data_path)
            checks = json.loads(checks_json)
            status = checks.get("status", "UNKNOWN")
            return (
                f"Profile complete for `{data_path}`.\n"
                f"- Validation status: **{status}**\n"
                f"- MOC check included: numeric type, range (0,1], and MAC==1 => MOC==1.0\n"
                f"- Profile JSON:\n{profile_json}\n\n"
                f"- Data checks JSON:\n{checks_json}"
            )

        if "band" in msg:
            bins = 5 if "5" in msg else 4
            col_match = re.search(r"for the\s+([A-Za-z_][A-Za-z0-9_]*)\s+column", user_message, re.IGNORECASE)
            source_column = col_match.group(1) if col_match else "Issue_Age"
            source_path = "data/input/synthetic_inforce.csv"
            if not Path(source_path).exists():
                source_path = "data/input/synthetic_inforce.csv"
            result = create_categorical_bands(
                source_column=source_column,
                strategy="equal_width",
                bins=bins,
                source_path=source_path,
                output_path="data/output/analysis_inforce.csv",
            )
            return (
                f"Banding complete for `{source_column}` using equal-width `{bins}` bins.\n"
                f"Source data: `{source_path}`.\n"
                f"Transformation saved to `data/output/analysis_inforce.csv`.\n"
                f"Tool JSON:\n{result}"
            )

        return (
            "I can help with profiling, actuarial data checks, and categorical banding. "
            "Please specify a dataset path and transformation request."
        )

    def run(self, user_message: str) -> str:
        """Handle a user message using OpenAI tool-calling."""
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
            except Exception:
                # Keep deterministic local usability when network/proxy is unavailable.
                return self._fallback_route(user_message)
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

        return "Unable to complete request within tool-calling loop."

    def respond(self, user_message: str) -> str:
        """Backward-compatible alias."""
        return self.run(user_message)


if __name__ == "__main__":
    agent = DataStewardAgent()

    msg_1 = (
        "Please profile our synthetic inforce dataset located at "
        "data/input/synthetic_inforce.csv. Let me know if the MOC column looks correct."
    )
    msg_2 = "Create 5 equal-width bands for the Issue_Age column."

    print("=== Test Case 1 (Profiling) ===")
    print(agent.run(msg_1))
    print("\n=== Test Case 2 (Banding) ===")
    print(agent.run(msg_2))
