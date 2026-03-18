"""Lead Actuary Agent: interprets dimensional sweep A/E results."""

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

from agents.schemas import DimensionalSweepSchema
from tools.insight_engine import run_dimensional_sweep


SYSTEM_PROMPT = """
You are a Lead Valuation Actuary with 20 years of experience in mortality experience studies.

Analyze A/E (Actual-to-Expected) outputs rigorously. Always compare AE_Ratio_Amount
(financial risk) against AE_Ratio_Count (selection or mortality-rate risk).

Be analytical, precise, and cautious. Never call a trend "certain" when the 95%
confidence interval is very wide or spans 1.0.

When summarizing results, use these exact definitions: MAC = Actual Death Count. MOC = Total Policy Exposure (Years). MEC = Expected Death Count. MAF = Actual Claim Amount ($). MEF = Expected Claim Amount ($).

CRITICAL GUARDRAILS:
1. Wait for Command: Do not assume what analysis to run. Only execute the specific sweep the user requested.
2. No LLM Arithmetic: You must NEVER calculate A/E ratios, Confidence Intervals, or exposures using your own internal logic. You must strictly use the Python tools provided and read the results from `data/output/sweep_summary.csv`.

ACTUARIAL DATA DICTIONARY (CRITICAL):
You must NEVER ask the user to clarify these columns. Treat these definitions as absolute truth:
- MAC = Actual Death Count
- MOC = Total Policy Exposure (Years)
- MEC = Expected Death Count
- MAF = Actual Claim Amount ($)
- MEF = Expected Claim Amount ($)
- A/E Ratio by Count = sum(MAC) / sum(MEC)
- A/E Ratio by Amount = sum(MAF) / sum(MEF)

TOOL USAGE INSTRUCTIONS (DIMENSIONAL SWEEPS):
When using the `run_dimensional_sweep` tool, you must strictly map the user's request to the correct `depth` parameter:
1. Pairwise / 2-Way Sweeps on Multiple Columns: If the user asks for a "2-way sweep" or "pairwise sweep" across 3 or more columns (e.g., A, B, and C), you MUST set `depth=2` and pass all the requested columns into `selected_columns=['A', 'B', 'C']`. The tool will automatically handle calculating the pairs (A×B, A×C, B×C). 
2. NEVER set `depth=3` or higher just because the user listed 3 columns. Only se
""".strip()


class ActuaryAgent:
    """Agent wrapper that routes actuarial questions to dimensional sweep tooling."""

    def __init__(self, model: str = "gpt-5.4") -> None:
        self.model = model
        self.client: OpenAI = client

        self.tool_handlers: Dict[str, Callable[..., str]] = {
            "run_dimensional_sweep": run_dimensional_sweep,
        }

    def _tools_spec(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_dimensional_sweep",
                    "description": (
                        "Run actuarial dimensional sweep and return top cohort intersections "
                        "with count/amount A/E ratios and Bayesian confidence intervals."
                    ),
                    "parameters": DimensionalSweepSchema.model_json_schema(),
                },
            }
        ]

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        if tool_name not in self.tool_handlers:
            return json.dumps({"error": f"Unknown tool: {tool_name}"}, indent=2)
        try:
            return self.tool_handlers[tool_name](**args)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            return json.dumps({"error": f"{tool_name} failed: {str(exc)}"}, indent=2)

    @staticmethod
    def _format_ci(ci: list[Any]) -> str:
        if not ci or ci[0] is None or ci[1] is None:
            return "N/A"
        return f"[{ci[0]:.2f}, {ci[1]:.2f}]"

    def _summarize_sweep(self, result_json: str, context: str) -> str:
        payload = json.loads(result_json)
        if "error" in payload:
            return f"Unable to complete sweep: {payload['error']}"
        rows = payload.get("results", [])
        if not rows:
            return "No cohorts met the requested visibility threshold."

        top = rows[0]
        count_ci = top.get("AE_Count_CI", [None, None])
        amount_ci = top.get("AE_Amount_CI", [None, None])

        wide_count = count_ci[0] is not None and count_ci[1] is not None and (count_ci[1] - count_ci[0]) > 3.0
        wide_amount = amount_ci[0] is not None and amount_ci[1] is not None and (amount_ci[1] - amount_ci[0]) > 3.0
        crosses_one_count = count_ci[0] is not None and count_ci[1] is not None and count_ci[0] <= 1.0 <= count_ci[1]
        crosses_one_amount = amount_ci[0] is not None and amount_ci[1] is not None and amount_ci[0] <= 1.0 <= amount_ci[1]

        caution_note = "Signal appears directionally adverse."
        if wide_count or wide_amount or crosses_one_count or crosses_one_amount:
            caution_note = (
                "Signal is directionally adverse, but statistical certainty is limited "
                "(wide CI and/or interval crossing 1.0)."
            )

        return (
            f"{context}\n"
            f"- Top cohort: `{top['Dimensions']}`\n"
            f"- Financial risk (AE_Ratio_Amount): {top['AE_Ratio_Amount']:.2f}\n"
            f"- Selection/mortality risk (AE_Ratio_Count): {top['AE_Ratio_Count']:.2f}\n"
            f"- Count 95% CI: {self._format_ci(count_ci)}\n"
            f"- Amount 95% CI: {self._format_ci(amount_ci)}\n"
            f"- Actuarial interpretation: {caution_note}\n"
            f"- Cohorts evaluated in output: {len(rows)}"
        )

    def _fallback_route(self, user_message: str) -> str:
        """Deterministic fallback when no API key is configured."""
        msg = user_message.lower()

        if "single cohort" in msg and "face amount" in msg:
            sweep = run_dimensional_sweep(
                depth=1,
                selected_columns=["Face_Amount"],
                min_mac=1,
                top_n=1,
                sort_by="AE_Ratio_Amount",
            )
            return self._summarize_sweep(
                sweep,
                "High-level 1-way sweep complete (Face Amount focus).",
            )

        if "2-way" in msg or "min_mac=2" in msg or "intersections" in msg:
            min_mac = 2
            min_match = re.search(r"min_mac\s*=\s*(\d+)", msg)
            if min_match:
                min_mac = int(min_match.group(1))
            sweep = run_dimensional_sweep(
                depth=2,
                min_mac=min_mac,
                top_n=10,
                sort_by="AE_Ratio_Amount",
            )
            return self._summarize_sweep(
                sweep,
                f"2-way dimensional sweep complete with min_mac={min_mac}.",
            )

        return (
            "I can run 1-way/2-way/3-way A/E sweeps and interpret count-vs-amount risk. "
            "Please specify sweep depth, min_mac, and preferred ranking metric."
        )

    def run(self, user_message: str) -> str:
        """Handle message with OpenAI tool-calling."""
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
    agent = ActuaryAgent()

    msg_1 = "Please run a high-level sweep on the data. What is the worst-performing single cohort by Face Amount?"
    msg_2 = (
        "Run a 2-way dimensional sweep, but only show me cohorts with at least 2 deaths "
        "(min_mac=2). Are there any statistically significant intersections we should worry about?"
    )

    print("=== Test Case 1 (1-Way Sweep) ===")
    print(agent.run(msg_1))
    print("\n=== Test Case 2 (Deep Dive) ===")
    print(agent.run(msg_2))
