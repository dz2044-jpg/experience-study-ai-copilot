"""Lead Actuary Agent: interprets dimensional sweep A/E results."""

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
2. NEVER set `depth=3` or higher just because the user listed 3 columns. Only set `depth=3` when the user explicitly requests a 3-way interaction term.

STRICT ROLE BOUNDARIES & DATA SOURCES:
1. Your ONLY source of truth for dimensional sweeps is `data/output/analysis_inforce.csv`. You must assume the Data Steward has already prepared this file.
2. NO MOONLIGHTING: You are the Lead Actuary, not the Data Steward. You must NEVER attempt to create bands, clean data, or impute missing values.
3. MISSING COLUMNS: If the user requests a sweep on a column (for example, `Face_Amount_band`) that is missing from `data/output/analysis_inforce.csv`, do not attempt feature engineering. Clearly instruct the user to run Data Steward first to create that feature, then rerun the sweep.
""".strip()


class ActuaryAgent:
    """Agent wrapper that routes actuarial questions to dimensional sweep tooling."""

    DEFAULT_ANALYSIS_PATH = "data/output/analysis_inforce.csv"

    def __init__(self, model: str = "gpt-5.4") -> None:
        self.model = model
        self.client = build_openai_client()
        self.latest_output_path: Optional[str] = None
        self.latest_output_alias_path: Optional[str] = None

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

    @staticmethod
    def _format_numeric(value: Any, decimals: int = 2) -> str:
        """Format numeric values for markdown output."""
        if value is None:
            return "N/A"
        return f"{float(value):.{decimals}f}"

    @staticmethod
    def _format_int(value: Any) -> str:
        """Format integer-like values for markdown output."""
        if value is None:
            return "N/A"
        return str(int(value))

    @staticmethod
    def _escape_markdown_cell(value: Any) -> str:
        """Escape pipes so cohort labels render correctly inside markdown tables."""
        return str(value).replace("|", "\\|")

    def _build_ranked_cohort_table(self, rows: list[dict[str, Any]]) -> str:
        """Render the top-ranked cohorts as a markdown table."""
        table_rows = rows[:10]
        header = [
            "| Rank | Cohort | AE_Ratio_Amount | AE_Ratio_Count | Sum_MAC |",
            "| --- | --- | --- | --- | --- |",
        ]

        for idx, row in enumerate(table_rows, start=1):
            header.append(
                "| "
                f"{idx} | "
                f"{self._escape_markdown_cell(row.get('Dimensions', 'N/A'))} | "
                f"{self._format_numeric(row.get('AE_Ratio_Amount'))} | "
                f"{self._format_numeric(row.get('AE_Ratio_Count'))} | "
                f"{self._format_int(row.get('Sum_MAC'))} |"
            )

        return "\n".join(header)

    @staticmethod
    def _top_cohort_label(sort_by: str) -> str:
        """Return metric-aware wording for the detailed summary block."""
        if sort_by == "AE_Ratio_Amount":
            return "Worst cohort"
        return f"Top-ranked cohort by {sort_by}"

    def _reset_sweep_artifacts(self) -> None:
        """Clear any previously recorded sweep output paths before a new run."""
        self.latest_output_path = None
        self.latest_output_alias_path = None

    def _record_sweep_artifacts(self, result_json: str) -> None:
        """Capture deterministic sweep artifact paths from tool JSON."""
        try:
            payload = json.loads(result_json)
        except json.JSONDecodeError:
            return

        if payload.get("error"):
            return

        output_path = payload.get("output_path")
        latest_alias_path = payload.get("latest_output_path")
        if isinstance(output_path, str) and output_path.strip():
            self.latest_output_path = output_path
        if isinstance(latest_alias_path, str) and latest_alias_path.strip():
            self.latest_output_alias_path = latest_alias_path

    @staticmethod
    def _normalize_column_name(name: str) -> str:
        """Normalize column names for case-insensitive matching."""
        normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
        return re.sub(r"_+", "_", normalized).strip("_")

    @classmethod
    def _is_explicit_sweep_request(cls, user_message: str) -> bool:
        """Identify requests that should deterministically execute the sweep tool."""
        msg = user_message.lower()
        if "sweep" in msg:
            return True
        return "a/e ratio" in msg and any(token in msg for token in ("run", "calculate", "show"))

    @classmethod
    def _extract_depth(cls, user_message: str) -> int:
        """Infer requested sweep depth from the prompt."""
        msg = user_message.lower()
        depth_match = re.search(r"\b([123])[- ]way\b", msg)
        if depth_match:
            return int(depth_match.group(1))
        if "pairwise" in msg or "all pairs" in msg:
            return 2
        return 1

    @staticmethod
    def _extract_top_n(user_message: str) -> Optional[int]:
        """Read a top-N hint when the user asked for a ranked subset."""
        msg = user_message.lower()
        if any(
            phrase in msg
            for phrase in (
                "all cohorts",
                "all rows",
                "full result set",
                "full results",
                "return all",
                "top_n=all",
            )
        ):
            return None
        top_match = re.search(r"\btop\s+(\d+)\b", user_message.lower())
        if top_match:
            return int(top_match.group(1))
        return 20

    @staticmethod
    def _extract_min_mac(user_message: str) -> int:
        """Read an explicit visibility floor from the prompt."""
        msg = user_message.lower()
        min_match = re.search(r"min_mac\s*=\s*(\d+)", msg)
        if min_match:
            return int(min_match.group(1))

        at_least_match = re.search(r"at least\s+(\d+)\s+deaths?", msg)
        if at_least_match:
            return int(at_least_match.group(1))
        return 1

    @staticmethod
    def _extract_sort_by(user_message: str) -> str:
        """Map ranking phrases to supported sort columns."""
        msg = user_message.lower()
        explicit_match = re.search(r"\b(ae_ratio_amount|ae_ratio_count|sum_mac|sum_moc|sum_mec|sum_maf|sum_mef)\b", msg)
        if explicit_match:
            explicit_map = {
                "ae_ratio_amount": "AE_Ratio_Amount",
                "ae_ratio_count": "AE_Ratio_Count",
                "sum_mac": "Sum_MAC",
                "sum_moc": "Sum_MOC",
                "sum_mec": "Sum_MEC",
                "sum_maf": "Sum_MAF",
                "sum_mef": "Sum_MEF",
            }
            return explicit_map[explicit_match.group(1)]

        if "rank" in msg or "sort" in msg:
            if "count" in msg:
                return "AE_Ratio_Count"
            if "amount" in msg:
                return "AE_Ratio_Amount"
        return "AE_Ratio_Amount"

    @classmethod
    def _load_analysis_columns(cls, data_path: str) -> Optional[list[str]]:
        """Read available columns from the prepared analysis dataset."""
        path = Path(data_path)
        if not path.exists():
            return None
        df = pd.read_csv(path, nrows=0)
        return list(df.columns)

    @classmethod
    def _extract_requested_columns(cls, user_message: str, available_columns: list[str]) -> tuple[Optional[list[str]], list[str]]:
        """Parse explicit requested dimensions from phrases like 'on X, Y' or 'between A, B, and C'."""
        patterns = [
            r"\bbetween\s+(.+?)(?:,?\s+then\b|,?\s+rank\b|,?\s+sort\b|,?\s+using\b|,?\s+with\b|,?\s+where\b|,?\s+to\b|[?.]|$)",
            r"\bacross\s+(.+?)(?:,?\s+then\b|,?\s+rank\b|,?\s+sort\b|,?\s+using\b|,?\s+with\b|,?\s+where\b|,?\s+to\b|[?.]|$)",
            r"\bon\s+(.+?)(?:,?\s+then\b|,?\s+rank\b|,?\s+sort\b|,?\s+using\b|,?\s+with\b|,?\s+where\b|,?\s+to\b|[?.]|$)",
        ]
        requested_segment: Optional[str] = None
        for pattern in patterns:
            match = re.search(pattern, user_message, flags=re.IGNORECASE)
            if match:
                requested_segment = match.group(1)
                break

        if not requested_segment:
            return (None, [])

        cleaned_segment = requested_segment.replace("×", ",").replace(" x ", ", ")
        cleaned_segment = re.sub(r"\bfor all pairs\b", "", cleaned_segment, flags=re.IGNORECASE)
        cleaned_segment = re.sub(r"\ball pairs\b", "", cleaned_segment, flags=re.IGNORECASE)
        tokens = [
            token.strip(" `.")
            for token in re.split(r",|\band\b|&", cleaned_segment, flags=re.IGNORECASE)
            if token.strip(" `.")
        ]

        column_lookup = {cls._normalize_column_name(col): col for col in available_columns}
        matched: list[str] = []
        missing: list[str] = []
        for token in tokens:
            normalized = cls._normalize_column_name(token)
            if not normalized:
                continue
            resolved = column_lookup.get(normalized)
            if resolved:
                if resolved not in matched:
                    matched.append(resolved)
            else:
                missing.append(token)

        return (matched or None, missing)

    def _deterministic_sweep_route(self, user_message: str) -> Optional[str]:
        """Run explicit sweep requests locally so the model cannot ask for known A/E mappings."""
        if not self._is_explicit_sweep_request(user_message):
            return None

        available_columns = self._load_analysis_columns(self.DEFAULT_ANALYSIS_PATH)
        if available_columns is None:
            return (
                "Unable to complete sweep: `data/output/analysis_inforce.csv` is missing. "
                "Run Data Steward first to prepare the analysis dataset, then rerun the sweep."
            )

        selected_columns, missing_columns = self._extract_requested_columns(user_message, available_columns)
        if missing_columns:
            missing_list = ", ".join(f"`{col}`" for col in missing_columns)
            return (
                f"Unable to complete sweep: requested column(s) not found in `{self.DEFAULT_ANALYSIS_PATH}`: {missing_list}. "
                "Run Data Steward first to create those features, then rerun the sweep."
            )

        depth = self._extract_depth(user_message)
        sort_by = self._extract_sort_by(user_message)
        min_mac = self._extract_min_mac(user_message)
        top_n = self._extract_top_n(user_message)

        sweep = run_dimensional_sweep(
            depth=depth,
            selected_columns=selected_columns,
            min_mac=min_mac,
            top_n=top_n,
            sort_by=sort_by,
            data_path=self.DEFAULT_ANALYSIS_PATH,
        )

        selected_label = ", ".join(selected_columns) if selected_columns else "auto-detected dimensions"
        context = (
            f"{depth}-way dimensional sweep complete on the prepared analysis dataset "
            f"using {selected_label}, ranked by {sort_by}."
        )
        return self._summarize_sweep(sweep, context, sort_by)

    def _summarize_sweep(self, result_json: str, context: str, sort_by: str = "AE_Ratio_Amount") -> str:
        self._record_sweep_artifacts(result_json)
        payload = json.loads(result_json)
        if "error" in payload:
            return f"Unable to complete sweep: {payload['error']}"
        rows = payload.get("results", [])
        if not rows:
            return "No cohorts met the requested visibility threshold."

        top = rows[0]
        ranked_table = self._build_ranked_cohort_table(rows)
        top_label = self._top_cohort_label(sort_by)
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
            f"\nTop 10 ranked cohorts\n"
            f"{ranked_table}\n"
            f"\n"
            f"- {top_label}: `{top['Dimensions']}`\n"
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
                "AE_Ratio_Amount",
            )

        if "1-way" in msg or "most adverse cohort" in msg or "rank cohorts by ae_ratio_amount" in msg:
            sweep = run_dimensional_sweep(
                depth=1,
                min_mac=1,
                top_n=5,
                sort_by="AE_Ratio_Amount",
            )
            return self._summarize_sweep(
                sweep,
                "High-level 1-way sweep complete on the prepared analysis dataset.",
                "AE_Ratio_Amount",
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
                "AE_Ratio_Amount",
            )

        return (
            "I can run 1-way/2-way/3-way A/E sweeps and interpret count-vs-amount risk. "
            "Please specify sweep depth, min_mac, and preferred ranking metric."
        )

    def run(self, user_message: str) -> str:
        """Handle message with OpenAI tool-calling."""
        self._reset_sweep_artifacts()

        deterministic_response = self._deterministic_sweep_route(user_message)
        if deterministic_response is not None:
            return deterministic_response

        if not self.client:
            return self._fallback_route(user_message)

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
                if name == "run_dimensional_sweep":
                    self._record_sweep_artifacts(tool_result)
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
