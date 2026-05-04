"""User-facing response formatting for deterministic copilot results."""

from __future__ import annotations

import re
from typing import Any

from core.session_state import SessionArtifactState


class ResponseFormatter:
    """Formats deterministic tool results without owning copilot orchestration."""

    _THINKING_BLOCK_RE = re.compile(r"<thinking>.*?</thinking>", re.IGNORECASE | re.DOTALL)

    def __init__(self, state: SessionArtifactState) -> None:
        self.state = state

    @classmethod
    def sanitize_user_facing_text(cls, text: str) -> str:
        if not text:
            return ""
        sanitized = cls._THINKING_BLOCK_RE.sub("", text)
        sanitized = re.sub(r"[ \t]+\n", "\n", sanitized)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized.strip()

    def format_schema_result(self, result: dict[str, Any]) -> str:
        data = result.get("data", {})
        source_path = data.get("source_path", "the dataset")
        columns = data.get("columns", [])
        data_types = data.get("data_types", {})
        if not columns:
            return result["message"]
        lines = [f"Columns in `{source_path}` ({len(columns)}):"]
        lines.extend(
            f"- `{column}`: `{data_types.get(column, 'unknown')}`" for column in columns
        )
        return "\n".join(lines)

    def format_profile_result(self, result: dict[str, Any]) -> str:
        data = result.get("data", {})
        artifacts = result.get("artifacts", {})
        source_path = artifacts.get("raw_input_path", "the source dataset")
        prepared_path = artifacts.get("prepared_dataset_path", "the prepared dataset")
        row_count = data.get("total_rows")
        column_count = len(data.get("columns", []))
        unique_policies = data.get("unique_policy_count")
        summary_bits = []
        if row_count is not None:
            summary_bits.append(f"{int(row_count):,} rows")
        if column_count:
            summary_bits.append(f"{column_count} columns")
        if unique_policies is not None:
            summary_bits.append(f"{int(unique_policies):,} unique policies")
        lines = [f"Created the prepared dataset `{prepared_path}` from `{source_path}`."]
        if summary_bits:
            lines.append(f"Profile summary: {', '.join(summary_bits)}.")
        return "\n".join(lines)

    @staticmethod
    def format_sweep_value(value: Any) -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    def analysis_summary_table(self, rows: list[dict[str, Any]]) -> str:
        headers = [
            "Cohort Dimension",
            "Actual Deaths (MAC)",
            "Expected (MEC)",
            "A/E Ratio (Count)",
            "A/E Ratio (Amount)",
        ]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            values = [
                str(row.get("Dimensions", "")),
                self.format_sweep_value(row.get("Sum_MAC")),
                self.format_sweep_value(row.get("Sum_MEC")),
                self.format_sweep_value(row.get("AE_Ratio_Count")),
                self.format_sweep_value(row.get("AE_Ratio_Amount")),
            ]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    def analysis_summary_sections(
        self,
        result: dict[str, Any],
        *,
        include_intro: bool,
    ) -> list[str]:
        lines: list[str] = []
        if include_intro:
            lines.append(result["message"])

        rows = result.get("data", {}).get("results", [])
        if not rows:
            return lines

        if lines:
            lines.append("")
        lines.extend(
            [
                "Summary of Sweep Results",
                "",
                self.analysis_summary_table(rows),
                "",
                "Credibility interval detail is available in the chat explorer and generated report.",
            ]
        )
        return lines

    def format_analysis_result(self, result: dict[str, Any]) -> str:
        return "\n".join(self.analysis_summary_sections(result, include_intro=True))

    def format_compact_result(self, result: dict[str, Any]) -> str:
        kind = result.get("kind")
        data = result.get("data", {})
        artifacts = result.get("artifacts", {})
        if kind == "profile":
            prepared_path = artifacts.get("prepared_dataset_path", "the prepared dataset")
            return f"Profiled the source dataset and saved `{prepared_path}`."
        if kind == "schema":
            source_path = data.get("source_path", "the dataset")
            column_count = data.get("column_count")
            if column_count is None:
                return f"Inspected the schema for `{source_path}`."
            return f"Inspected the schema for `{source_path}` ({int(column_count)} columns)."
        if kind == "validation":
            return result["message"]
        if kind == "feature_engineering":
            return result["message"]
        if kind == "analysis":
            return result["message"]
        if kind == "visualization":
            return result["message"]
        return result["message"]

    def next_steps(self) -> list[str]:
        next_steps: list[str] = []
        if self.state.prepared_dataset_ready and not self.state.latest_sweep_ready:
            next_steps.append("Run a dimensional sweep when you are ready for A/E analysis.")
        if self.state.latest_sweep_ready and not self.state.latest_visualization_ready:
            next_steps.append("Generate the combined report to visualize the latest sweep.")
        return next_steps

    def summarize_tool_results(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return "No deterministic work was executed."

        if len(results) == 1:
            result = results[0]
            if result.get("kind") == "schema":
                lines = [self.format_schema_result(result)]
            elif result.get("kind") == "profile":
                lines = [self.format_profile_result(result)]
            elif result.get("kind") == "analysis":
                lines = [self.format_analysis_result(result)]
            else:
                lines = [self.format_compact_result(result)]
        else:
            lines = ["Completed deterministic steps:"]
            lines.extend(f"- {self.format_compact_result(result)}" for result in results)
            schema_results = [result for result in results if result.get("kind") == "schema"]
            if schema_results:
                lines.append("")
                lines.append(self.format_schema_result(schema_results[-1]))
            analysis_results = [result for result in results if result.get("kind") == "analysis"]
            if analysis_results:
                lines.append("")
                lines.extend(
                    self.analysis_summary_sections(
                        analysis_results[-1],
                        include_intro=False,
                    )
                )

        next_steps = self.next_steps()
        if next_steps:
            lines.append("")
            lines.extend(next_steps)
        return "\n".join(lines)
