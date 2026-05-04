"""Deterministic fallback planning for the unified copilot runtime."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from core.prerequisite_guard import IntentSummary, guard_missing_prerequisites
from core.session_state import SessionArtifactState


class FallbackPlanner:
    """Builds deterministic fallback tool plans from the current intent and state."""

    _MAX_SWEEP_TOP_N = 20
    _PATH_RE = re.compile(r"((?:/|[A-Za-z]:[\\/])?[\w./\\-]+\.(?:csv|parquet|xlsx))")
    _FILTER_PATTERNS = (
        r"\bwhere\s+(.+?)(?:,?\s+then\b|,?\s+rank\b|,?\s+sort\b|,?\s+using\b|[?.]|$)",
        r"\bonly\s+for\s+(.+?)(?:,?\s+then\b|,?\s+rank\b|,?\s+sort\b|,?\s+using\b|[?.]|$)",
        r"\bwith\s+(.+?)(?:,?\s+then\b|,?\s+rank\b|,?\s+sort\b|,?\s+using\b|[?.]|$)",
    )
    _TEXT_OPERATOR_MAP = {
        "greater than or equal to": ">=",
        "less than or equal to": "<=",
        "not equal to": "!=",
        "equal to": "=",
        "greater than": ">",
        "less than": "<",
        "at least": ">=",
        "at most": "<=",
        "equals": "=",
        "over": ">",
        "under": "<",
        "is not": "!=",
        "is": "=",
        "not": "!=",
    }

    def __init__(self, state: SessionArtifactState) -> None:
        self.state = state

    def _extract_data_path(self, user_input: str) -> str | None:
        match = self._PATH_RE.search(user_input)
        if not match:
            return None
        return match.group(1).replace("\\", "/")

    def _extract_depth(self, user_input: str) -> int:
        lowered = user_input.lower()
        match = re.search(r"\b([123])[- ]way\b", lowered)
        if match:
            return int(match.group(1))
        if "pairwise" in lowered or "all pairs" in lowered:
            return 2
        return 1

    def _extract_top_n(self, user_input: str) -> int:
        match = re.search(r"\btop\s+(\d+)\b", user_input.lower())
        requested = int(match.group(1)) if match else self._MAX_SWEEP_TOP_N
        return max(1, min(requested, self._MAX_SWEEP_TOP_N))

    def _extract_min_mac(self, user_input: str) -> int:
        lowered = user_input.lower()
        match = re.search(r"\bmin_mac\s*=\s*(\d+)", lowered)
        if match:
            return int(match.group(1))
        match = re.search(r"\bat least\s+(\d+)\s+deaths?\b", lowered)
        return int(match.group(1)) if match else 0

    def _extract_sort_by(self, user_input: str) -> str:
        lowered = user_input.lower()
        explicit_match = re.search(
            r"\b(ae_ratio_amount|ae_ratio_count|sum_mac|sum_moc|sum_mec|sum_maf|sum_mef)\b",
            lowered,
        )
        if explicit_match:
            return {
                "ae_ratio_amount": "AE_Ratio_Amount",
                "ae_ratio_count": "AE_Ratio_Count",
                "sum_mac": "Sum_MAC",
                "sum_moc": "Sum_MOC",
                "sum_mec": "Sum_MEC",
                "sum_maf": "Sum_MAF",
                "sum_mef": "Sum_MEF",
            }[explicit_match.group(1)]
        if "count" in lowered:
            return "AE_Ratio_Count"
        return "AE_Ratio_Amount"

    def _extract_metric(self, user_input: str) -> str:
        return "count" if "count" in user_input.lower() else "amount"

    def _extract_selected_columns(self, user_input: str) -> list[str] | None:
        patterns = [
            r"\bbetween\s+(.+?)(?:,?\s+then\b|,?\s+where\b|,?\s+rank\b|[?.]|$)",
            r"\bacross\s+(.+?)(?:,?\s+then\b|,?\s+where\b|,?\s+rank\b|[?.]|$)",
            r"\bon\s+(.+?)(?:,?\s+then\b|,?\s+where\b|,?\s+rank\b|[?.]|$)",
            r"\bfor\s+(.+?)(?:,?\s+then\b|,?\s+where\b|,?\s+rank\b|[?.]|$)",
        ]
        requested_segment: str | None = None
        for pattern in patterns:
            match = re.search(pattern, user_input, flags=re.IGNORECASE)
            if match:
                requested_segment = match.group(1)
                break
        if not requested_segment:
            return None

        cleaned = requested_segment.replace("×", ",").replace(" x ", ", ")
        cleaned = re.split(
            r",?\s+(?:and\s+)?(?:generate|create|show|visuali[sz]e|report)\b",
            cleaned,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        cleaned = re.sub(r"\b(all pairs|pairwise|the latest sweep|latest sweep summary|the sweep)\b", "", cleaned, flags=re.IGNORECASE)
        tokens = [
            token.strip(" `.")
            for token in re.split(r",|\band\b|&", cleaned, flags=re.IGNORECASE)
            if token.strip(" `.")
        ]
        selected_columns: list[str] = []
        for token in tokens:
            if token.lower() in {"all", "all eligible dimensions", "all dimensions"}:
                return None
            normalized = re.sub(r"[^A-Za-z0-9]+", "_", token).strip("_")
            if normalized and normalized not in selected_columns:
                selected_columns.append(normalized)
        return selected_columns or None

    def _parse_scalar_value(self, value: str) -> str | int | float:
        cleaned = value.strip().strip("`")
        if (cleaned.startswith("'") and cleaned.endswith("'")) or (
            cleaned.startswith('"') and cleaned.endswith('"')
        ):
            cleaned = cleaned[1:-1]
        if re.fullmatch(r"-?\d+", cleaned):
            return int(cleaned)
        if re.fullmatch(r"-?\d+\.\d+", cleaned):
            return float(cleaned)
        return cleaned

    def _parse_filter_clause(self, clause: str) -> dict[str, Any] | None:
        symbolic_match = re.match(
            r"^(?P<column>.+?)\s*(?P<operator>>=|<=|!=|=|>|<)\s*(?P<value>.+)$",
            clause.strip(" ."),
        )
        if symbolic_match:
            return {
                "column": re.sub(
                    r"^(?:the\s+column\s+|column\s+)",
                    "",
                    symbolic_match.group("column").strip(),
                    flags=re.IGNORECASE,
                ),
                "operator": symbolic_match.group("operator"),
                "value": self._parse_scalar_value(symbolic_match.group("value")),
            }

        for text_operator, symbol in sorted(
            self._TEXT_OPERATOR_MAP.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            pattern = rf"^(?P<column>.+?)\s+{re.escape(text_operator)}\s+(?P<value>.+)$"
            match = re.match(pattern, clause.strip(" ."), flags=re.IGNORECASE)
            if match:
                return {
                    "column": re.sub(
                        r"^(?:the\s+column\s+|column\s+)",
                        "",
                        match.group("column").strip(),
                        flags=re.IGNORECASE,
                    ),
                    "operator": symbol,
                    "value": self._parse_scalar_value(match.group("value")),
                }
        return None

    def _extract_filters(self, user_input: str) -> list[dict[str, Any]]:
        lowered = user_input.lower()
        if re.search(r"\bat least\s+\d+\s+deaths?\b", lowered):
            scrubbed_input = re.sub(
                r"\bwith\s+at least\s+\d+\s+deaths?\b",
                "",
                user_input,
                flags=re.IGNORECASE,
            )
        else:
            scrubbed_input = user_input

        for pattern in self._FILTER_PATTERNS:
            match = re.search(pattern, scrubbed_input, flags=re.IGNORECASE)
            if not match:
                continue
            segment = match.group(1).strip()
            clauses = [
                clause.strip(" ,.")
                for clause in re.split(r"\band\b", segment, flags=re.IGNORECASE)
                if clause.strip(" ,.")
            ]
            parsed: list[dict[str, Any]] = []
            for clause in clauses:
                parsed_clause = self._parse_filter_clause(clause)
                if not parsed_clause:
                    return []
                parsed.append(parsed_clause)
            return parsed
        return []

    def _extract_band_args(self, user_input: str, intent: IntentSummary) -> dict[str, Any] | None:
        patterns = [
            r"group\s+(?P<column>[A-Za-z_][A-Za-z0-9_ ]*)\s+into\s+(?P<bins>\d+)\s+(?P<strategy>equal[-\s]+width|quantiles?|equal[-\s]+quantile|same[-\s]+quantile)\s+bands?",
            r"create\s+(?P<bins>\d+)\s+(?P<strategy>equal[-\s]+width|quantiles?|equal[-\s]+quantile|same[-\s]+quantile)\s+bands?\s+(?:for|on)\s+(?P<column>[A-Za-z_][A-Za-z0-9_ ]*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, user_input, flags=re.IGNORECASE)
            if not match:
                continue
            strategy_text = re.sub(r"[-\s]+", " ", match.group("strategy").strip().lower())
            strategy = "quantiles" if "quantile" in strategy_text else "equal_width"
            return {
                "source_column": re.sub(
                    r"[^A-Za-z0-9]+",
                    "_",
                    match.group("column").strip(),
                ).strip("_"),
                "strategy": strategy,
                "bins": int(match.group("bins")),
            }
        return None

    def _extract_regroup_args(self, user_input: str, intent: IntentSummary) -> dict[str, Any] | None:
        column_match = re.search(
            r"regroup.*?(?:for|on)\s+(?P<column>[A-Za-z_][A-Za-z0-9_ ]*)",
            user_input,
            flags=re.IGNORECASE,
        )
        mapping_match = re.search(r"(\{.*\})", user_input)
        if not column_match or not mapping_match:
            return None
        try:
            mapping_dict = json.loads(mapping_match.group(1))
        except json.JSONDecodeError:
            return None
        return {
            "source_column": re.sub(
                r"[^A-Za-z0-9]+", "_", column_match.group("column").strip()
            ).strip("_"),
            "mapping_dict": mapping_dict,
        }

    def _extract_sweep_args(self, user_input: str) -> dict[str, Any]:
        return {
            "depth": self._extract_depth(user_input),
            "min_mac": self._extract_min_mac(user_input),
            "top_n": self._extract_top_n(user_input),
            "sort_by": self._extract_sort_by(user_input),
            "filters": self._extract_filters(user_input),
            "selected_columns": self._extract_selected_columns(user_input),
        }

    def _extract_visualization_args(self, user_input: str) -> dict[str, Any]:
        explicit_csv = self._extract_data_path(user_input)
        data_path = (
            explicit_csv
            if explicit_csv
            and explicit_csv.endswith(".csv")
            and "sweep_summary" in Path(explicit_csv).name
            else None
        )
        return {"metric": self._extract_metric(user_input), "data_path": data_path}

    def _extract_schema_args(self, intent: IntentSummary) -> dict[str, Any]:
        if intent.wants_profile or intent.wants_full_pipeline:
            return {"data_path": None}
        return {"data_path": intent.explicit_data_path}

    def _build_fallback_plan(
        self,
        user_input: str,
        intent: IntentSummary,
    ) -> tuple[list[tuple[str, dict[str, Any]]], str | None]:
        guidance = guard_missing_prerequisites(intent, self.state)
        if guidance:
            return ([], guidance)

        plan: list[tuple[str, dict[str, Any]]] = []
        profile_path = intent.explicit_data_path or (
            str(self.state.raw_input_path) if self.state.raw_input_path else None
        )

        if (intent.wants_profile or intent.wants_full_pipeline) and profile_path:
            plan.append(("profile_dataset", {"data_path": profile_path}))

        if intent.wants_schema:
            plan.append(("inspect_dataset_schema", self._extract_schema_args(intent)))

        if intent.wants_validate:
            plan.append(("run_actuarial_data_checks", {"data_path": intent.explicit_data_path}))

        if intent.wants_band:
            band_args = self._extract_band_args(user_input, intent)
            if band_args is None:
                return ([], "Specify the column, number of bins, and strategy for banding.")
            plan.append(("create_categorical_bands", band_args))

        if intent.wants_regroup:
            regroup_args = self._extract_regroup_args(user_input, intent)
            if regroup_args is None:
                return (
                    [],
                    "Specify the source column and a valid JSON mapping dictionary for regrouping.",
                )
            plan.append(("regroup_categorical_features", regroup_args))

        if intent.wants_analysis or intent.wants_full_pipeline:
            plan.append(("run_dimensional_sweep", self._extract_sweep_args(user_input)))

        if intent.wants_visualize:
            plan.append(("generate_combined_report", self._extract_visualization_args(user_input)))

        if not plan:
            return (
                [],
                "I can inspect dataset schemas, profile a dataset, validate it, engineer features, run dimensional sweeps, or generate the combined report.",
            )
        return (plan, None)
