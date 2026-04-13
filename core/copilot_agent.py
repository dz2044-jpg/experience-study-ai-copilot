"""Unified copilot runtime with session-safe tool gating and event streaming."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterator, Literal
from uuid import uuid4

from core.model_config import resolve_copilot_model
from core.openai_compat import get_client, log_openai_error, openai_error_type
from core.skill_loader import LoadedSkill, load_skill


EventType = Literal[
    "status",
    "text_delta",
    "tool_start",
    "tool_result",
    "artifact_update",
    "final",
]


@dataclass(slots=True)
class CopilotEvent:
    """Structured event emitted by the unified copilot."""

    type: EventType
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SessionArtifactState:
    """Tracks the session-local artifact graph owned by the copilot."""

    session_id: str
    output_base_dir: Path
    raw_input_path: Path | None = None
    prepared_dataset_path: Path | None = None
    prepared_dataset_ready: bool = False
    latest_sweep_path: Path | None = None
    latest_sweep_ready: bool = False
    latest_sweep_paths_by_depth: dict[int, Path] = field(default_factory=dict)
    latest_visualization_path: Path | None = None
    latest_visualization_ready: bool = False

    @property
    def output_dir(self) -> Path:
        return self.output_base_dir / self.session_id

    def refresh(self) -> None:
        self.prepared_dataset_ready = bool(
            self.prepared_dataset_path and self.prepared_dataset_path.exists()
        )
        self.latest_sweep_ready = bool(self.latest_sweep_path and self.latest_sweep_path.exists())
        self.latest_visualization_ready = bool(
            self.latest_visualization_path and self.latest_visualization_path.exists()
        )
        self.latest_sweep_paths_by_depth = {
            depth: path
            for depth, path in self.latest_sweep_paths_by_depth.items()
            if path.exists()
        }

    def apply_tool_result(self, result: dict[str, Any]) -> bool:
        artifacts = result.get("artifacts", {})
        changed = False

        raw_input_path = artifacts.get("raw_input_path")
        if raw_input_path:
            self.raw_input_path = Path(raw_input_path)
            changed = True

        prepared_dataset_path = artifacts.get("prepared_dataset_path")
        if prepared_dataset_path:
            self.prepared_dataset_path = Path(prepared_dataset_path)
            changed = True

        sweep_summary_path = artifacts.get("sweep_summary_path")
        if sweep_summary_path:
            self.latest_sweep_path = Path(sweep_summary_path)
            changed = True

        sweep_depth = artifacts.get("sweep_depth")
        sweep_depth_path = artifacts.get("sweep_depth_path")
        if sweep_depth is not None and sweep_depth_path:
            self.latest_sweep_paths_by_depth[int(sweep_depth)] = Path(sweep_depth_path)
            changed = True

        visualization_path = artifacts.get("visualization_path")
        if visualization_path:
            self.latest_visualization_path = Path(visualization_path)
            changed = True

        if changed:
            self.refresh()
        return changed

    def to_prompt(self) -> str:
        self.refresh()
        available_depths = sorted(self.latest_sweep_paths_by_depth)
        sweep_paths_by_depth = {
            depth: str(path) for depth, path in sorted(self.latest_sweep_paths_by_depth.items())
        }
        lines = [
            "Current Session State:",
            f"- session_id: {self.session_id}",
            f"- output_dir: {self.output_dir}",
            f"- raw_input_path: {self.raw_input_path or 'None'}",
            f"- prepared_dataset_ready: {self.prepared_dataset_ready}",
            f"- prepared_dataset_path: {self.prepared_dataset_path or 'None'}",
            f"- latest_sweep_ready: {self.latest_sweep_ready}",
            f"- latest_sweep_path: {self.latest_sweep_path or 'None'}",
            f"- latest_sweep_paths_by_depth: {sweep_paths_by_depth}",
            f"- available_sweep_depths: {available_depths}",
            f"- latest_visualization_ready: {self.latest_visualization_ready}",
            f"- latest_visualization_path: {self.latest_visualization_path or 'None'}",
        ]
        return "\n".join(lines)

    def to_event_payload(self) -> dict[str, Any]:
        self.refresh()
        return {
            "session_id": self.session_id,
            "output_dir": str(self.output_dir),
            "raw_input_path": str(self.raw_input_path) if self.raw_input_path else None,
            "prepared_dataset_ready": self.prepared_dataset_ready,
            "prepared_dataset_path": (
                str(self.prepared_dataset_path) if self.prepared_dataset_path else None
            ),
            "latest_sweep_ready": self.latest_sweep_ready,
            "latest_sweep_path": str(self.latest_sweep_path) if self.latest_sweep_path else None,
            "latest_sweep_paths_by_depth": {
                str(depth): str(path)
                for depth, path in self.latest_sweep_paths_by_depth.items()
            },
            "latest_visualization_ready": self.latest_visualization_ready,
            "latest_visualization_path": (
                str(self.latest_visualization_path)
                if self.latest_visualization_path
                else None
            ),
        }


@dataclass(slots=True)
class IntentSummary:
    """High-level intent classification used for gating and fallback routing."""

    explicit_data_path: str | None
    wants_profile: bool
    wants_schema: bool
    wants_validate: bool
    wants_band: bool
    wants_regroup: bool
    wants_analysis: bool
    wants_visualize: bool
    wants_full_pipeline: bool

    @property
    def is_general(self) -> bool:
        return not any(
            (
                self.wants_profile,
                self.wants_schema,
                self.wants_validate,
                self.wants_band,
                self.wants_regroup,
                self.wants_analysis,
                self.wants_visualize,
                self.wants_full_pipeline,
            )
        )


class UnifiedCopilot:
    """Single-agent copilot backed by a self-contained skill package."""

    _MAX_SWEEP_TOP_N = 20
    _PATH_RE = re.compile(r"((?:/|[A-Za-z]:[\\/])?[\w./\\-]+\.(?:csv|parquet|xlsx))")
    _THINKING_BLOCK_RE = re.compile(r"<thinking>.*?</thinking>", re.IGNORECASE | re.DOTALL)
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

    def __init__(
        self,
        *,
        skill_name: str = "experience-study-skill",
        model: str | None = None,
        session_id: str | None = None,
        output_base_dir: str | Path = "data/output/sessions",
    ) -> None:
        self.client = get_client()
        self.model = resolve_copilot_model(model)
        self.active_skill: LoadedSkill = load_skill(skill_name)
        self.history: list[dict[str, str]] = []
        self.state = SessionArtifactState(
            session_id=session_id or self.new_session_id(),
            output_base_dir=Path(output_base_dir),
        )
        self.state.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def new_session_id() -> str:
        return uuid4().hex[:12]

    def reset_session(self) -> str:
        old_output_dir = self.state.output_dir
        shutil.rmtree(old_output_dir, ignore_errors=True)
        self.history = []
        self.state = SessionArtifactState(
            session_id=self.new_session_id(),
            output_base_dir=self.state.output_base_dir,
        )
        self.state.output_dir.mkdir(parents=True, exist_ok=True)
        return self.state.session_id

    def _build_tool_context(self) -> Any:
        return self.active_skill.tool_context_type(
            session_id=self.state.session_id,
            output_dir=self.state.output_dir,
            raw_input_path=self.state.raw_input_path,
            prepared_dataset_path=self.state.prepared_dataset_path,
            latest_sweep_path=self.state.latest_sweep_path,
            latest_sweep_paths_by_depth=dict(self.state.latest_sweep_paths_by_depth),
            latest_visualization_path=self.state.latest_visualization_path,
        )

    def _stream_text(self, text: str) -> Iterator[CopilotEvent]:
        for chunk in re.findall(r"\S+\s*", text):
            yield CopilotEvent("text_delta", message=chunk)

    @classmethod
    def _sanitize_user_facing_text(cls, text: str) -> str:
        if not text:
            return ""
        sanitized = cls._THINKING_BLOCK_RE.sub("", text)
        sanitized = re.sub(r"[ \t]+\n", "\n", sanitized)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized.strip()

    def _finalize_response(
        self,
        user_input: str,
        text: str,
        *,
        fallback_text: str = "",
    ) -> Iterator[CopilotEvent]:
        final_text = self._sanitize_user_facing_text(text)
        if not final_text and fallback_text:
            final_text = self._sanitize_user_facing_text(fallback_text)
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": final_text})
        yield from self._stream_text(final_text)
        yield CopilotEvent(
            "final",
            message=final_text,
            data={"artifact_state": self.state.to_event_payload()},
        )

    def _extract_data_path(self, user_input: str) -> str | None:
        match = self._PATH_RE.search(user_input)
        if not match:
            return None
        return match.group(1).replace("\\", "/")

    def _summarize_intent(self, user_input: str) -> IntentSummary:
        lowered = self._PATH_RE.sub(" ", user_input).lower()
        wants_profile = "profile" in lowered
        wants_schema = any(
            token in lowered for token in ("columns", "schema", "data types", "dtype", "dtypes")
        )
        wants_validate = any(
            token in lowered for token in ("validate", "check the data", "check data", "missing values", "errors")
        )
        wants_band = "band" in lowered or "bucket" in lowered or "equal-width" in lowered
        wants_regroup = "regroup" in lowered or "mapping" in lowered
        wants_analysis = any(
            token in lowered
            for token in ("a/e", "actual-to-expected", "cohort", "mortality", "analyze")
        ) or (
            "sweep" in lowered
            and any(
                token in lowered
                for token in ("run", "calculate", "rank", "show", "find", "analyze")
            )
        )
        wants_visualize = any(
            token in lowered
            for token in ("visual", "chart", "plot", "report", "treemap", "forest plot")
        )
        wants_full_pipeline = any(
            token in lowered
            for token in ("end-to-end", "full pipeline", "full workflow", "do everything", "run everything")
        ) or sum(
            int(flag)
            for flag in (wants_profile, wants_band, wants_analysis, wants_visualize)
        ) >= 3
        return IntentSummary(
            explicit_data_path=self._extract_data_path(user_input),
            wants_profile=wants_profile,
            wants_schema=wants_schema,
            wants_validate=wants_validate,
            wants_band=wants_band,
            wants_regroup=wants_regroup,
            wants_analysis=wants_analysis,
            wants_visualize=wants_visualize,
            wants_full_pipeline=wants_full_pipeline,
        )

    def _guard_missing_prerequisites(
        self,
        intent: IntentSummary,
        *,
        current_state: SessionArtifactState | None = None,
    ) -> str | None:
        state = current_state or self.state
        state.refresh()
        has_prepared = state.prepared_dataset_ready or state.prepared_dataset_path is not None
        has_sweep = state.latest_sweep_ready or state.latest_sweep_path is not None
        has_dataset = (
            intent.explicit_data_path
            or state.prepared_dataset_ready
            or state.prepared_dataset_path is not None
            or state.raw_input_path is not None
        )
        if intent.wants_schema and not has_dataset and not (
            intent.wants_profile or intent.wants_full_pipeline
        ):
            return "No dataset is available. Profile a dataset first or provide a data_path."
        if intent.wants_visualize and not (
            has_sweep or intent.wants_analysis or intent.wants_full_pipeline
        ):
            return "No sweep artifact exists for this session. Run a dimensional sweep first."
        if intent.wants_analysis and not (
            has_prepared
            or intent.wants_profile
            or intent.wants_band
            or intent.wants_regroup
            or intent.wants_full_pipeline
        ):
            return "No prepared dataset exists for this session. Profile a dataset first."
        if (intent.wants_profile or intent.wants_full_pipeline) and not (
            intent.explicit_data_path or state.raw_input_path
        ):
            return "Provide a dataset path to start the experience study workflow."
        if (intent.wants_band or intent.wants_regroup) and not (
            state.prepared_dataset_ready or state.raw_input_path or intent.explicit_data_path
        ):
            return "No dataset is available for feature engineering. Profile a dataset first or provide a data_path."
        return None

    def _enabled_tool_names(
        self,
        intent: IntentSummary,
        *,
        current_state: SessionArtifactState | None = None,
    ) -> set[str]:
        state = current_state or self.state
        state.refresh()
        has_prepared = state.prepared_dataset_ready or state.prepared_dataset_path is not None
        has_sweep = state.latest_sweep_ready or state.latest_sweep_path is not None
        enabled: set[str] = set()

        if intent.wants_profile or intent.wants_full_pipeline:
            if intent.explicit_data_path or state.raw_input_path:
                enabled.add("profile_dataset")
        if intent.wants_schema:
            if (
                intent.explicit_data_path
                or state.prepared_dataset_ready
                or state.prepared_dataset_path
                or state.raw_input_path
            ):
                enabled.add("inspect_dataset_schema")
        if intent.wants_validate:
            if intent.explicit_data_path or state.raw_input_path or state.prepared_dataset_ready:
                enabled.add("run_actuarial_data_checks")
        if intent.wants_band:
            if intent.explicit_data_path or state.raw_input_path or state.prepared_dataset_ready:
                enabled.add("create_categorical_bands")
        if intent.wants_regroup:
            if intent.explicit_data_path or state.raw_input_path or state.prepared_dataset_ready:
                enabled.add("regroup_categorical_features")
        if has_prepared and (intent.wants_analysis or intent.wants_full_pipeline):
            enabled.add("run_dimensional_sweep")
        if has_sweep and intent.wants_visualize:
            enabled.add("generate_combined_report")
        return enabled

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
        guidance = self._guard_missing_prerequisites(intent)
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

    def _format_schema_result(self, result: dict[str, Any]) -> str:
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

    def _format_profile_result(self, result: dict[str, Any]) -> str:
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
    def _format_sweep_value(value: Any) -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    def _analysis_summary_table(self, rows: list[dict[str, Any]]) -> str:
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
                self._format_sweep_value(row.get("Sum_MAC")),
                self._format_sweep_value(row.get("Sum_MEC")),
                self._format_sweep_value(row.get("AE_Ratio_Count")),
                self._format_sweep_value(row.get("AE_Ratio_Amount")),
            ]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    def _analysis_summary_sections(
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
                self._analysis_summary_table(rows),
                "",
                "Credibility interval detail is available in the chat explorer and generated report.",
            ]
        )
        return lines

    def _format_analysis_result(self, result: dict[str, Any]) -> str:
        return "\n".join(self._analysis_summary_sections(result, include_intro=True))

    def _format_compact_result(self, result: dict[str, Any]) -> str:
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

    def _next_steps(self) -> list[str]:
        next_steps: list[str] = []
        if self.state.prepared_dataset_ready and not self.state.latest_sweep_ready:
            next_steps.append("Run a dimensional sweep when you are ready for A/E analysis.")
        if self.state.latest_sweep_ready and not self.state.latest_visualization_ready:
            next_steps.append("Generate the combined report to visualize the latest sweep.")
        return next_steps

    def _summarize_tool_results(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return "No deterministic work was executed."

        if len(results) == 1:
            result = results[0]
            if result.get("kind") == "schema":
                lines = [self._format_schema_result(result)]
            elif result.get("kind") == "profile":
                lines = [self._format_profile_result(result)]
            elif result.get("kind") == "analysis":
                lines = [self._format_analysis_result(result)]
            else:
                lines = [self._format_compact_result(result)]
        else:
            lines = ["Completed deterministic steps:"]
            lines.extend(f"- {self._format_compact_result(result)}" for result in results)
            schema_results = [result for result in results if result.get("kind") == "schema"]
            if schema_results:
                lines.append("")
                lines.append(self._format_schema_result(schema_results[-1]))
            analysis_results = [result for result in results if result.get("kind") == "analysis"]
            if analysis_results:
                lines.append("")
                lines.extend(
                    self._analysis_summary_sections(
                        analysis_results[-1],
                        include_intro=False,
                    )
                )

        next_steps = self._next_steps()
        if next_steps:
            lines.append("")
            lines.extend(next_steps)
        return "\n".join(lines)

    def _execute_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> tuple[dict[str, Any], list[CopilotEvent]]:
        handler = self.active_skill.tool_handlers[tool_name]
        context = self._build_tool_context()
        result = handler(args, context)
        events: list[CopilotEvent] = [
            CopilotEvent("tool_start", message=f"Executing `{tool_name}`.", data={"args": args})
        ]
        for status_message in context.status_events:
            events.append(CopilotEvent("status", message=status_message))
        events.append(CopilotEvent("tool_result", message=result["message"], data={"result": result}))
        if self.state.apply_tool_result(result):
            events.append(
                CopilotEvent(
                    "artifact_update",
                    message="Session artifacts updated.",
                    data=self.state.to_event_payload(),
                )
            )
        return result, events

    def _fallback_process(self, user_input: str, intent: IntentSummary) -> Iterator[CopilotEvent]:
        plan, guidance = self._build_fallback_plan(user_input, intent)
        if guidance:
            yield from self._finalize_response(user_input, guidance)
            return

        tool_results: list[dict[str, Any]] = []
        for tool_name, args in plan:
            result, events = self._execute_tool_call(tool_name, args)
            for event in events:
                yield event
            tool_results.append(result)
            if not result.get("ok", False):
                yield from self._finalize_response(user_input, result["message"])
                return

        yield from self._finalize_response(user_input, self._summarize_tool_results(tool_results))

    def _llm_messages(self, user_input: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.active_skill.instructions},
            {"role": "system", "content": self.state.to_prompt()},
        ]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_input})
        return messages

    def process_message(self, user_input: str) -> Iterator[CopilotEvent]:
        intent = self._summarize_intent(user_input)
        yield CopilotEvent("status", message="Copilot received a new request.")

        if intent.is_general:
            text = (
                "I can inspect dataset schemas, profile a dataset, validate it, engineer features, "
                "run dimensional sweeps, or generate the combined report."
            )
            yield from self._finalize_response(user_input, text)
            return

        if self.client is None:
            yield CopilotEvent(
                "status",
                message="OpenAI is unavailable. Using deterministic local planning.",
            )
            yield from self._fallback_process(user_input, intent)
            return

        guard_message = self._guard_missing_prerequisites(intent)
        if guard_message:
            yield from self._finalize_response(user_input, guard_message)
            return

        working_messages = self._llm_messages(user_input)
        tool_results: list[dict[str, Any]] = []

        for _ in range(6):
            enabled_tools = self._enabled_tool_names(intent)
            if not enabled_tools:
                break
            yield CopilotEvent("status", message="Requesting the next action from the model.")
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=working_messages,
                    tools=self.active_skill.tool_spec_factory(enabled_tools),
                    tool_choice="auto",
                )
            except Exception as exc:  # pragma: no cover - depends on network/runtime
                log_openai_error("UnifiedCopilot", "Tool-calling request", exc)
                yield CopilotEvent(
                    "status",
                    message=(
                        "OpenAI tool-calling is unavailable "
                        f"({openai_error_type(exc)}). Falling back to deterministic local planning."
                    ),
                )
                yield from self._fallback_process(user_input, intent)
                return

            message = completion.choices[0].message
            tool_calls = message.tool_calls or []
            if not tool_calls:
                fallback_text = self._summarize_tool_results(tool_results)
                final_text = message.content or fallback_text
                yield from self._finalize_response(
                    user_input,
                    final_text,
                    fallback_text=fallback_text,
                )
                return

            working_messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [tool_call.model_dump() for tool_call in tool_calls],
                }
            )

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                result, events = self._execute_tool_call(tool_name, args)
                for event in events:
                    yield event
                tool_results.append(result)
                working_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )
                if not result.get("ok", False):
                    yield from self._finalize_response(user_input, result["message"])
                    return

        yield from self._finalize_response(user_input, self._summarize_tool_results(tool_results))
