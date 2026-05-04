"""Unified copilot runtime with session-safe tool gating and event streaming."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterator, Literal
from uuid import uuid4

from core.fallback_planner import FallbackPlanner
from core.model_config import resolve_copilot_model
from core.openai_compat import get_client, log_openai_error, openai_error_type
from core.prerequisite_guard import (
    IntentSummary,
    enabled_tool_names,
    guard_missing_prerequisites,
)
from core.response_formatter import ResponseFormatter
from core.session_state import SessionArtifactState
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


class UnifiedCopilot:
    """Single-agent copilot backed by a self-contained skill package."""

    _MAX_SWEEP_TOP_N = FallbackPlanner._MAX_SWEEP_TOP_N
    _PATH_RE = FallbackPlanner._PATH_RE
    _FILTER_PATTERNS = FallbackPlanner._FILTER_PATTERNS
    _TEXT_OPERATOR_MAP = FallbackPlanner._TEXT_OPERATOR_MAP
    _THINKING_BLOCK_RE = ResponseFormatter._THINKING_BLOCK_RE

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
        self.fallback_planner = FallbackPlanner(self.state)
        self.response_formatter = ResponseFormatter(self.state)
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
        self.fallback_planner = FallbackPlanner(self.state)
        self.response_formatter = ResponseFormatter(self.state)
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
        return ResponseFormatter.sanitize_user_facing_text(text)

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
        return self.fallback_planner._extract_data_path(user_input)

    def _summarize_intent(self, user_input: str) -> IntentSummary:
        lowered = self.fallback_planner._PATH_RE.sub(" ", user_input).lower()
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
        return guard_missing_prerequisites(intent, state)

    def _enabled_tool_names(
        self,
        intent: IntentSummary,
        *,
        current_state: SessionArtifactState | None = None,
    ) -> set[str]:
        state = current_state or self.state
        return enabled_tool_names(intent, state)

    def _extract_depth(self, user_input: str) -> int:
        return self.fallback_planner._extract_depth(user_input)

    def _extract_top_n(self, user_input: str) -> int:
        return self.fallback_planner._extract_top_n(user_input)

    def _extract_min_mac(self, user_input: str) -> int:
        return self.fallback_planner._extract_min_mac(user_input)

    def _extract_sort_by(self, user_input: str) -> str:
        return self.fallback_planner._extract_sort_by(user_input)

    def _extract_metric(self, user_input: str) -> str:
        return self.fallback_planner._extract_metric(user_input)

    def _extract_selected_columns(self, user_input: str) -> list[str] | None:
        return self.fallback_planner._extract_selected_columns(user_input)

    def _parse_scalar_value(self, value: str) -> str | int | float:
        return self.fallback_planner._parse_scalar_value(value)

    def _parse_filter_clause(self, clause: str) -> dict[str, Any] | None:
        return self.fallback_planner._parse_filter_clause(clause)

    def _extract_filters(self, user_input: str) -> list[dict[str, Any]]:
        return self.fallback_planner._extract_filters(user_input)

    def _extract_band_args(self, user_input: str, intent: IntentSummary) -> dict[str, Any] | None:
        return self.fallback_planner._extract_band_args(user_input, intent)

    def _extract_regroup_args(self, user_input: str, intent: IntentSummary) -> dict[str, Any] | None:
        return self.fallback_planner._extract_regroup_args(user_input, intent)

    def _extract_sweep_args(self, user_input: str) -> dict[str, Any]:
        return self.fallback_planner._extract_sweep_args(user_input)

    def _extract_visualization_args(self, user_input: str) -> dict[str, Any]:
        return self.fallback_planner._extract_visualization_args(user_input)

    def _extract_schema_args(self, intent: IntentSummary) -> dict[str, Any]:
        return self.fallback_planner._extract_schema_args(intent)

    def _build_fallback_plan(
        self,
        user_input: str,
        intent: IntentSummary,
    ) -> tuple[list[tuple[str, dict[str, Any]]], str | None]:
        return self.fallback_planner._build_fallback_plan(user_input, intent)

    def _format_schema_result(self, result: dict[str, Any]) -> str:
        return self.response_formatter.format_schema_result(result)

    def _format_profile_result(self, result: dict[str, Any]) -> str:
        return self.response_formatter.format_profile_result(result)

    @staticmethod
    def _format_sweep_value(value: Any) -> str:
        return ResponseFormatter.format_sweep_value(value)

    def _analysis_summary_table(self, rows: list[dict[str, Any]]) -> str:
        return self.response_formatter.analysis_summary_table(rows)

    def _analysis_summary_sections(
        self,
        result: dict[str, Any],
        *,
        include_intro: bool,
    ) -> list[str]:
        return self.response_formatter.analysis_summary_sections(
            result,
            include_intro=include_intro,
        )

    def _format_analysis_result(self, result: dict[str, Any]) -> str:
        return self.response_formatter.format_analysis_result(result)

    def _format_compact_result(self, result: dict[str, Any]) -> str:
        return self.response_formatter.format_compact_result(result)

    def _next_steps(self) -> list[str]:
        return self.response_formatter.next_steps()

    def _summarize_tool_results(self, results: list[dict[str, Any]]) -> str:
        return self.response_formatter.summarize_tool_results(results)

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
