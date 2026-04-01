"""Data Steward Agent: profiles, validates, and engineers inforce data."""

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.model_config import resolve_steward_model
from agents.openai_compat import (
    build_openai_client,
    log_openai_error,
    openai_error_type,
)
from agents.schemas import (
    CategoricalBandingSchema,
    ProfileDatasetSchema,
    RegroupCategoricalSchema,
)

from tools.data_steward import (
    CategoricalBandSpec,
    create_categorical_bands,
    create_categorical_bands_batch,
    profile_dataset,
    regroup_categorical_features,
    run_actuarial_data_checks,
)
from tools.data_io import (
    CANONICAL_ANALYSIS_OUTPUT_PATH,
    get_tabular_columns,
    resolve_prepared_analysis_path,
)

RAW_INPUT_PATH_RE = re.compile(r"(?<!\w)((?:/|data/)?[\w./-]+\.(?:csv|parquet|xlsx))(?!\w)", re.IGNORECASE)
DEFAULT_RAW_INPUT_PATH = "data/input/synthetic_inforce.csv"
BANDING_STRATEGY_PATTERN = (
    r"equal[-\s]+width|"
    r"same[-\s]+quantile|"
    r"equal[-\s]+quantile|"
    r"quantiles?"
)
BANDING_CLAUSE_RE = re.compile(
    rf"(?:\b(?:create|add|make)\b\s+)?"
    rf"(?P<bins>\d+)\s+"
    rf"(?P<strategy>{BANDING_STRATEGY_PATTERN})\s+"
    rf"bands?\s+(?:for|on)\s+"
    rf"(?P<column>.+?)"
    rf"(?=(?:\s*,?\s*(?:and|then)\s+(?:\b(?:create|add|make)\b\s+)?\d+\s+(?:{BANDING_STRATEGY_PATTERN})\s+bands?\s+(?:for|on)\s+)|[?.!]|$)",
    re.IGNORECASE,
)
MIXED_BANDING_BLOCKERS = (
    "profile",
    "validate",
    "schema",
    "null count",
    "data type",
    "column names",
    "list columns",
    "show columns",
    "check the data",
    "check data",
    "regroup",
)
IGNORABLE_BANDING_TOKENS = {
    "a",
    "add",
    "an",
    "and",
    "also",
    "create",
    "for",
    "from",
    "in",
    "kindly",
    "make",
    "please",
    "then",
    "the",
    "to",
    "use",
    "using",
}


SYSTEM_PROMPT = """
You are a meticulous Data Engineer and Actuarial Data Steward.

Your job is to profile datasets to ensure actuarial integrity, including MAC/MEC/MOC logic,
and perform feature engineering (for example, age or face-amount banding) so actuaries can run
multi-dimensional sweeps.

Use tools whenever possible. Be technical, concise, and helpful.
Always confirm what data transformations were applied and where output was saved.

CRITICAL GUARDRAILS: 
1. Immutable Input: You may read from `data/input/`, but you must save all outputs strictly to `data/output/analysis_inforce.parquet`.
2. Null Preservation: You shall NEVER automatically drop rows with Null/NaN values, nor impute them, unless explicitly instructed by the user. 
3. Domain Rule: In life insurance, a Null in the 'COLA' column is expected if there is no claim. Do not flag this as an error.
""".strip()


class DataStewardAgent:
    """Agent wrapper that routes user requests to deterministic steward tools."""

    def __init__(
        self,
        model: Optional[str] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.model = resolve_steward_model(model)
        self.client = build_openai_client()
        self.status_callback = status_callback

        self.tool_handlers: Dict[str, Callable[..., str]] = {
            "profile_dataset": profile_dataset,
            "run_actuarial_data_checks": run_actuarial_data_checks,
            "create_categorical_bands": create_categorical_bands,
            "regroup_categorical_features": regroup_categorical_features,
        }

    def set_status_callback(self, status_callback: Optional[Callable[[str], None]]) -> None:
        """Set a per-request status callback."""
        self.status_callback = status_callback

    def _emit_status(self, message: str) -> None:
        """Publish a UI status update when available."""
        if self.status_callback:
            self.status_callback(message)

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
            {
                "type": "function",
                "function": {
                    "name": "regroup_categorical_features",
                    "description": "Regroup categorical values into a derived analysis feature and save to analysis dataset.",
                    "parameters": RegroupCategoricalSchema.model_json_schema(),
                },
            },
        ]

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        if tool_name not in self.tool_handlers:
            return json.dumps({"error": f"Unknown tool: {tool_name}"}, indent=2)

        if tool_name == "create_categorical_bands" and "operation" in args:
            # If FeatureEngineeringSchema is used as alias, strip non-function key.
            args = {k: v for k, v in args.items() if k != "operation"}
        if tool_name in {"create_categorical_bands", "regroup_categorical_features"}:
            if "data_path" in args and "source_path" not in args:
                # Schema alias compatibility: data_path maps to source_path.
                args["source_path"] = args.pop("data_path")
            else:
                args.pop("data_path", None)
        elif "data_path" in args:
            args.pop("data_path")

        try:
            return self.tool_handlers[tool_name](**args)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            return json.dumps({"error": f"{tool_name} failed: {str(exc)}"}, indent=2)

    @staticmethod
    def _extract_tabular_path(user_message: str) -> Optional[str]:
        """Extract an explicit supported raw-input path from the user message."""
        path_match = RAW_INPUT_PATH_RE.search(user_message)
        if not path_match:
            return None
        raw_path = path_match.group(1)
        candidate = Path(raw_path)

        if candidate.is_absolute() or "/" in raw_path or raw_path.startswith("data/"):
            return raw_path

        candidate_paths = [
            candidate,
            Path("data/output") / raw_path,
            Path("data/input") / raw_path,
        ]
        for candidate_path in candidate_paths:
            if candidate_path.exists():
                return str(candidate_path)
        return raw_path

    @staticmethod
    def _is_schema_listing_request(user_message: str) -> bool:
        """Detect explicit requests to list columns or dataset schema details."""
        msg = user_message.lower()
        schema_hits = [
            "what columns",
            "what are the columns",
            "list columns",
            "show columns",
            "column names",
            "schema",
            "data types",
            "null counts",
        ]
        return any(hit in msg for hit in schema_hits)

    @staticmethod
    def _is_prepared_dataset_reference(user_message: str, explicit_path: Optional[str]) -> bool:
        """Detect references to the prepared analysis artifact by name or path."""
        msg = user_message.lower()
        if (
            "analysis_inforce" in msg
            or "prepared analysis" in msg
            or "prepared dataset" in msg
        ):
            return True
        return bool(explicit_path and Path(explicit_path).name in {"analysis_inforce.parquet", "analysis_inforce.csv"})

    @classmethod
    def _should_use_deterministic_schema_route(
        cls,
        user_message: str,
        explicit_path: Optional[str],
    ) -> bool:
        """Prefer deterministic schema output for schema questions and prepared-dataset profiling."""
        if cls._is_schema_listing_request(user_message):
            return True

        msg = user_message.lower()
        if "profile" not in msg:
            return False

        return cls._is_prepared_dataset_reference(user_message, explicit_path)

    @staticmethod
    def _normalize_column_name(name: str) -> str:
        """Normalize column names for case-insensitive matching."""
        normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
        return re.sub(r"_+", "_", normalized).strip("_")

    @staticmethod
    def _default_schema_target_path(user_message: str) -> str:
        """Choose the default dataset for schema inspection requests."""
        msg = user_message.lower()
        if "analysis_inforce" in msg or "prepared analysis" in msg or "prepared dataset" in msg:
            return str(resolve_prepared_analysis_path(CANONICAL_ANALYSIS_OUTPUT_PATH))
        return DEFAULT_RAW_INPUT_PATH

    @staticmethod
    def _extract_requested_schema_columns(user_message: str, available_columns: list[str]) -> list[str]:
        """Return explicitly requested columns mentioned in the user's schema question."""
        column_lookup = {
            DataStewardAgent._normalize_column_name(column): column for column in available_columns
        }
        requested_columns: list[str] = []

        for raw_token in re.split(r"[^A-Za-z0-9_]+", user_message):
            normalized = DataStewardAgent._normalize_column_name(raw_token)
            if not normalized:
                continue
            resolved = column_lookup.get(normalized)
            if resolved and resolved not in requested_columns:
                requested_columns.append(resolved)

        return requested_columns

    @staticmethod
    def _format_profile_summary(profile_json: str, data_path: str, user_message: str) -> str:
        """Render a deterministic schema summary from the profile tool JSON."""
        payload = json.loads(profile_json)
        if payload.get("error"):
            return f"Unable to profile `{data_path}`: {payload['error']}"

        columns = payload.get("columns", [])
        data_types = payload.get("data_types", {})
        feature_classification = payload.get("feature_classification", {})
        null_counts = payload.get("null_counts", {})
        requested_columns = DataStewardAgent._extract_requested_schema_columns(user_message, columns)
        columns_to_show = requested_columns or columns

        lines = []
        if requested_columns:
            lines.append(f"Requested columns for `{data_path}`:")
        else:
            lines.append(f"Columns, data types, feature classes, and null counts for `{data_path}`:")
        lines.append("")
        for column in columns_to_show:
            lines.append(
                f"- `{column}`: `{data_types.get(column, 'unknown')}`; "
                f"class: `{feature_classification.get(column, 'unknown')}`; "
                f"nulls: `{null_counts.get(column, 'unknown')}`"
            )
        return "\n".join(lines)

    def _deterministic_schema_route(self, user_message: str) -> Optional[str]:
        """Handle explicit schema/column inspection requests without model summarization."""
        explicit_path = self._extract_tabular_path(user_message)
        if not self._should_use_deterministic_schema_route(user_message, explicit_path):
            return None

        active_data_path = explicit_path or self._default_schema_target_path(user_message)
        self._emit_status("Data Steward: profiling the dataset schema.")
        profile_json = profile_dataset(data_path=active_data_path)
        return self._format_profile_summary(profile_json, active_data_path, user_message)

    @staticmethod
    def _normalize_banding_strategy(strategy_text: str) -> Optional[str]:
        """Map natural-language banding phrases to tool strategies."""
        normalized = re.sub(r"[-\s]+", " ", strategy_text.strip().lower())
        if normalized == "equal width":
            return "equal_width"
        if normalized in {"quantile", "quantiles", "equal quantile", "same quantile"}:
            return "quantiles"
        return None

    @staticmethod
    def _clean_requested_band_column(raw_column: str) -> str:
        """Remove banding-specific filler text from a requested column token."""
        cleaned = raw_column.strip(" `.,")
        cleaned = re.sub(r"^(?:the\s+)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+columns?$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip(" `.,")

    @classmethod
    def _resolve_requested_band_column(cls, raw_column: str, available_columns: list[str]) -> str:
        """Resolve a requested banding column against available dataset columns."""
        cleaned = cls._clean_requested_band_column(raw_column)
        if not available_columns:
            return re.sub(r"[^A-Za-z0-9]+", "_", cleaned).strip("_")

        column_lookup = {cls._normalize_column_name(column): column for column in available_columns}
        normalized_cleaned = cls._normalize_column_name(cleaned)
        if normalized_cleaned in column_lookup:
            return column_lookup[normalized_cleaned]

        underscored = re.sub(r"[^A-Za-z0-9]+", "_", cleaned).strip("_")
        return column_lookup.get(cls._normalize_column_name(underscored), underscored)

    @staticmethod
    def _load_bandable_columns(data_path: str) -> list[str]:
        """Load available source columns for case-insensitive band request resolution."""
        resolved_path = resolve_prepared_analysis_path(data_path)
        if not resolved_path.exists():
            return []
        try:
            return get_tabular_columns(str(resolved_path))
        except Exception:
            return []

    @staticmethod
    def _leftover_banding_tokens(user_message: str, matches: list[re.Match[str]]) -> list[str]:
        """Return non-band-request tokens left after removing parsed banding clauses."""
        fragments: list[str] = []
        cursor = 0
        for match in matches:
            fragments.append(user_message[cursor:match.start()])
            cursor = match.end()
        fragments.append(user_message[cursor:])
        leftover = " ".join(fragments).lower()
        return [
            token
            for token in re.findall(r"[a-z0-9_/-]+", leftover)
            if token not in IGNORABLE_BANDING_TOKENS
        ]

    @classmethod
    def _parse_deterministic_band_specs(
        cls,
        user_message: str,
        data_path: str,
    ) -> Optional[list[CategoricalBandSpec]]:
        """Parse explicit banding-only prompts into deterministic band specs."""
        lowered = user_message.lower()
        if "band" not in lowered:
            return None
        if any(blocker in lowered for blocker in MIXED_BANDING_BLOCKERS):
            return None

        available_columns = cls._load_bandable_columns(data_path)
        matches = list(BANDING_CLAUSE_RE.finditer(user_message))
        if not matches:
            return None

        leftover_tokens = cls._leftover_banding_tokens(user_message, matches)
        if leftover_tokens:
            return None

        band_specs: list[CategoricalBandSpec] = []
        for match in matches:
            strategy = cls._normalize_banding_strategy(match.group("strategy"))
            if strategy is None:
                return None

            source_column = cls._resolve_requested_band_column(match.group("column"), available_columns)
            band_specs.append(
                CategoricalBandSpec(
                    source_column=source_column,
                    strategy=strategy,
                    bins=int(match.group("bins")),
                )
            )

        return band_specs

    @staticmethod
    def _format_deterministic_banding_summary(result_json: str, source_path: str) -> str:
        """Render a consistent human summary for deterministic banding responses."""
        payload = json.loads(result_json)
        if payload.get("error"):
            return f"Unable to create categorical bands: {payload['error']}"

        operations = payload.get("operations", [])
        output_path = payload.get("output_path", CANONICAL_ANALYSIS_OUTPUT_PATH)
        lines = [
            f"Categorical banding complete. Output saved to `{output_path}`.",
            f"Source data: `{source_path}`.",
            "",
            "Created features:",
        ]

        strategy_labels = {
            "equal_width": "equal-width",
            "quantiles": "quantile",
            "custom": "custom",
        }
        for operation in operations:
            lines.append(
                f"- `{operation['new_column']}` from `{operation['source_column']}` "
                f"using `{operation.get('bins', 'N/A')}` {strategy_labels.get(operation['strategy'], operation['strategy'])} bins."
            )

        return "\n".join(lines)

    def _deterministic_banding_route(self, user_message: str) -> Optional[str]:
        """Handle pure banding prompts deterministically before the LLM path."""
        explicit_path = self._extract_tabular_path(user_message)
        working_message = user_message.replace(explicit_path, " ") if explicit_path else user_message
        active_data_path = explicit_path or DEFAULT_RAW_INPUT_PATH
        band_specs = self._parse_deterministic_band_specs(working_message, active_data_path)
        if band_specs is None:
            return None

        self._emit_status("Data Steward: applying deterministic categorical banding operations.")
        result_json = create_categorical_bands_batch(
            band_specs=band_specs,
            source_path=active_data_path,
            output_path=CANONICAL_ANALYSIS_OUTPUT_PATH,
        )
        return self._format_deterministic_banding_summary(result_json, active_data_path)

    def _execute_tool_with_context(
        self,
        tool_name: str,
        args: dict[str, Any],
        dataset_context: Optional[dict[str, Any]],
    ) -> str:
        if tool_name not in self.tool_handlers:
            return json.dumps({"error": f"Unknown tool: {tool_name}"}, indent=2)
        return self._execute_tool(tool_name, args)

    def _fallback_route(self, user_message: str) -> str:
        """Deterministic fallback when no API key is configured."""
        msg = user_message.lower()
        active_data_path = self._extract_tabular_path(user_message) or DEFAULT_RAW_INPUT_PATH
        active_sheet_name = None

        if "profile" in msg or "moc" in msg:
            self._emit_status("Data Steward: profiling the dataset and running actuarial data checks.")
            profile_json = profile_dataset(data_path=active_data_path, sheet_name=active_sheet_name)
            checks_json = run_actuarial_data_checks(data_path=active_data_path, sheet_name=active_sheet_name)
            profile = json.loads(profile_json)
            checks = json.loads(checks_json)
            status = checks.get("status", "UNKNOWN")

            # Build a readable markdown summary instead of dumping raw JSON.
            columns = profile.get("columns", [])
            dtypes = profile.get("data_types", {})
            nulls = profile.get("null_counts", {})
            classification = checks.get("feature_classification", {})

            col_table = "| Column | Type | Nulls | Classification |\n|---|---|---|---|\n"
            for col in columns:
                col_table += f"| `{col}` | {dtypes.get(col, '–')} | {nulls.get(col, '–')} | {classification.get(col, '–')} |\n"

            issues = checks.get("issues", [])
            issues_section = "None" if not issues else "\n".join(f"- {issue}" for issue in issues)

            return (
                f"## Dataset Profile: `{active_data_path}`\n\n"
                f"| Metric | Value |\n|---|---|\n"
                f"| Total Rows | {profile.get('total_rows', '–'):,} |\n"
                f"| Unique Policies | {profile.get('unique_policy_count', '–'):,} |\n"
                f"| Memory Usage | {profile.get('memory_usage_human', '–')} |\n"
                f"| Validation Status | **{status}** |\n\n"
                f"### Columns\n\n{col_table}\n"
                f"### Actuarial Data Checks\n\n"
                f"- MOC check: numeric type, range (0,1], and MAC==1 ⇒ MOC==1.0\n"
                f"- Issues: {issues_section}\n"
            )

        if "band" in msg:
            self._emit_status("Data Steward: creating categorical bands in the analysis dataset.")
            bins = 5 if "5" in msg else 4
            col_match = re.search(r"for the\s+([A-Za-z_][A-Za-z0-9_]*)\s+column", user_message, re.IGNORECASE)
            source_column = col_match.group(1) if col_match else "Issue_Age"
            source_path = active_data_path
            result = create_categorical_bands(
                source_column=source_column,
                strategy="equal_width",
                bins=bins,
                source_path=source_path,
                output_path=CANONICAL_ANALYSIS_OUTPUT_PATH,
                sheet_name=active_sheet_name,
            )
            return (
                f"Banding complete for `{source_column}` using equal-width `{bins}` bins.\n"
                f"Source data: `{source_path}`.\n"
                f"Transformation saved to `{CANONICAL_ANALYSIS_OUTPUT_PATH}`.\n"
                f"Tool JSON:\n{result}"
            )

        if "regroup" in msg:
            self._emit_status("Data Steward: regrouping categorical values into a derived feature.")
            column_match = re.search(
                r"for the\s+([A-Za-z_][A-Za-z0-9_]*)\s+column",
                user_message,
                re.IGNORECASE,
            )
            mapping_match = re.search(r"(\{.*\})", user_message)
            if not column_match or not mapping_match:
                return (
                    "To regroup a categorical column without OpenAI access, provide the source column "
                    "and a JSON mapping dictionary, for example: "
                    "`Regroup categories for the Risk_Class column using {\"Standard Plus\": \"Standard\"}`."
                )

            mapping_text = mapping_match.group(1)
            try:
                mapping_dict = json.loads(mapping_text)
            except json.JSONDecodeError:
                return "Mapping dictionary must be valid JSON when using deterministic regrouping fallback."

            source_column = column_match.group(1)
            source_path = str(resolve_prepared_analysis_path(CANONICAL_ANALYSIS_OUTPUT_PATH))
            if not Path(source_path).exists():
                source_path = active_data_path

            result = regroup_categorical_features(
                source_column=source_column,
                mapping_dict=mapping_dict,
                source_path=source_path,
                output_path=CANONICAL_ANALYSIS_OUTPUT_PATH,
                sheet_name=active_sheet_name if source_path == active_data_path else None,
            )
            return (
                f"Regrouping complete for `{source_column}`.\n"
                f"Source data: `{source_path}`.\n"
                f"Transformation saved to `{CANONICAL_ANALYSIS_OUTPUT_PATH}`.\n"
                f"Tool JSON:\n{result}"
            )

        return (
            "I can help with profiling, actuarial data checks, and categorical banding. "
            "Please specify a transformation request and dataset path when needed."
        )

    def run(self, user_message: str) -> str:
        """Handle a user message using OpenAI tool-calling."""
        deterministic_response = self._deterministic_schema_route(user_message)
        if deterministic_response is not None:
            return deterministic_response

        deterministic_banding_response = self._deterministic_banding_route(user_message)
        if deterministic_banding_response is not None:
            return deterministic_banding_response

        if not self.client:
            self._emit_status("Data Steward: OpenAI is unavailable, using deterministic local tooling.")
            return self._fallback_route(user_message)

        self._emit_status("Data Steward: requesting the next data-prep action from the model.")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        last_tool_result: Optional[str] = None
        seen_tool_calls: set[tuple[str, str]] = set()

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
                log_openai_error("Data Steward", "Tool-calling request", exc)
                self._emit_status(
                    "Data Steward: tool-calling is unavailable "
                    f"({openai_error_type(exc)}), falling back to deterministic logic."
                )
                return self._fallback_route(user_message)
            message = completion.choices[0].message
            tool_calls = message.tool_calls or []

            if not tool_calls:
                self._emit_status("Data Steward: returning a direct response without tool execution.")
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
                raw_args = tool_call.function.arguments or "{}"
                call_signature = (name, raw_args)
                if call_signature in seen_tool_calls:
                    # Prevent infinite tool-call loops on repeated identical calls.
                    if last_tool_result:
                        return last_tool_result
                    return (
                        "I hit a repeated tool-call loop. Please rephrase your request with "
                        "an explicit dataset path, e.g. `data/input/synthetic_inforce.csv` or `data/input/example.parquet`."
                    )
                seen_tool_calls.add(call_signature)

                args = json.loads(raw_args)
                self._emit_status(f"Data Steward: executing `{name}`.")
                tool_result = self._execute_tool(name, args)
                last_tool_result = tool_result
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

        if last_tool_result:
            return last_tool_result
        return (
            "Unable to complete request within tool-calling loop. "
            "Please provide an explicit dataset path and request (for example: "
            "'Profile data/input/synthetic_inforce.csv')."
        )

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
