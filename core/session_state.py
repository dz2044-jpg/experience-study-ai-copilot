"""Session-local artifact state for the unified copilot runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.artifact_manifest import read_artifact_manifest
from core.methodology_log import read_methodology_log


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
    methodology_log_path: Path | None = None
    artifact_manifest_path: Path | None = None
    audit_ready: bool = False
    latest_state_fingerprint: str | None = None

    @property
    def output_dir(self) -> Path:
        return self.output_base_dir / self.session_id

    @property
    def audit_dir(self) -> Path:
        return self.output_dir / "audit"

    @property
    def default_methodology_log_path(self) -> Path:
        return self.audit_dir / "methodology_log.json"

    @property
    def default_artifact_manifest_path(self) -> Path:
        return self.audit_dir / "artifact_manifest.json"

    @staticmethod
    def _audit_files_are_ready(methodology_log_path: Path, artifact_manifest_path: Path) -> bool:
        if not methodology_log_path.exists() or not artifact_manifest_path.exists():
            return False
        try:
            read_methodology_log(methodology_log_path)
            read_artifact_manifest(artifact_manifest_path)
        except (OSError, ValueError):
            return False
        return True

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
        methodology_log_path = self.methodology_log_path or self.default_methodology_log_path
        artifact_manifest_path = self.artifact_manifest_path or self.default_artifact_manifest_path
        if methodology_log_path.exists():
            self.methodology_log_path = methodology_log_path
        if artifact_manifest_path.exists():
            self.artifact_manifest_path = artifact_manifest_path
        self.audit_ready = self._audit_files_are_ready(
            methodology_log_path,
            artifact_manifest_path,
        )

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

    def apply_audit_result(
        self,
        *,
        methodology_log_path: Path | None = None,
        artifact_manifest_path: Path | None = None,
        latest_state_fingerprint: str | None = None,
    ) -> bool:
        changed = False
        if methodology_log_path and self.methodology_log_path != methodology_log_path:
            self.methodology_log_path = methodology_log_path
            changed = True
        if artifact_manifest_path and self.artifact_manifest_path != artifact_manifest_path:
            self.artifact_manifest_path = artifact_manifest_path
            changed = True
        if latest_state_fingerprint and self.latest_state_fingerprint != latest_state_fingerprint:
            self.latest_state_fingerprint = latest_state_fingerprint
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
            f"- audit_ready: {self.audit_ready}",
            f"- methodology_log_path: {self.methodology_log_path or 'None'}",
            f"- artifact_manifest_path: {self.artifact_manifest_path or 'None'}",
            f"- latest_state_fingerprint: {self.latest_state_fingerprint or 'None'}",
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
            "audit_ready": self.audit_ready,
            "methodology_log_path": (
                str(self.methodology_log_path) if self.methodology_log_path else None
            ),
            "artifact_manifest_path": (
                str(self.artifact_manifest_path) if self.artifact_manifest_path else None
            ),
            "latest_state_fingerprint": self.latest_state_fingerprint,
        }
