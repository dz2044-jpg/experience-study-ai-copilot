"""Session-local artifact state for the unified copilot runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
