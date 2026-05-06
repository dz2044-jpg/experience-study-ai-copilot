"""Generate the canonical Experience Study AI Copilot architecture diagram.

The diagram is rendered twice from the same layout data:
- editable SVG written directly with SVG primitives
- high-resolution PNG drawn with Pillow

Pillow is used only for drawing the PNG. The SVG is not rasterized.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import math
import textwrap
from typing import Literal

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "architecture"
SVG_PATH = OUTPUT_DIR / "experience_study_ai_copilot_workflow.svg"
PNG_PATH = OUTPUT_DIR / "experience_study_ai_copilot_workflow.png"

WIDTH = 3400
HEIGHT = 1900
FONT_FAMILY = "Inter, Avenir Next, Segoe UI, Arial, sans-serif"

COLORS = {
    "background": "#07111f",
    "panel": "#0f1d2e",
    "panel_alt": "#142337",
    "tile": "#102238",
    "tile_alt": "#122a40",
    "text": "#f6f8fb",
    "muted": "#b6c2d2",
    "dim": "#8d9bad",
    "border": "#e7eef7",
    "soft_border": "#6d8199",
    "arrow": "#d9e6f2",
    "trust": "#f4b860",
    "trust_dark": "#241a08",
    "planned": "#aeb7c4",
    "planned_fill": "#1b2431",
    "input": "#2f8bd8",
    "deterministic": "#58c77b",
    "audit": "#f0a646",
    "ai": "#a888ff",
    "output": "#dce9f7",
}


TileStatus = Literal["implemented", "planned"]


@dataclass(frozen=True)
class Tile:
    title: str
    caption: str
    status: TileStatus = "implemented"


@dataclass(frozen=True)
class Group:
    number: int
    title: str
    caption: str
    x: int
    y: int
    w: int
    h: int
    accent: str
    tiles: tuple[Tile, ...]
    columns: int = 1


GROUP_Y = 430
GROUP_H = 900
GROUPS = (
    Group(
        1,
        "Input & Data Pipeline",
        "User requests become session-scoped workflow state.",
        90,
        GROUP_Y,
        520,
        GROUP_H,
        COLORS["input"],
        (
            Tile("Streamlit UI", "main.py app entry"),
            Tile("Chat + Explorer", "Chat input, sweep table, AI panel entry"),
            Tile("Dataset Handling", "CSV / Parquet / XLSX paths into prepared data"),
        ),
    ),
    Group(
        2,
        "Deterministic Analysis Pipeline",
        "Python tools own every actuarial calculation.",
        650,
        GROUP_Y,
        820,
        GROUP_H,
        COLORS["deterministic"],
        (
            Tile("Skill Package", "native_tools.py registry + schemas"),
            Tile("Profile / Schema / Validate", "io.py + validation.py"),
            Tile("Feature Logic", "Bands, regroups, filters"),
            Tile("Mortality A/E Sweep", "sweeps.py + ae_math.py confidence intervals"),
        ),
        columns=2,
    ),
    Group(
        3,
        "Artifact & Audit Pipeline",
        "Outputs are saved, hashed, and made traceable.",
        1510,
        GROUP_Y,
        620,
        GROUP_H,
        COLORS["audit"],
        (
            Tile("Prepared Dataset", "analysis_inforce.parquet"),
            Tile("Sweep + Visuals", "sweep_summary.csv + combined A/E HTML"),
            Tile("Methodology Log", "Append-only workflow events"),
            Tile("Manifest + Fingerprint", "Content hashes and freshness metadata"),
        ),
        columns=2,
    ),
    Group(
        4,
        "Grounded AI Interpretation Pipeline",
        "AI receives sanitized aggregate packets only.",
        2220,
        GROUP_Y,
        650,
        GROUP_H,
        COLORS["ai"],
        (
            Tile("AI Packet Builder", "ai_packets.py allowlisted fields"),
            Tile("Sanitization", "Sensitive dimensions + rare cohorts masked"),
            Tile("AI Orchestrator", "ai_orchestrator.py with fallback mode"),
            Tile("Interpret Actions", "Summarize sweep / explain cohort"),
        ),
        columns=2,
    ),
    Group(
        5,
        "Report / Output Pipeline",
        "Current interpretation plus planned handoff work.",
        2910,
        GROUP_Y,
        400,
        GROUP_H,
        COLORS["output"],
        (
            Tile("AI Panel", "Current: interpretation UI"),
            Tile("Report Drafting", "Overview + findings draft", "planned"),
            Tile("Editor & Export", "Editable Markdown/ZIP", "planned"),
            Tile("Style Reference Layer", "Tone guidance only", "planned"),
            Tile("Human actuarial sign-off", "Actuary final review"),
        ),
    ),
)

FOOTERS = (
    (
        "Implemented runtime and tools",
        "main.py, core/copilot_agent.py, core/fallback_planner.py, core/prerequisite_guard.py, "
        "core/session_state.py, skills/experience_study_skill/native_tools.py, schemas.py",
    ),
    (
        "Implemented deterministic, audit, and AI modules",
        "io.py, validation.py, feature_engineering.py, ae_math.py, sweeps.py, visualization.py, "
        "core/methodology_log.py, core/artifact_manifest.py, ai_packets.py, ai_sanitization.py, "
        "ai_validation.py, ai_fallbacks.py, ai_orchestrator.py",
    ),
)


def _wrap_chars(text: str, max_chars: int) -> list[str]:
    return textwrap.wrap(text, width=max_chars, break_long_words=False) or [text]


def _svg_attrs(attrs: dict[str, object | None]) -> str:
    return " ".join(
        f'{key.replace("_", "-")}="{escape(str(value), quote=True)}"'
        for key, value in attrs.items()
        if value is not None
    )


def _svg_tag(name: str, attrs: dict[str, object | None]) -> str:
    return f"<{name} {_svg_attrs(attrs)}/>"


def _svg_text(
    text: str,
    *,
    x: float,
    y: float,
    size: int,
    fill: str = COLORS["text"],
    weight: int | str = 400,
    anchor: str = "start",
    opacity: float = 1,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT_FAMILY}" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}" opacity="{opacity}">{escape(text)}</text>'
    )


def _svg_multiline_text(
    lines: list[str],
    *,
    x: float,
    y: float,
    size: int,
    fill: str = COLORS["text"],
    weight: int | str = 400,
    line_height: float = 1.2,
    anchor: str = "start",
) -> str:
    tspans = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else size * line_height
        tspans.append(
            f'<tspan x="{x:.1f}" dy="{dy:.1f}">{escape(line)}</tspan>'
        )
    return (
        f'<text y="{y:.1f}" font-family="{FONT_FAMILY}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        + "".join(tspans)
        + "</text>"
    )


def _svg_center_text(
    text: str,
    *,
    x: float,
    y: float,
    size: int,
    fill: str = COLORS["text"],
    weight: int | str = 700,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT_FAMILY}" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="middle" '
        f'dominant-baseline="middle" fill="{fill}">{escape(text)}</text>'
    )


def _tile_grid(group: Group) -> list[tuple[Tile, int, int, int, int]]:
    inner_x = group.x + 34
    inner_y = group.y + 188
    inner_w = group.w - 68
    inner_h = group.h - 240
    gap = 22
    columns = group.columns
    rows = math.ceil(len(group.tiles) / columns)
    tile_w = int((inner_w - gap * (columns - 1)) / columns)
    tile_h = int((inner_h - gap * (rows - 1)) / rows)
    placements = []
    for index, tile in enumerate(group.tiles):
        row = index // columns
        col = index % columns
        x = inner_x + col * (tile_w + gap)
        y = inner_y + row * (tile_h + gap)
        placements.append((tile, x, y, tile_w, tile_h))
    return placements


def _svg_group(group: Group) -> list[str]:
    parts: list[str] = []
    parts.append(
        _svg_tag(
            "rect",
            {
                "x": group.x,
                "y": group.y,
                "width": group.w,
                "height": group.h,
                "rx": 34,
                "fill": COLORS["panel"],
                "stroke": COLORS["border"],
                "stroke_width": 3,
            },
        )
    )
    parts.append(
        _svg_tag(
            "rect",
            {
                "x": group.x + 3,
                "y": group.y + 3,
                "width": group.w - 6,
                "height": 16,
                "rx": 8,
                "fill": group.accent,
            },
        )
    )
    circle_x = group.x + 52
    circle_y = group.y + 74
    parts.append(
        _svg_tag(
            "circle",
            {
                "cx": circle_x,
                "cy": circle_y,
                "r": 34,
                "fill": group.accent,
                "stroke": COLORS["border"],
                "stroke_width": 3,
            },
        )
    )
    parts.append(
        _svg_center_text(
            str(group.number),
            x=circle_x,
            y=circle_y + 1,
            size=30,
            fill=COLORS["background"],
            weight=850,
        )
    )
    title_lines = _wrap_chars(group.title, 26 if group.w < 600 else 42)
    parts.append(
        _svg_multiline_text(
            title_lines,
            x=group.x + 104,
            y=group.y + 58,
            size=30,
            fill=COLORS["text"],
            weight=850,
            line_height=1.12,
        )
    )
    parts.append(
        _svg_multiline_text(
            _wrap_chars(group.caption, 38 if group.w < 600 else 54),
            x=group.x + 36,
            y=group.y + 142,
            size=21,
            fill=COLORS["muted"],
            weight=500,
            line_height=1.2,
        )
    )
    for tile, x, y, w, h in _tile_grid(group):
        planned = tile.status == "planned"
        parts.append(
            _svg_tag(
                "rect",
                {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "rx": 20,
                    "fill": COLORS["planned_fill"] if planned else COLORS["tile"],
                    "stroke": COLORS["planned"] if planned else group.accent,
                    "stroke_width": 2.5,
                    "stroke_dasharray": "12 9" if planned else None,
                    "opacity": 0.96,
                },
            )
        )
        parts.append(
            _svg_multiline_text(
                _wrap_chars(tile.title, 18 if group.columns > 1 else 24),
                x=x + 22,
                y=y + 43,
                size=24,
                fill=COLORS["planned"] if planned else COLORS["text"],
                weight=800,
                line_height=1.12,
            )
        )
        parts.append(
            _svg_multiline_text(
                _wrap_chars(tile.caption, 24 if group.columns > 1 else 31),
                x=x + 22,
                y=y + 86,
                size=18,
                fill=COLORS["muted"],
                weight=500,
                line_height=1.2,
            )
        )
    return parts


def _svg_arrow(x1: float, y1: float, x2: float, y2: float, color: str) -> str:
    return _svg_tag(
        "path",
        {
            "d": f"M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}",
            "fill": "none",
            "stroke": color,
            "stroke_width": 7,
            "stroke_linecap": "round",
            "marker_end": "url(#arrowhead)",
        },
    )


def build_svg() -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
            f'aria-label="End-to-end framework for an auditable AI-powered experience study copilot">'
        ),
        "<defs>",
        (
            '<marker id="arrowhead" markerWidth="18" markerHeight="14" '
            'refX="16" refY="7" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M 0 0 L 18 7 L 0 14 z" fill="{COLORS["arrow"]}"/>'
            "</marker>"
        ),
        "</defs>",
        _svg_tag("rect", {"width": WIDTH, "height": HEIGHT, "fill": COLORS["background"]}),
        _svg_center_text(
            "End-to-end framework for an auditable AI-powered experience study copilot",
            x=WIDTH / 2,
            y=82,
            size=52,
            fill=COLORS["text"],
            weight=850,
        ),
        _svg_center_text(
            "Deterministic actuarial tools generate audited artifacts before AI interprets sanitized cohort-level packets.",
            x=WIDTH / 2,
            y=145,
            size=29,
            fill=COLORS["muted"],
            weight=500,
        ),
    ]

    deterministic_x = GROUPS[0].x - 26
    deterministic_y = GROUP_Y - 36
    deterministic_w = GROUPS[2].x + GROUPS[2].w - deterministic_x + 18
    deterministic_h = GROUP_H + 70
    parts.append(
        _svg_tag(
            "rect",
            {
                "x": deterministic_x,
                "y": deterministic_y,
                "width": deterministic_w,
                "height": deterministic_h,
                "rx": 40,
                "fill": "#0a1a2a",
                "stroke": "#27435e",
                "stroke_width": 2,
                "opacity": 0.58,
            },
        )
    )
    parts.append(
        _svg_text(
            "Deterministic before AI",
            x=deterministic_x + 34,
            y=deterministic_y - 16,
            size=24,
            fill=COLORS["deterministic"],
            weight=850,
        )
    )

    trust_x = 2175
    parts.append(
        _svg_tag(
            "path",
            {
                "d": f"M {trust_x} 360 L {trust_x} 1368",
                "fill": "none",
                "stroke": COLORS["trust"],
                "stroke_width": 9,
                "stroke_dasharray": "24 18",
                "stroke_linecap": "round",
            },
        )
    )
    callout_x = 1720
    callout_y = 218
    callout_w = 930
    callout_h = 120
    parts.append(
        _svg_tag(
            "rect",
            {
                "x": callout_x,
                "y": callout_y,
                "width": callout_w,
                "height": callout_h,
                "rx": 28,
                "fill": COLORS["trust_dark"],
                "stroke": COLORS["trust"],
                "stroke_width": 3,
            },
        )
    )
    parts.append(
        _svg_center_text(
            "Trust Boundary",
            x=callout_x + callout_w / 2,
            y=callout_y + 34,
            size=30,
            fill=COLORS["trust"],
            weight=850,
        )
    )
    parts.append(
        _svg_center_text(
            "AI receives only sanitized cohort-level aggregate artifacts",
            x=callout_x + callout_w / 2,
            y=callout_y + 73,
            size=24,
            fill=COLORS["text"],
            weight=700,
        )
    )
    parts.append(
        _svg_center_text(
            "never raw policy/claim-level data.",
            x=callout_x + callout_w / 2,
            y=callout_y + 103,
            size=23,
            fill=COLORS["muted"],
            weight=500,
        )
    )

    for group in GROUPS:
        parts.extend(_svg_group(group))

    arrow_y = GROUP_Y + GROUP_H / 2
    for left, right in zip(GROUPS, GROUPS[1:]):
        color = COLORS["trust"] if left.number == 3 else COLORS["arrow"]
        parts.append(_svg_arrow(left.x + left.w + 18, arrow_y, right.x - 20, arrow_y, color))

    footer_y = 1460
    for index, (title, body) in enumerate(FOOTERS):
        x = 110 + index * 1610
        y = footer_y
        w = 1545
        h = 220
        parts.append(
            _svg_tag(
                "rect",
                {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "rx": 26,
                    "fill": COLORS["panel_alt"],
                    "stroke": COLORS["soft_border"],
                    "stroke_width": 2,
                },
            )
        )
        parts.append(
            _svg_text(
                title,
                x=x + 36,
                y=y + 50,
                size=24,
                fill=COLORS["text"],
                weight=850,
            )
        )
        for line_index, line in enumerate(_wrap_chars(body, 88)):
            parts.append(
                _svg_text(
                    line,
                    x=x + 36,
                    y=y + 94 + line_index * 34,
                    size=21,
                    fill=COLORS["muted"],
                    weight=500,
                )
            )
    parts.append(
        _svg_center_text(
            "Status: current deterministic workflow and AI interpretation are implemented; report handoff capabilities are roadmap items.",
            x=WIDTH / 2,
            y=1810,
            size=22,
            fill=COLORS["dim"],
            weight=500,
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _wrap_pixels(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if _text_size(draw, candidate, font)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    *,
    anchor: str | None = None,
) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _draw_multiline(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str],
    font: ImageFont.ImageFont,
    fill: str,
    *,
    line_gap: int,
) -> None:
    x, y = xy
    for index, line in enumerate(lines):
        _draw_text(draw, (x, y + index * line_gap), line, font, fill)


def _draw_center_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    x: float,
    y: float,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    _draw_text(draw, (x, y), text, font, fill, anchor="mm")


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: str,
    width: int,
) -> None:
    draw.line((*start, *end), fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 24
    left = (
        end[0] - size * math.cos(angle - math.pi / 7),
        end[1] - size * math.sin(angle - math.pi / 7),
    )
    right = (
        end[0] - size * math.cos(angle + math.pi / 7),
        end[1] - size * math.sin(angle + math.pi / 7),
    )
    draw.polygon([end, left, right], fill=fill)


def _draw_dashed_vertical(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y1: int,
    y2: int,
    dash: int,
    gap: int,
    fill: str,
    width: int,
) -> None:
    y = y1
    while y < y2:
        draw.line((x, y, x, min(y + dash, y2)), fill=fill, width=width)
        y += dash + gap


def _draw_dashed_rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    radius: int,
    fill: str,
    outline: str,
    width: int,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill)
    x1, y1, x2, y2 = box
    dash = 14
    gap = 10
    for x in range(x1 + radius, x2 - radius, dash + gap):
        draw.line((x, y1, min(x + dash, x2 - radius), y1), fill=outline, width=width)
        draw.line((x, y2, min(x + dash, x2 - radius), y2), fill=outline, width=width)
    for y in range(y1 + radius, y2 - radius, dash + gap):
        draw.line((x1, y, x1, min(y + dash, y2 - radius)), fill=outline, width=width)
        draw.line((x2, y, x2, min(y + dash, y2 - radius)), fill=outline, width=width)


def _draw_group(draw: ImageDraw.ImageDraw, group: Group) -> None:
    draw.rounded_rectangle(
        (group.x, group.y, group.x + group.w, group.y + group.h),
        radius=34,
        fill=COLORS["panel"],
        outline=COLORS["border"],
        width=3,
    )
    draw.rounded_rectangle(
        (group.x + 3, group.y + 3, group.x + group.w - 3, group.y + 19),
        radius=8,
        fill=group.accent,
    )
    circle_x = group.x + 52
    circle_y = group.y + 74
    draw.ellipse(
        (circle_x - 34, circle_y - 34, circle_x + 34, circle_y + 34),
        fill=group.accent,
        outline=COLORS["border"],
        width=3,
    )
    _draw_center_text(
        draw,
        str(group.number),
        x=circle_x,
        y=circle_y + 1,
        font=_font(30, bold=True),
        fill=COLORS["background"],
    )
    title_font = _font(30, bold=True)
    title_lines = _wrap_pixels(draw, group.title, title_font, group.w - 130)
    _draw_multiline(
        draw,
        (group.x + 104, group.y + 42),
        title_lines,
        title_font,
        COLORS["text"],
        line_gap=34,
    )
    caption_font = _font(21)
    caption_lines = _wrap_pixels(draw, group.caption, caption_font, group.w - 72)
    _draw_multiline(
        draw,
        (group.x + 36, group.y + 122),
        caption_lines,
        caption_font,
        COLORS["muted"],
        line_gap=26,
    )
    for tile, x, y, w, h in _tile_grid(group):
        planned = tile.status == "planned"
        box = (x, y, x + w, y + h)
        if planned:
            _draw_dashed_rounded_rect(
                draw,
                box,
                radius=20,
                fill=COLORS["planned_fill"],
                outline=COLORS["planned"],
                width=3,
            )
        else:
            draw.rounded_rectangle(
                box,
                radius=20,
                fill=COLORS["tile"],
                outline=group.accent,
                width=3,
            )
        title_font = _font(24, bold=True)
        title_lines = _wrap_pixels(draw, tile.title, title_font, w - 44)
        _draw_multiline(
            draw,
            (x + 22, y + 26),
            title_lines,
            title_font,
            COLORS["planned"] if planned else COLORS["text"],
            line_gap=29,
        )
        caption_font = _font(19)
        caption_lines = _wrap_pixels(draw, tile.caption, caption_font, w - 44)
        _draw_multiline(
            draw,
            (x + 22, y + 82),
            caption_lines,
            caption_font,
            COLORS["muted"],
            line_gap=24,
        )


def build_png() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["background"])
    draw = ImageDraw.Draw(image)

    _draw_center_text(
        draw,
        "End-to-end framework for an auditable AI-powered experience study copilot",
        x=WIDTH / 2,
        y=82,
        font=_font(52, bold=True),
        fill=COLORS["text"],
    )
    _draw_center_text(
        draw,
        "Deterministic actuarial tools generate audited artifacts before AI interprets sanitized cohort-level packets.",
        x=WIDTH / 2,
        y=145,
        font=_font(29),
        fill=COLORS["muted"],
    )

    deterministic_x = GROUPS[0].x - 26
    deterministic_y = GROUP_Y - 36
    deterministic_w = GROUPS[2].x + GROUPS[2].w - deterministic_x + 18
    deterministic_h = GROUP_H + 70
    draw.rounded_rectangle(
        (
            deterministic_x,
            deterministic_y,
            deterministic_x + deterministic_w,
            deterministic_y + deterministic_h,
        ),
        radius=40,
        fill="#0a1a2a",
        outline="#27435e",
        width=2,
    )
    _draw_text(
        draw,
        (deterministic_x + 34, deterministic_y - 42),
        "Deterministic before AI",
        _font(24, bold=True),
        COLORS["deterministic"],
    )

    trust_x = 2175
    _draw_dashed_vertical(
        draw,
        x=trust_x,
        y1=360,
        y2=1368,
        dash=24,
        gap=18,
        fill=COLORS["trust"],
        width=9,
    )
    callout_x = 1720
    callout_y = 218
    callout_w = 930
    callout_h = 120
    draw.rounded_rectangle(
        (callout_x, callout_y, callout_x + callout_w, callout_y + callout_h),
        radius=28,
        fill=COLORS["trust_dark"],
        outline=COLORS["trust"],
        width=3,
    )
    _draw_center_text(
        draw,
        "Trust Boundary",
        x=callout_x + callout_w / 2,
        y=callout_y + 34,
        font=_font(30, bold=True),
        fill=COLORS["trust"],
    )
    _draw_center_text(
        draw,
        "AI receives only sanitized cohort-level aggregate artifacts",
        x=callout_x + callout_w / 2,
        y=callout_y + 73,
        font=_font(24, bold=True),
        fill=COLORS["text"],
    )
    _draw_center_text(
        draw,
        "never raw policy/claim-level data.",
        x=callout_x + callout_w / 2,
        y=callout_y + 103,
        font=_font(23),
        fill=COLORS["muted"],
    )

    for group in GROUPS:
        _draw_group(draw, group)

    arrow_y = GROUP_Y + GROUP_H / 2
    for left, right in zip(GROUPS, GROUPS[1:]):
        color = COLORS["trust"] if left.number == 3 else COLORS["arrow"]
        _draw_arrow(
            draw,
            (left.x + left.w + 18, arrow_y),
            (right.x - 20, arrow_y),
            fill=color,
            width=7,
        )

    footer_y = 1460
    for index, (title, body) in enumerate(FOOTERS):
        x = 110 + index * 1610
        y = footer_y
        w = 1545
        h = 220
        draw.rounded_rectangle(
            (x, y, x + w, y + h),
            radius=26,
            fill=COLORS["panel_alt"],
            outline=COLORS["soft_border"],
            width=2,
        )
        _draw_text(draw, (x + 36, y + 30), title, _font(24, bold=True), COLORS["text"])
        body_font = _font(21)
        body_lines = _wrap_pixels(draw, body, body_font, w - 72)
        _draw_multiline(
            draw,
            (x + 36, y + 78),
            body_lines,
            body_font,
            COLORS["muted"],
            line_gap=32,
        )
    _draw_center_text(
        draw,
        "Status: current deterministic workflow and AI interpretation are implemented; report handoff capabilities are roadmap items.",
        x=WIDTH / 2,
        y=1810,
        font=_font(22),
        fill=COLORS["dim"],
    )
    return image


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(build_svg(), encoding="utf-8")
    build_png().save(PNG_PATH)
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {PNG_PATH}")


if __name__ == "__main__":
    main()
