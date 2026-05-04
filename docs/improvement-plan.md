# Experience Study AI Copilot Improvement Plan

Last updated: May 3, 2026

This is the canonical repository-tracked improvement plan for Experience Study AI Copilot. It replaces the local planning copy in `Downloads` as the source of truth for implementation sequencing, acceptance gates, and future PR scope.

## Purpose

Experience Study AI Copilot is an actuarial workflow copilot designed to accelerate first-draft experience study analysis.

It helps with:

- Deterministic A/E workflow execution.
- Interpretation of validated outputs.
- Report section drafting.
- Methodology documentation.
- Workflow handoff through editable and exportable artifacts.

It does not replace:

- Actuarial peer review.
- Final actuarial sign-off.
- Assumption setting.
- Pricing decisions.
- Underwriting strategy recommendations.

The actuary remains the ultimate reviewer, owner, and signatory.

## Current Status

Completed:

- The project has been renamed conceptually to Experience Study AI Copilot.
- The product direction has moved from a technical skill demo to an actuarial AI workflow copilot.
- The existing app has deterministic workflow capabilities:
  - Profile dataset.
  - Inspect schema.
  - Validate actuarial data.
  - Create bands and regroups.
  - Run dimensional A/E sweep.
  - Generate combined visualization report.
- README now reflects the Experience Study AI Copilot positioning.
- Governance docs now exist for product vision, trust boundary, data privacy, and roadmap.
- Golden workflow tests have been strengthened around artifacts, output columns, known A/E values, and stable public tool names.

Important product framing:

- Python owns calculations.
- AI owns interpretation and drafting.
- Past reports guide style only.
- Humans own final judgment and sign-off.

## Progress Tracker

| PR | Workstream | Status |
| --- | --- | --- |
| PR 1 | Governance Docs, Privacy, and README Polish | Completed in current working branch |
| PR 2 | Golden Workflow Test Hardening | Completed in current working branch |
| PR 3 | Small Runtime Extraction | Completed in current working branch |
| PR 4 | Deterministic Tool Module Split | Planned |
| PR 5 | Methodology Log and Artifact Manifest | Planned |
| PR 6 | AI Packet, Sanitization, Validation, and Fallback Layer | Planned |
| PR 7 | AI Interpretation UI | Planned |
| PR 8 | Report Drafting MVP | Planned |
| PR 9 | Human Editor and Export Package | Planned |
| PR 10 | Prior Report Style Reference Layer | Planned |

## Guiding Principles

1. Protect the working app first.
   Do not start large refactors or AI expansion without golden workflow tests.

2. Separate refactor work from new behavior.
   Behavior-preserving module splits should not be mixed with methodology logging, AI features, or export features in the same PR.

3. Keep deterministic outputs stable.
   A/E calculations, output columns, artifact names, and public tool names should not change unless intentionally versioned.

4. Make privacy and auditability testable.
   Do not rely only on documentation. Add tests proving that raw row-level data is not sent to the LLM.

5. Add AI only after deterministic artifacts are clean and traceable.
   AI should interpret validated outputs, not create actuarial calculations.

6. Do not over-split modules too early.
   Extract the clearest seams first. Avoid creating tiny abstract modules just to make the repo look organized.

## Phase 1 - Governance Docs, Privacy, and README Polish

Goal: make the product story clear, credible, and enterprise-ready.

Add or update:

- `README.md`.
- `docs/product-vision.md`.
- `docs/trust-boundary.md`.
- `docs/data-privacy.md`.
- `docs/roadmap.md`.

README should explain:

- What Experience Study AI Copilot is.
- What workflow it supports.
- What the deterministic layer does.
- What the AI layer does.
- What the AI layer does not do.
- How humans remain responsible for final judgment.
- How to run tests.
- How to launch the app.

Acceptance gate:

- README updated away from old Experience Study Skill positioning.
- Product boundary is clear.
- Privacy boundary is explicit.
- No code behavior changes required.

## Phase 2 - Golden Workflow Test Hardening

Goal: protect the existing deterministic workflow before refactoring.

Build on the existing full-pipeline test in `tests/test_unified_copilot.py`.

Strengthen tests for:

1. Profile dataset.
2. Inspect schema.
3. Validate actuarial data.
4. Create categorical bands or regroup features.
5. Run dimensional A/E sweep.
6. Generate combined visualization report.

Expected artifact checks:

- Prepared dataset parquet exists.
- Sweep summary CSV exists.
- Combined visualization HTML exists.
- Session state correctly tracks prepared dataset.
- Session state correctly tracks latest sweep.
- Session state correctly tracks latest visualization.

Expected output checks:

- Required A/E columns exist.
- Expected confidence interval columns exist.
- Selected known A/E values match current baseline.
- Selected known row counts match current baseline.
- Public tool names remain unchanged.

Snapshot checks:

- Record expected CSV columns from the golden sweep.
- Record a few known A/E values from the sample dataset.
- Record expected artifact naming conventions.

Acceptance gate:

- All existing tests pass.
- New golden workflow tests pass.
- Deterministic outputs are unchanged.
- No public tool names changed.

## Phase 3 - Small Runtime Extraction

Status: Completed in current working branch.

Goal: reduce the size and complexity of `core/copilot_agent.py` without changing behavior.

Extract only the clearest seams first:

- `core/session_state.py`.
- `core/prerequisite_guard.py`.
- `core/fallback_planner.py`.
- `core/response_formatter.py`.

Possible later extraction targets:

- `core/intent_parser.py`.
- `core/tool_runner.py`.

Acceptance gate:

- All tests pass.
- Existing behavior unchanged.
- Existing user-facing output unchanged unless intentionally improved and approved.
- Public tool names unchanged.
- No deterministic output changes.

## Phase 4 - Deterministic Tool Module Split

Goal: split `skills/experience_study_skill/native_tools.py` into auditable deterministic modules while preserving behavior.

Proposed structure:

- `skills/experience_study_skill/io.py`.
- `skills/experience_study_skill/validation.py`.
- `skills/experience_study_skill/feature_engineering.py`.
- `skills/experience_study_skill/ae_math.py`.
- `skills/experience_study_skill/sweeps.py`.
- `skills/experience_study_skill/visualization.py`.
- `skills/experience_study_skill/native_tools.py`.

Important:

- `native_tools.py` should remain as a compatibility layer.
- Existing public function names must remain available.
- Artifact names, output columns, and known deterministic values must remain stable.
- Do not add methodology logging in this phase.

Acceptance gate:

- All tests pass.
- Golden workflow outputs unchanged.
- Public tool names unchanged.
- Artifact names unchanged.
- Output CSV columns unchanged.
- Known A/E values unchanged.

## Phase 5 - Methodology Log and Artifact Manifest

Goal: add auditability after deterministic tools are cleanly modularized.

New files:

- `core/methodology_log.py`.
- `core/artifact_manifest.py`.

Methodology event model:

```python
@dataclass
class MethodologyEvent:
    step_name: str
    tool_name: str
    input_path: str | None
    output_path: str | None
    parameters: dict[str, object]
    timestamp: str
```

Track events for:

- Source dataset profiled.
- Schema inspected.
- Validation checks run.
- Age bands created.
- Categorical regrouping applied.
- Dimensional sweep run.
- Filters applied.
- Selected columns used.
- Sweep depth used.
- Ranking metric selected.
- Minimum MAC threshold applied.
- Visualization generated.

Artifact manifest should track:

- Artifact type.
- Path.
- Created timestamp.
- Content hash.
- Source artifact relationship.
- Generating tool.
- Relevant parameters.

Fingerprint inputs should include:

- Source artifact SHA256.
- Selected columns.
- Filters.
- Sweep depth.
- `sort_by`.
- `min_mac`.
- Packet schema version.
- Skill version, when applicable.

Acceptance gate:

- Methodology log is created during deterministic workflow.
- Artifact manifest is created or updated.
- Existing deterministic outputs unchanged.
- Tests verify methodology events are recorded for key workflow steps.
- Tests verify content hash changes when source artifact changes.

## Phase 6 - AI Packet, Sanitization, Validation, and Fallback Layer

Goal: create a safe AI foundation before adding AI buttons to the UI.

New files:

- `skills/experience_study_skill/actions/summarize_sweep.md`.
- `skills/experience_study_skill/actions/explain_cohort.md`.
- `skills/experience_study_skill/actions/compare_cohorts.md`.
- `skills/experience_study_skill/actions/analyze_count_amount_divergence.md`.
- `skills/experience_study_skill/ai_models.py`.
- `skills/experience_study_skill/ai_skill_loader.py`.
- `skills/experience_study_skill/ai_skill_renderer.py`.
- `skills/experience_study_skill/ai_packets.py`.
- `skills/experience_study_skill/ai_sanitization.py`.
- `skills/experience_study_skill/ai_baselines.py`.
- `skills/experience_study_skill/ai_validation.py`.
- `skills/experience_study_skill/ai_fallbacks.py`.
- `skills/experience_study_skill/ai_orchestrator.py`.

Use allowlist-based sanitization. Do not rely only on a denylist of sensitive columns.

Allowed AI packet fields should align with actual app output column names:

- `Dimensions`.
- `Dimension_Columns`.
- `Sum_MAC`.
- `Sum_MEC`.
- `Sum_MOC`.
- `Sum_MAF`.
- `Sum_MEF`.
- `AE_Ratio_Count`.
- `AE_Ratio_Amount`.
- `AE_Count_CI_Lower`.
- `AE_Count_CI_Upper`.
- `AE_Amount_CI_Lower`.
- `AE_Amount_CI_Upper`.
- `depth`.
- `filters`.
- `selected_columns`.
- `sort_by`.
- `min_mac`.

Strong privacy rule:

- The LLM should only receive aggregated cohort-level metrics.
- It should never receive raw row-level data.

Rare-cohort masking:

- Suppress detailed cohort labels when cohort volume is too small.
- Mark rows as low credibility.
- Prevent AI from over-interpreting low-credibility results.
- Allow only aggregate or cautionary discussion.

AI validation layer should block unsupported claims:

- Causal conclusions.
- Pricing recommendations.
- Underwriting strategy recommendations.
- Assumption changes.
- Final actuarial conclusions.

Acceptance gate:

- AI packet construction works from latest sweep artifact.
- Sanitization allowlist works.
- Rare-cohort masking works.
- No raw row-level data reaches the packet.
- Fallback response works when LLM is unavailable.
- Tests cover privacy behavior.

## Phase 7 - AI Interpretation UI

Goal: add MVP AI interpretation to Streamlit.

First AI actions:

- `summarize_sweep`.
- `explain_cohort`.

UI section:

- AI Interpretation Panel.

Buttons:

- Summarize Latest Sweep.
- Explain Top Cohort.
- Explain Selected Cohort.

Response should show:

- Summary text.
- Key findings.
- Caution flags.
- Next review steps.
- Evidence references.
- Source mode: LLM or fallback.
- Stale/fingerprint status.

Acceptance gate:

- AI panel only enables when required artifacts exist.
- AI response shows source mode.
- AI response shows evidence refs.
- AI response shows stale/fingerprint status.
- Existing deterministic workflow remains unaffected.

## Phase 8 - Report Drafting MVP

Goal: draft first report-ready sections using deterministic outputs and methodology events.

New skill files:

- `skills/experience_study_skill/actions/draft_mortality_overview.md`.
- `skills/experience_study_skill/actions/draft_key_findings.md`.
- `skills/experience_study_skill/actions/draft_methodology_appendix.md`.

Report packet should consume:

- Latest sweep artifact.
- Visualization artifact path.
- Top elevated cohorts.
- Top below-expected cohorts.
- Uncertainty notes.
- Methodology events.
- Artifact manifest.
- Approved style snippets, if available later.

Draft sections:

1. Mortality Overview.
2. Key Findings.
3. Methodology and Audit Trail.

Acceptance gate:

- Draft sections are generated from deterministic packet only.
- Methodology appendix uses actual methodology events.
- Draft does not invent assumptions, causes, or recommendations.
- Draft contains evidence refs or source artifact references.
- Fallback draft works if LLM is unavailable.

## Phase 9 - Human Editor and Export Package

Goal: turn AI-generated sections into an editable workflow handoff.

UI additions:

- Editable text area for mortality overview.
- Editable text area for key findings.
- Editable text area for methodology appendix.
- Session-state persistence for edited text.

Export options:

- Download edited section as Markdown.
- Download full artifact package as ZIP.
- Later: export as Word.

New file:

- `core/export_package.py`.

ZIP contents:

```text
experience_study_report_package.zip
  report_sections/
    mortality_overview.md
    key_findings.md
    methodology_appendix.md

  data/
    sweep_summary.csv

  visuals/
    combined_ae_report.html

  audit/
    methodology_log.json
    artifact_manifest.json
```

Acceptance gate:

- User edits are preserved in session state.
- Markdown export works.
- ZIP export works.
- ZIP includes expected files.
- Audit files are included.
- Export does not include raw sensitive input data unless explicitly designed and approved.

## Phase 10 - Prior Report Style Reference Layer

Goal: use prior reports for style, structure, and tone only.

Important confidentiality rule:

- Use only synthetic examples or explicitly approved internal examples.

New folder:

- `data/reference/report_style/mortality_overview_example.md`.
- `data/reference/report_style/key_findings_example.md`.
- `data/reference/report_style/credibility_commentary_example.md`.

Prompt rule:

- Use prior reports only to understand structure, tone, section ordering, and phrasing style.
- Do not copy prior-year numbers, prior conclusions, prior drivers, or verbatim language unless explicitly approved.
- Prior reports are style references, not factual sources and not text templates.

Acceptance gate:

- Style examples are synthetic or approved.
- Prompt includes no-verbatim-copy rule.
- AI output is based on current deterministic packet, not prior report facts.
- Tests or review checks confirm prior-year numbers are not copied into current report unless present in current packet.

## Updated PR Plan

1. Governance Docs, Privacy, and README Polish.
   Product story, privacy rules, and human sign-off boundary.

2. Golden Workflow Test Hardening.
   Stronger deterministic workflow tests before refactor work.

3. Small Runtime Extraction.
   Behavior-preserving extraction from the copilot runtime.

4. Deterministic Tool Module Split.
   Behavior-preserving split of deterministic tools into focused modules.

5. Methodology Log and Artifact Manifest.
   Audit events and content-hash based artifact tracking.

6. AI Packet, Sanitization, Validation, and Fallback Layer.
   Allowlisted cohort-level packets, privacy tests, and fallback behavior.

7. AI Interpretation UI.
   Streamlit AI interpretation panel gated by deterministic artifacts.

8. Report Drafting MVP.
   Draft report sections from deterministic packets and methodology events.

9. Human Editor and Export Package.
   Editable text and exportable handoff package.

10. Prior Report Style Reference Layer.
    Style-only prior report examples with no-verbatim-copy rules.

## Final MVP Definition

Existing deterministic capabilities:

1. Profile dataset.
2. Inspect schema.
3. Validate actuarial data.
4. Engineer bands and regroups.
5. Run dimensional A/E sweep.
6. Generate combined visualization report.

New governance capabilities:

7. Product vision document.
8. Trust boundary document.
9. Data privacy document.
10. Roadmap document.
11. README with product framing and run/test instructions.

New audit capabilities:

12. Methodology log.
13. Artifact manifest.
14. Content-hash based state fingerprint.

New AI capabilities:

15. Summarize latest sweep.
16. Explain selected/top cohort.
17. Draft mortality overview.
18. Draft key findings.
19. Generate methodology appendix.

New workflow handoff capabilities:

20. Human-edit drafted text.
21. Export report sections.
22. Export full ZIP package.
23. Include methodology log and artifact manifest.

## Immediate Next Move

PR 1, PR 2, and PR 3 are complete in the current working branch. The next implementation slice is PR 4: Deterministic Tool Module Split.

Do not start methodology logging, AI packet work, or report drafting until PR 4 is completed and all tests still pass.
