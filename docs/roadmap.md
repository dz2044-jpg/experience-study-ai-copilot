# Roadmap

This roadmap follows the project improvement plan and prioritizes a credible, testable actuarial workflow before expanding AI features. See the full [improvement plan](improvement-plan.md) for phase details, acceptance gates, and progress tracking.

## 1. Governance Docs, Privacy, And README Polish

Clarify the product purpose, human sign-off boundary, deterministic calculation boundary, AI interpretation boundary, and privacy requirements.

## 2. Golden Workflow Test Hardening

Strengthen the full deterministic workflow tests before refactoring. The golden path should cover profiling, schema inspection, validation, feature engineering, dimensional sweeps, visualization generation, expected artifacts, stable output columns, and selected known A/E values.

## 3. Small Runtime Extraction

Reduce the size of the copilot runtime by extracting only clear behavior-preserving seams first, such as session state, prerequisite guidance, fallback planning, and response formatting.

## 4. Deterministic Tool Module Split

Split the deterministic tool implementation into focused modules while preserving public tool names, artifact names, output columns, and known deterministic results.

## 5. Methodology Log And Artifact Manifest

Add auditability through methodology events and content-hash based artifact tracking. These records should support future methodology appendices, export packages, and stale-status detection.

## 6. AI Packet, Sanitization, Validation, And Fallback Layer

Build a safe AI foundation using allowlisted cohort-level packets, rare-cohort masking, unsupported-claim validation, and deterministic fallback responses when LLM access is unavailable.

## 7. AI Interpretation UI

Add an AI Interpretation Panel for summarizing the latest sweep and explaining selected cohorts, gated by artifact availability and backed by evidence references.

## 8. Report Drafting MVP

Draft mortality overview, key findings, and methodology appendix sections from deterministic packets and actual methodology events.

## 9. Human Editor And Export Package

Provide editable report sections and export options for Markdown and ZIP packages that include report sections, deterministic artifacts, visuals, and audit files.

## 10. Prior Report Style Reference Layer

Use approved or synthetic prior-report examples for style, structure, and tone only. Prior reports are not factual sources and should not be copied verbatim unless explicitly approved.
