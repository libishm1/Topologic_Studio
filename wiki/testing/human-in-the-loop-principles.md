# HITL Testing Principles

This page converts the two requested PDFs into testing rules for this project.

## Sources

- `C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\Human.pdf`
- `C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\1-s2.0-S1093968726030215-main.pdf`

`Human.pdf` is a 2026 document on Human-in-the-Loop AI for strategic decision-making in critical infrastructure systems. The gridshell paper is an Elsevier pre-proof titled "AI-driven human-in-the-loop structural optimization of gridshells" with DOI `10.1016/j.cacaie.2026.100044`.

## Rules Adopted For TopologicStudio

- AI agents may run autonomous checks, but humans retain decision authority for high-impact changes.
- Agents must expose evidence: console errors, failed network calls, screenshots, route payloads, and bug-log updates.
- Humans must be asked to judge ambiguous visual outcomes: graph correctness, route plausibility, IFC orientation, fire color interpretation, and UX acceptability.
- Agents must avoid automation bias: a passing build or absent console error is not proof that the graph/path is correct.
- Agents must avoid human overload: summarize findings and ask for a narrow decision, not a full open-ended review.

## HITL Checkpoint Types

| Checkpoint | Agent can do autonomously | Human must decide |
|---|---|---|
| Console and network health | Capture console, page errors, failed requests, route statuses | Whether a warning is acceptable if it affects trust or safety |
| IFC load smoke test | Confirm model appears, page does not crash, no fatal console errors | Whether orientation, scale, and visibility look right |
| Egress graph build | Check route status, node/edge counts, graph overlay presence | Whether graph density and connectivity are plausible |
| Start/exit/path flow | Click/select scripted points and capture path result | Whether route is architecturally plausible |
| Fire and dynamic path | Capture SSE events and UI state changes | Whether hazard visualization communicates the intended state |
| Known bug update | Draft or update issue entry | Approve priority/severity if user-facing or research-critical |

## Gridshell Paper Pattern Adapted

The gridshell paper uses a loop where AI generates candidates, groups or reduces them, asks humans to evaluate representative cases, and propagates that feedback into optimization. For this app, use the same pattern:

1. Agent generates fast browser evidence from headless tests.
2. Agent clusters results into small categories: fatal, functional, visual, research-validity, deployment.
3. Human reviews only representative screenshots or summaries.
4. Agent updates the known-bugs docs and reruns focused checks.
5. Human gives final approval for visual or research-facing behavior.

## Mandatory Documentation Rule

Every autonomous browser debugging session must end with one of these updates:

- Add a new bug to [bug-log-template.md](bug-log-template.md) copied into the appropriate issue page.
- Mark a bug fixed or mitigated in [known-bugs-fixed.md](../issues/known-bugs-fixed.md).
- Add a residual risk to [open-risks.md](../issues/open-risks.md).
- Add a verification note to [verification-status.md](../issues/verification-status.md).

