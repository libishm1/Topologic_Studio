# Agent Instructions For Codex And Claude Code

These instructions are written for coding agents. Apply them whenever you test or debug TopologicStudio through a browser.

## Non-Negotiable Rules

- Work inside the current repo unless the user explicitly requests otherwise.
- Do not treat a build pass as full validation.
- Always inspect browser console output for frontend workflows.
- Always record new bugs, fixed bugs, or residual risks in `wiki/issues/`.
- Use human-in-the-loop checkpoints for visual, safety, research-validity, or ambiguous behavior.
- Do not ask the human to inspect raw logs unless you have summarized the decision needed.

## Autonomous Browser Test Loop

1. Read [agent-index.md](../agent-index.md) and the relevant implementation chunk.
2. Start backend and frontend if needed.
3. Use headless browser automation to open the app.
4. Capture console, page errors, failed requests, API responses, and screenshots. Use [playwright-console-harness.md](playwright-console-harness.md) when a harness is needed.
5. Run one focused workflow: load IFC, build graph, path, fire stream, RL, upload JSON, or deployment preview.
6. Classify findings:
   - `fatal`: app crashes, blank viewer, route unavailable.
   - `functional`: feature result wrong or missing.
   - `visual`: visible issue requiring human judgment.
   - `research-validity`: result may mislead egress/fire interpretation.
   - `environment`: OneDrive, Node, CORS, WASM, port, build problem.
7. Update issue docs before final response.
8. Ask the human only for the smallest needed decision.

## Required Bug Documentation

For every test/debug session, update at least one:

- [known-bugs-fixed.md](../issues/known-bugs-fixed.md)
- [open-risks.md](../issues/open-risks.md)
- [verification-status.md](../issues/verification-status.md)

If the finding does not fit, create a new issue page under `wiki/issues/` and link it from [index.md](../index.md).

## Human-In-The-Loop Trigger List

Pause for human input when:

- The IFC orientation, floor/stair extraction, or route plausibility is uncertain.
- A fix changes graph sampling, wall blocking, fire spread, or hazard weighting.
- A visual screenshot looks technically valid but may be misleading.
- A deployment change affects public URLs, CORS, or external hosted assets.
- A test requires credentials, private data, or destructive cleanup.

## Final Response Contract

When reporting results, include:

- What workflow was tested.
- Whether browser console had errors or warnings.
- Whether API requests succeeded.
- Screenshot/artifact paths if produced.
- Which bug/risk/verification docs were updated.
- The exact human decision still needed, if any.
