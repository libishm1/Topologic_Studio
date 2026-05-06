# Agent Loading Rules

This wiki is designed for targeted context loading. Do not load the full wiki before a code change.

## Rule Of Thumb

1. Start with [agent-index.md](agent-index.md).
2. Load only the intent row that matches the task.
3. Load one API contract page if changing requests or responses.
4. For browser work, load [testing/agent-instructions-codex-claude.md](testing/agent-instructions-codex-claude.md).
5. Load [issues/open-risks.md](issues/open-risks.md) before final verification.

## Browser Testing Rule

Any agent that tests or debugs the app in a browser must capture browser console output, page errors, failed requests, API statuses, and at least one screenshot when visual behavior is involved. Every browser debugging session must update [issues/known-bugs-fixed.md](issues/known-bugs-fixed.md), [issues/open-risks.md](issues/open-risks.md), or [issues/verification-status.md](issues/verification-status.md).

Human-in-the-loop review is required for route plausibility, IFC orientation/scale, graph density, fire visualization meaning, deployment URL changes, and any ambiguity that affects research validity.

## Avoid Loading

- `wiki/verification/frontend-dist/` unless inspecting the production bundle.
- Node runtime folders, zip files, logs, or generated assets unless the task explicitly needs them.
- The entire app source when one chunk points to the relevant file and function group.

## Codebase Cautions

- The git worktree is dirty outside the wiki. Treat non-wiki edits as pre-existing user work.
- Backend graph state is in process memory through `LAST_GRAPHS`; route tests may depend on call order.
- Frontend build succeeds, but lint currently fails. See [verification](verification/README.md).
- Local development is inside a OneDrive path with spaces. Use quoted paths and the bundled Node path where needed.
