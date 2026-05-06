# Agent Index

Load the minimum useful slice for the task. Each row lists the first files to read.

| Task intent | Load these wiki chunks | Then inspect code |
|---|---|---|
| Fix IFC graph generation | [backend/ifc-egress-graph.md](backend/ifc-egress-graph.md), [api/ifc-egress.md](api/ifc-egress.md), [issues/open-risks.md](issues/open-risks.md) | `topologicpy-web-backend/app/main.py`, `topologicpy-web-frontend/src/App.jsx`, `topologicpy-web-frontend/src/IFCViewer.jsx` |
| Fix IFC viewer loading or extraction | [frontend/ifc-viewer.md](frontend/ifc-viewer.md), [sources/external-references.md](sources/external-references.md) | `topologicpy-web-frontend/src/IFCViewer.jsx` |
| Change fire simulation | [backend/fire-routing-rl.md](backend/fire-routing-rl.md), [api/fire-rl.md](api/fire-rl.md), [frontend/app-orchestration.md](frontend/app-orchestration.md) | `app/main.py`, `src/App.jsx`, `src/IFCViewer.jsx` |
| Change RL path training | [backend/fire-routing-rl.md](backend/fire-routing-rl.md), [api/fire-rl.md](api/fire-rl.md) | `app/main.py`, `src/App.jsx` |
| Fix JSON or IFC upload | [backend/topology-ifc-upload.md](backend/topology-ifc-upload.md), [api/upload-contracts.md](api/upload-contracts.md), [issues/known-bugs-fixed.md](issues/known-bugs-fixed.md) | `app/main.py`, `src/App.jsx` |
| Frontend state or workflow bug | [frontend/app-orchestration.md](frontend/app-orchestration.md), [frontend/topology-viewer.md](frontend/topology-viewer.md) | `src/App.jsx`, `src/TopologyViewer.jsx` |
| Deployment bug | [deployment/vite-github-pages.md](deployment/vite-github-pages.md), [deployment/render-backend.md](deployment/render-backend.md), [deployment/onedrive-windows.md](deployment/onedrive-windows.md) | `.github/workflows/deploy-frontend.yml`, `render.yaml`, `Dockerfile`, `vite.config.js` |
| Dependency upgrade | [sources/dependencies.md](sources/dependencies.md), [references/context7-api-notes.md](references/context7-api-notes.md), [issues/open-risks.md](issues/open-risks.md) | `package.json`, `requirements.txt`, lockfiles |
| Future IFC Lite work | [roadmap/ifc-lite-profile.md](roadmap/ifc-lite-profile.md), [references/ifc-standard-notes.md](references/ifc-standard-notes.md), [backend/ifc-egress-graph.md](backend/ifc-egress-graph.md), [frontend/ifc-viewer.md](frontend/ifc-viewer.md) | extractor and graph functions only |
| Browser console test/debug | [testing/agent-instructions-codex-claude.md](testing/agent-instructions-codex-claude.md), [testing/headless-browser-console-testing.md](testing/headless-browser-console-testing.md), [testing/playwright-console-harness.md](testing/playwright-console-harness.md), [testing/browser-test-matrix.md](testing/browser-test-matrix.md), [testing/bug-log-template.md](testing/bug-log-template.md) | browser automation harness, then only files implicated by evidence |
| Human-in-the-loop review | [testing/human-in-the-loop-principles.md](testing/human-in-the-loop-principles.md), [issues/open-risks.md](issues/open-risks.md), [issues/known-bugs-fixed.md](issues/known-bugs-fixed.md) | screenshots, console log summaries, API evidence |

## Execution Checklist

- Confirm the current git status before editing app files.
- Use the existing route and state contracts unless the task requires a contract change.
- Run focused verification from [verification/README.md](verification/README.md).
- Update the relevant wiki chunk if implementation behavior changes.
- For browser tests, update a known-bug, open-risk, or verification page before final response.
