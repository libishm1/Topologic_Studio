# Wiki Log

## 2026-05-06

- Created OpenKB-style wiki structure under `TopologicStudio/wiki/`.
- Parsed local PDFs for page counts and metadata; the local PDF has no extractable text and needs OCR for semantic notes.
- Inspected active source files, dependency metadata, deployment config, git history, and dirty working-tree diffs.
- Queried Context7 for Vite, FastAPI, and Three.js API/deployment references.
- Checked OpenKB repository documentation through GitHub.
- Verified frontend build using bundled Node with output redirected to `wiki/verification/frontend-dist`.
- Verified frontend lint status, which currently fails.
- Verified backend import/routes using `topologicpy-web-backend/.venv`.
- Added chunked agent and human indexes plus backend, frontend, API, deployment, issue, roadmap, source, reference, and knowledge-graph slices.
- Replaced long compatibility pages with short indexes that point to smaller chunks.
- Added HITL testing instructions from `Human.pdf` and the Elsevier gridshell HITL optimization paper.
- Added Codex/Claude browser testing instructions, headless console capture workflow, browser test matrix, and bug-log template.
