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

## 2026-10-27

- Reviewed fire egress, simulation and RL on the Next line against TopologicPy
  0.9.72-0.9.87, syntopy.io and its tutorials, ModRes, and the LOD1 model in
  `Shoaib_masters`. Write-up: [roadmap/review-2026-10-27.md](roadmap/review-2026-10-27.md).
- Fixed the temperature-model `reduceat` bug, five RL defects, and
  temperature-only rerouting. Cached components; vectorised wall blocking.
- topologicpy 0.9.71 -> 0.9.86 (0.9.87 removes `Graph`); kernel pinned with
  `TOPOLOGICPY_CORE_BACKEND`; ifcopenshell capped `<0.9`; Dockerfile slimmed;
  `render.yaml` asks for Frankfurt.
- Measured the live Classic backend: 33.1 s cold start, 0.31 s warm.
- Backend 95/95, lint clean, build clean, browser suite 30/30 with 0 console
  errors. Docker image not built (no Docker on this machine); nothing deployed.
