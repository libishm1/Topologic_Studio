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
- Adopted two TopologicPy additions (review section 7):
  - `/api/ifc/path/alternatives`: Yen (TGraph.KShortestPaths' semantics,
    delegated to it on topologicpy >= 0.9.87, which was verified in a throwaway
    venv) plus a default "distinct" mode, because Yen's runners-up tie on a
    lattice.
  - `/api/ifc/navigation-check`: TopologicPy's `Navigation` graph against ours,
    aligned exactly by That Open's coordination matrix. Duplex: 99% / 91%
    coverage; Level 2 has edge bands 0.54-0.60 m apart.
  - Corrected the review: `KShortestPaths` is 0.9.87, not 0.9.86.
- Lean: proved L3, L4, L8, L12, L13 and L14 (11 theorems) in
  `formal/fireegress`; standard axioms only. Research note section 8.5.
- Routes drawn straight (src/core/straighten.js), Wire.Straighten's rule with
  doors and stairs kept and 0.25 m wall clearance; 600 routes on three
  examples, none through a wall (tools/straighten-check.mjs).
- Redesign from the Claude Design canvas "Topologic Studio redesign": the
  inspector is a four-step workflow (Model, Graph, Route, Simulate), a
  welcome screen with the examples, result cards, notices instead of toasts
  for a split graph, a fire playback bar, and a Methods page with the
  Lean-proved properties and the models' limits (W2-W4).
- UI/UX pass (tools/a11y-audit.mjs, axe-core 4.10, WCAG 2.1 A/AA): fixed
  text, status and control-border contrast in both themes, 24 px+ targets,
  reflow at 200% zoom, focus on the file button, landmarks, and a keyboard
  way to place start, exit and fire. 0 axe violations over 10 states. Heat
  is drawn on a sequential ramp instead of a rainbow.
- Multi-agent equation review: 90 equations extracted, checked against read
  sources (71 confirmed, 11 corrected, 8 unverifiable;
  research/fire-equations-verified-2026-10-27.json) and formalised in Lean
  (FireDynamics, Tenability, GraphNumerics; 175 axiom checks, standard axioms
  only). Found: L2's boundary clause and R3's prefix claim false (counter-
  examples proved), SFPE speed non-monotone at its 0.54 switch, density 3.8
  past jam, MQH 740 vs 6.85 off by 4%, FED kelvin trap.

