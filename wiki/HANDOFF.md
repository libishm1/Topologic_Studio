# Handoff: Topologic Studio Next (2026-10-28)

## State

- Live: https://libishm1.github.io/Topologic_Studio/ is `main` (see `git log origin/main`; deployed 2026-10-28 after the gates below). Work happens on `next/browser-core` in the `TopologicStudio-Next` worktree; `main` is fast-forwarded to it only after the gates below pass. The Render backend (`topologicstudio-backend`) is deployed by hand from its dashboard; it had no changes in this round.
- Gates before any deploy: `npm test` (126 tests), `npx eslint src`, a Pages build (`BASE_PATH=/Topologic_Studio/`), and `node tools/browser-test.mjs` desktop and `--mobile` against `vite preview` on the sub-path, all with 0 failed steps.
- Live check after the deploy: every step passes except the TopologicPy Navigation cross-check, which times out at 240 s on Render's free tier (known; not a regression).

## What this round added

- Gap analysis, scoop check and impact ranking: `wiki/research/gap-analysis-2026-10-27/README.md` and its data files.
- Forecast planning (gaps LG1, LG2) and escape-deadline maps (the recommended method C1): `src/core/forecast.js`, `src/core/deadline.js`, `src/workers/forecast.worker.js`, the Simulate step's "Escape deadlines and forecast routes".
- RL's return against the exact optimum (LG6): `src/core/rl.js`.
- IFC handling from `wiki/research/ifc-handling-survey-2026-10-27.md`, items 1 to 5: exits from Pset_DoorCommon and EXTERNAL space boundaries (`src/core/ifcSemantics.js`, `src/workers/semantics.worker.js`), roof slabs excluded, a model egress check, a versioned fragments cache, storey levels where slab geometry collapses.
- Speed: wall sections cut only where the plane meets a triangle (graph build 2.5 times faster on large models, byte-identical output by `tools/graph-hash.mjs`).
- Lean: 20 modules, 863 theorems, all on standard axioms or none (`formal/fireegress`, README fifth batch). The proofs and their property tests found, and the code now fixes: an RL horizon potential and its clock rule, a tie in the dose front, a negative-time read, a map rounding fault and a bisection end in doubles.
- Cross-model harnesses: `tools/forecast-check.mjs`, `tools/deadline-check.mjs`, `tools/exits.mjs`, `tools/profile-graph.mjs`, over the local corpus in `C:\Users\lmurugesan\ifc-samples` and `ifc-samples\more` (local testing only; never publish those files).

## Open

1. IFC conversion runs in a worker (survey item 7, merged 2026-10-28) with one worker kept warm while the page is idle. Follow-ups: run the semantic pass inside that worker, so the page loads one web-ifc instead of two (the semantics worker is 391 KB gzipped); and `clearModel` does not wait for `disposeModel`, so a reload that reuses a model id could race (seen once by the agent with ids it then changed; not seen since).
2. IFC survey items 6 and 8: door widths as capacity in the forecast and deadline models; a room graph from IfcSpace and door space boundaries.
3. Gaps not built: LG5 (publish the cross-model harness as a benchmark), LG7 (routing-ready surrogates with error bounds), LG8 (a two-zone fire model on the room graph). The heat in all three fire models is an index, not a measured temperature, so every dose ranks routes under the model only.
4. Novelty of C1 is provisional until Chen et al. (2023, https://doi.org/10.1016/j.ress.2023.109571) and Tabirca et al. (2015, https://doi.org/10.1016/j.procs.2015.05.425) are read in full and the Semantic Scholar queries are rerun with an API key.
5. Gaps no proof bounds: the deadline map's distance below the exact deadline (up to 7.8 s in tests, C20-Institute); the graph front's error with walls (only "never early" is proved).
