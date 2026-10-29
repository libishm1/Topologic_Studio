# Handoff: Topologic Studio Next (2026-10-29)

## State

- Live: https://libishm1.github.io/Topologic_Studio/ is `main` at e05bb0f (deployed 2026-10-28 after the gates below; live bundle, Methods chunk and sampler worker checked). Work happens on `next/browser-core` in the `TopologicStudio-Next` worktree; `main` is fast-forwarded to it only after the gates pass. The Render backend (`topologicstudio-backend`) is deployed by hand from its dashboard; it had no changes in this round.
- Gates before any deploy: `npm test` (166 tests), `npx eslint src`, a Pages build (`npm run build` with `BASE_PATH=/Topologic_Studio/`; from Git Bash prefix `MSYS_NO_PATHCONV=1`), and `node tools/browser-test.mjs` desktop and `--mobile` against `vite preview` on the sub-path, all with 0 failed steps.
- Live check after the deploy: every step passes except the TopologicPy Navigation cross-check, which times out at 240 s on Render's free tier (known; not a regression).

## Loading speed (2026-10-29)

Opening an IFC on a phone was slow: every step of a first open waited on the one before it, web-ifc was downloaded and started twice, and the examples were converted on the phone like any other file.

- The import worker reads only what the page uses from the fragments (`src/core/importSettings.js`): property sets, quantities, materials, type objects and their relations are left out (ifcSemantics reads them from the IFC). Conversion 40% faster on the Duplex and 26% on Office A at 4x CPU; fragments a third the size. `IMPORT_SETTINGS` is 2, so models cached before are converted again once.
- The same worker then reads the egress semantics with its already-loaded web-ifc and posts them after the fragments; `semantics.worker.js` (419 KB gzipped, a second web-ifc) is gone. The semantics are cached with the fragments, so reopening a cached model starts no web-ifc at all (Duplex reopen 0.22 s).
- Pointing at or tapping a file button starts the import worker at once and fetches web-ifc's WASM and the fragments worker in parallel, instead of one after another after the file arrives.
- The examples are converted when the site is built (`scripts/prepare-examples.mjs`, run by `npm run build`; output in `public/examples/prepared/`, not committed). An example whose converter string matches the page's opens from its fragments (Duplex: 81 KB instead of a 487 KB gzipped IFC plus 1.1 MB of web-ifc) with no IFC worker; the IFC is fetched only if the server cross-check needs it. Otherwise the example opens from its IFC as before.
- The viewer's chunks (three, the That Open engine, ViewerManager) get `modulepreload` links (a plugin in `vite.config.js`), so they download while the page script runs instead of after React renders.
- Measured on an emulated phone (390 x 844, main thread 4x slower, 4G at 9 Mbps and 85 ms, cold cache; median of 3), deployed build against this one, Duplex: tapping the example until the model is usable 4.08 s to 0.97 s; picking the file 2.94 s to 1.20 s; page start 2.44 s to 2.24 s; reopening from the cache 0.67 s to 0.69 s. Chrome will not slow workers, so on a real phone the conversion share is larger in both builds.
- Equivalence: the Model and Graph steps report identical counts, nodes, links, doors and reachability in both builds on the Duplex example and on AC20-FZK-Haus, Office A, example.ifc, AC-20 Smiley West (10 buildings), 4351, the dental clinic, HHS Office and S_Office (stair parts); every one opens faster (desktop 0.2 to 0.9 s). The browser suite adds a step that the Duplex example opens with no IFC worker and still reports 4 of 14 exit doors.
- Not done, possible next: a service worker so repeat visits skip the network (GitHub Pages caches assets for only 10 minutes); geometry conversion itself is now most of a first open of a user's own file.

## What the 2026-10-28 round added

- Two-zone fire model on the IfcSpace room graph (gap LG8): `src/core/rooms.js`, `src/core/zone.js`, `tools/zone-check.mjs`, offered in Simulate when a model has rooms.
- Surrogates with error bounds (LG7): `src/core/surrogate.js`.
- Vertical cores: `src/core/cores.js` and the lattice. Stair parts aggregated into IfcStair are read (S_Office); a stair without treads (an IfcStair with no steps, a room named as a stair) is walked along an assumed dog-leg line; a core is linked wherever its two floors are not already joined there (Smiley West's treads join neither floor); lifts and escalators (IfcTransportElement, rooms and proxies named as lifts) join floors only with "Use lifts and escalators" on. `tools/core-check.mjs` reproduces the comparisons, including a treads-removed experiment.
- Exits in every large part of the graph (`exitsPerPart` in `src/core/ifcSemantics.js`): AdvancedProject 1 exit to 18, canLeave 0.005 to 0.149.
- Review fixes C1, M1 to M11, K1, K2, m1 to m12, w1 to w4 (error boundary, plan staleness and cancellation, worker failure handling, RL in a worker, capped deadline-map work, colour ownership, accessibility).
- Lean: 23 modules, more than 990 theorems and lemmas, standard axioms only; new Surrogate.lean, Zone.lean, Cores.lean (README sixth batch). The Methods dialog renders 21 properties (84 TeX lines, all render in KaTeX).
- Benchmark (gap LG5): `node tools/benchmark.mjs` over 38 public models listed with source and SHA-256 in `benchmark/manifest.json`; speed, accuracy against an exact free-plane geodesic (`tools/geodesic.mjs`), exit-rule precision and recall, deadline checks, and a stress suite with 15 known-bug regressions, each shown to fail on the parent of its fix. Three result-preserving speed-ups (deadline map 1.3 to 1.6 times, RL optimum 1.5 to 6.6 times, A* 1.4 to 2.2 times); outputs identical on the corpus (`tools/output-hash.mjs`, and every non-timing field of `tools/forecast-check.mjs`). Results in `benchmark/results/`.
- Research: `wiki/research/open-source-evacuation-study-2026-10-28.md` (JuPedSim, Vadere, FDS+Evac, CFAST, Menge, MomenTUM from source; Pathfinder and buildingEXODUS from documents only), ranked gaps G1 to G9 and which RiMEA and NIST TN 1822 tests the model can run today.
- Test corpus: 11 openly licensed models in `C:\Users\lmurugesan\ifc-samples\more2\` with `SOURCES.md` (local testing only; HHS Office is no-derivatives).

## Open

1. Robustness bugs the stress suite (`node tools/benchmark.mjs --stress`; see `benchmark/README.md`) marks open: an infinite vertex in a floor triangle makes the lattice build hang, and in a wall empties the graph; millimetre-scaled geometry crashes the lattice build (`Invalid array length`, it samples the whole grid before coarsening); broken or non-IFC files give raw web-ifc WebAssembly traps instead of a readable error, three taking 5 to 14 s to fail.
2. Gap G1 from the study: stairs are walked at level speed (reference tools use 0.45 to 0.88 of it), so stair times are 13% to 50% short. Needs a per-link pace used by every router (route.js, forecast.js, deadline.js, rl.js, alternatives.js, fire.js) and a Lean restatement with link times scaled.
3. Gaps G2 to G5 from the study: walking slowed in smoke, toxic gases and radiant heat in the dose, door capacity and queues, pre-movement distributions (pairs with the deadline map).
4. HITOS (`more2\osb-testfiles-hitos`) joins its floors only with lifts on; its stair cores link but its floors stay apart. Not yet investigated.
5. Lift rides cost their rise at walking speed with no wait; assumed stairs have no known width.
6. Novelty of C1 is provisional until Chen et al. (2023, https://doi.org/10.1016/j.ress.2023.109571) and Tabirca et al. (2015, https://doi.org/10.1016/j.procs.2015.05.425) are read in full.
7. Not proved: floating point for the zone model (tested), the zone Euler step's accuracy, componentLabels' correctness, the deadline map's distance below the exact deadline (up to 7.8 s in tests).
