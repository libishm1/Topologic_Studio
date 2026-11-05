# Handoff: Topologic Studio Next (2026-11-05, evening)

## State

- **Live:** https://libishm1.github.io/Topologic_Studio/ is `origin/main` = `next/browser-core` at 49ae6e5, 26 commits after the previous live f8bc65e.
  - **Live checks:** the last three deploys (4f0f82a, 00ec1a4, 49ae6e5) each passed the browser suite against the live site with 0 failed steps.
  - **The TopologicPy Navigation cross-check** passes in 1.6 to 82 s when the Render backend is awake. Asleep (free tier), the page's capability probe fails and the step reports the button disabled; wake it with `curl https://topologicstudio-backend.onrender.com/health` and rerun.
  - **The local `main`** is a stale ref (f936b86). Deploy by fast-forwarding `origin/main` to `next/browser-core` (`git push origin HEAD:main` after `git merge-base --is-ancestor origin/main HEAD`).
  - **The Render backend** is unchanged since 4f09929 and is deployed by hand.
- **Gates before any deploy** (all 0 failed steps):
  - `npm test` in `topologicpy-web-frontend`: 217 tests;
  - `npx eslint src`;
  - the Pages build: `MSYS_NO_PATHCONV=1 BASE_PATH=/Topologic_Studio/ npm run build` from Git Bash;
  - `node tools/browser-test.mjs --url http://localhost:4173/Topologic_Studio/`, desktop then `--mobile`, against `vite preview --port 4173` on that sub-path.

  The suite now also checks:
  - the Model step identifies elements;
  - the run card says embers are on and gives the openings;
  - the shielded agent escapes when an escape exists (at fire speed 0.25), and Q-learning still reports its return;
  - every Methods equation typesets;
  - a guided run reaches its route.
- **Lean:** 28 modules, 1297 theorems, 317 AxiomCheck lines, standard axioms only. The working copy (`lean/fireegress` under the user's home) and `formal/fireegress` match byte for byte.
- **Both research agents reported** and their work is committed:
  - **The learned-escape study** (`rlSafe.js`, `SafeEscape.lean`, `wiki/research/rl-egress-study-2026-11-05.md`). The shielded learner is the app's default.
  - **The IFCNet furniture study, phase 1** (feffefc, `wiki/research/ifcnet-furniture-study-plan-2026-11-05.md`). It is research and a pilot only; no training campaign has been run.
- **Uncommitted on purpose:** `wiki/research/free-inference-gpu-options-2026-11-05.md`. Its application drafts speak in the user's voice and the repository is public; the user decides.

## Resume here (2026-11-05, evening)

Nothing is half-done. The next steps are the user's decisions:

1. **IFCNet retraining** (the phase 1 plan, section 7):
   - **Scope:** a feature model trained only on open data (a few CPU hours of fitting, but 2 to 4 weeks of supervised data work), with or without a DGCNN trained from scratch (about a week more of CPU).
   - **Data:** which open sources to download, and the disk budget, outside OneDrive.
   - **Licence:** the published model's licence (CC BY 4.0 or MIT), and whether to train on CC0 data only.
   - **Labels:**
     - approve the sub-type vocabulary;
     - decide whether to hand-label the 473 unnamed furniture elements as the final test set;
     - decide whether the app should use the rules-then-features cascade for typed elements only, since it hurts proxies.
   - **Venue:** Automation in Construction or AEI, or Fire Safety Journal or Fire Technology.
   - **Pilot findings:**
     - fine-tuning IFCNet's weights neither helps nor can be published;
     - only a model trained on open data alone can be published;
     - the rules-then-features cascade beats the rules on typed elements (+0.022 accuracy) but not on proxies;
     - furniture sub-types are no better than the box alone yet.
2. **IFCNet weights now shipped:** write to the authors (Christoph Emunds, E3D, RWTH Aachen) for a licence, or replace them once an open model exists.
3. **"Describe this model":** whether to build it. Chrome's on-device Prompt API first (Chrome 148+ desktop), and an opt-in Cloudflare Worker only for other browsers. Not FreeInference: its terms log prompts.
4. **The free-inference note:** whether to commit it.

Before any deploy: the gates above, then the live check, then update this file.

## This round (2026-11-05, afternoon and evening)

- **Identification** (study 4ef5e1e; app 8ad8b3d):
  - **Rules:** `src/core/identify.js` names elements by IFC class, else name, else box. Accuracy 0.86 and macro-F1 0.83 on 6,066 typed elements from 45 models; 0.55 to 0.60 on 159 hand-labelled proxies. The full study is `wiki/research/ifc-identification-benchmark-2026-11-05.md`.
  - **IFCNet:** the DGCNN is ported to ONNX, exactly and pruned: 1.45 MB, and 22 to 45 times faster with no answer changed. It reproduces the paper's balanced accuracy (0.799 against 0.791). It adds almost nothing to the rules.
  - **Shipped weights:** by the user's decision the weights are in the public build (`public/models/ifcnet-dgcnn-2048.onnx`, `NOTICE-ifcnet.md`). IFCNet's code is MIT; the weights state no licence, and they come down on request.
  - **The Model step:** "What the elements are" shows the rules at once; "Identify by shape" runs IFCNet on generic elements the name could not place and keeps its answer at 90% or more. onnxruntime-web 1.30.0 loads from jsDelivr on first use.
  - **LLMs:** Qwen 0.5 to 0.6B in the browser scored below the rules and made them worse when asked to correct them.
- **Flamescapes:**
  - **Liquid fires** (classes B and K) never lit; fixed with the pool burning-rate cap (0e9c2f1; Babrauskas 1983 via NUREG-1805).
  - **GPU-lost fallback:** a GPU lost mid-run restarts the fire on the CPU (fac59bb).
  - **Smoke cloud and embers** (4f0f82a): the solver's soot is drawn as a smoke cloud (`smokeCloud`, both backends); the run card says "Embers: off" or "on: N thrown, M spot fires".
  - **Speed** (4f0f82a): on this machine's integrated GPU a 120 s fire went from 72 s to 25 to 29 s.
    - The view draws 15 times a second while the GPU computes (`ViewerManager.setDrawRate`).
    - The GPU grid is sized to its measured speed in fixed levels (`flamescapes.worker.js`, `GPU_STEP_MS` 33). The speed is kept per GPU in localStorage.
    - "Grid on the GPU" offers speed-sized or finest.
    - One CPU core: 2.8 to 2.9 times real time on the Duplex (`benchmark/results/2026-11-05-fire.json`).
  - **Grid dependence (important):** the same Duplex sofa fire peaked at 0.09 MW on 0.20 m, 2.5 MW on 0.23 m, 6.1 MW on 0.27 m and 0.75 MW on the CPU's 0.34 m. The Methods page and the panel say to compare runs on one grid. See Open 11.
  - **Openings** (00ec1a4, 49ae6e5):
    - doors open or closed: a switch beside the fire class, for the smoke only; routes still use every door;
    - windows closed or open;
    - ventilation off or running: every air terminal swaps gas with one well-mixed duct volume at an assumed 100 L/s a terminal (`ducts` on both backends; `DuctMixing.lean` proves conservation and that values stay within range).

    Checked on the CC BY TalTech DS3 model: 60 of 82 terminals inside the clipped grid.
- **Export for TopologicPy (JSON)** (49ae6e5, `src/core/topologicExport.js`): in the Graph step and under a finished fire. It holds the graph in IFC world coordinates at floor level, with per-node fire time series and untenable times, edges with lengths, and the run's settings. topologicpy 0.9.86 `Graph.ByJSONFile` reads the Duplex export with all 1,597 nodes and 4,597 edges.
- **Guided runs** (49ae6e5, `src/core/guided.js`): three presets on the welcome screen run the steps themselves and stop at the result (*The way out*; *A fire on the route*; *Smoke behind closed doors*).
- **Notes for later:**
  - `wiki/research/fire-suppression-note-2026-11-05.md`: sprinklers, RTI activation, Fire-X's droplet evaporation, manual agents;
  - `wiki/research/sbc-firefighting-future-notes-2026-11-05.md`: SBC egress checks, hose coverage, firefighter access, doors in the graph fire models. No code value is quoted.
- **Tools:**
  - `tools/fire-gif.mjs` records the Flamescapes fire as a GIF with no encoder to install (`--origin`, `--smoke-at`);
  - `tools/flamescapes-gpu-check.mjs` takes `--class` and `--ducts`.

## Flamescapes, fire classes and the ignition study (2026-11-05)

- Papers: Flamescapes (Wrede, Wagner et al. 2026, ACM TOG 45(6) 246; DOI printed as 10.1145/3842518, not yet registered at Crossref on 2026-11-05) with its supplement, and Fire-X (Wrede et al. 2025, doi:10.1145/3763338), both CC BY 4.0, transcribed in `wiki/sources/papers/` with attribution and equation catalogues (Flamescapes 20 equations and 217 parameters, Fire-X 32 and 133). The PDFs are in the ignored `reference/papers/`. Their GitHub repositories (helgewrede/flamescapes, helgewrede/firex) hold project pages only, so no code was reused; saskaZs/Fire-Simulation is a particle visual with no licence, used for no code.
- Solver: `src/core/voxelize.js`, `flamescapes.js`, `flamescapesRun.js` follow Algorithm 1 on a grid built from the IFC (walls, slabs, roofs, stairs, obstacles; doors open; contents the paper's statistical fire load, Eqs. 3 to 5). The header of `flamescapes.js` lists what is the paper's and what is changed for a coarse CPU grid: exact integration of each reaction and two-body exchange over a step, a thermally thick surface layer (1 kg/m^2), sub-grid ember spot ignition by excess heat only, the first item alight at the start, thin walls snapped to the cell holding most of them, solid-aware trilinear advection. Constants are the paper's Tables 3 to 5 where it gives them, Fire-X's 417 kJ per mol O2, and the fire class's fuel for the first item.
- App: an optional toggle in the Simulate step (off by default) adds "Flamescapes: fire and smoke" and "... and embers". `workers/flamescapes.worker.js` sizes the grid to the device (`lib/hardware.js`) and streams frames without keeping them; the fire plays them as they arrive, reroutes on the gas temperature at head height, and the plan reuses the run (`forecastFromFrames` in `forecast.js`: heat FED plus ISO 13571 CO FED; smoke closes a node below the visibility limit; the plan warns when the grid covers only part of the graph). Heat or smoke view; glowing flame and ember particles in persistent layers.
- Fire classes (`src/core/fireClasses.js`): NFPA 10 A, B, C, D, K and ISO 3941:2026 L, each value with its source key; CO2 yields from each fuel's carbon balance; suitable and unsuitable extinguishers shown in the panel. Values come from the review `wiki/research/ignition-sources-2026-11-05.md` and `.json` (ignition sources, design fires, IFC mapping, DOIs checked at Crossref).
- Likely ignition points (`src/core/ignition.js`, catalogue generated by `tools/ignition-catalog.mjs` into `ignitionCatalog.js` and `ignitionClasses.js`): elements and rooms ranked by share of fire deaths times match confidence over candidates; one click sets the fire origin and its class. Lithium-ion scenarios are class L.
- Lean: FireX.lean, Flamescapes.lean, FlamescapesPort.lean (seventh batch; 26 modules, more than 1,230 results, standard axioms only). The proofs confirmed the transcribers' suspected typos (none refuted) and found six defects in the port, fixed in 30c019a; class D's heat capacity followed in 0d0c676.
- Architecture review (2026-11-05, 16 findings): all fixed in 7df6041. The two criticals were walls acting as an air source in advection and a clip that dropped long walls crossing a clipped storey; each has a regression test.
- Speed (`tools/fire-bench.mjs`, one CPU core, quiet machine, rerun 2026-11-05): Flamescapes 34 to 36 ms per step on the Duplex (26k cells, 0.37 m), 2.8 to 2.9 times real time; advection and projection are most of it. The graph models take 0.1 to 7 ms. GPU timings: see This round.
- Tests: 207 committed unit and property tests (2026-11-05 evening); the browser suite runs a 30 s Flamescapes fire with embers (15 to 23 s).
- Live check: 0 failed steps when the Render backend is awake (see State).
- README (2026-10-29): rewritten for new users, with a tour of the four steps on the Duplex, the other examples, a phone view and the Methods page. The screenshots in `docs/images/` come from `node tools/readme-shots.mjs` against a running dev server (or any build with `?debug`), using only the shipped examples; re-run it after a UI change. The RL shot's success rate varies between runs, so the README quotes none.

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
- Test corpus: 11 openly licensed models in `%USERPROFILE%\ifc-samples\more2\` with `SOURCES.md` (local testing only; HHS Office is no-derivatives).

## Open

1. Robustness bugs the stress suite (`node tools/benchmark.mjs --stress`; see `benchmark/README.md`) marks open: an infinite vertex in a floor triangle makes the lattice build hang, and in a wall empties the graph; millimetre-scaled geometry crashes the lattice build (`Invalid array length`, it samples the whole grid before coarsening); broken or non-IFC files give raw web-ifc WebAssembly traps instead of a readable error, three taking 5 to 14 s to fail.
2. Gap G1 from the study: stairs are walked at level speed (reference tools use 0.45 to 0.88 of it), so stair times are 13% to 50% short. Needs a per-link pace used by every router (route.js, forecast.js, deadline.js, rl.js, alternatives.js, fire.js) and a Lean restatement with link times scaled.
3. Gaps G2 to G5 from the study: walking slowed in smoke, toxic gases and radiant heat in the dose, door capacity and queues, pre-movement distributions (pairs with the deadline map).
4. HITOS (`more2\osb-testfiles-hitos`) joins its floors only with lifts on; its stair cores link but its floors stay apart. Not yet investigated.
5. Lift rides cost their rise at walking speed with no wait; assumed stairs have no known width.
6. Novelty of C1 is provisional until Chen et al. (2023, https://doi.org/10.1016/j.ress.2023.109571) and Tabirca et al. (2015, https://doi.org/10.1016/j.procs.2015.05.425) are read in full.
7. Not proved: floating point for the zone model (tested), the zone Euler step's accuracy, componentLabels' correctness, the deadline map's distance below the exact deadline (up to 7.8 s in tests).
8. Occupant simulation add-on, asked for by the user (2026-11-05): many occupants moving at once, each with a pre-movement time, walking speed slowed on stairs (gap G1) and in smoke (the Flamescapes visibility, gap G2), door queues with a specific flow per metre of width (gap G4), and a running FED from heat and CO; route choice by shortest path, by re-planning on the fire as it is, by the forecast's earliest safe route, or by a learned Q-policy (today's Q-learning trains one start and one exit, so a per-exit or per-room policy is needed). Outputs: evacuation curve, RSET, incapacitated count, animated occupants. Needs a source review for each parameter before code.
9. Fire classes: the parameters marked `provisional` in `fireClasses.js` (B and C ignition temperatures for embers, K's vapour latent heat, L's fuel and agents) have no source yet.
10. Flamescapes in the browser is not calibrated against room fire tests; the review's room-scale data (NIST furniture calorimetry, cooktop fires) are the natural first comparisons. Indoors at 0.37 m the embers add little: the Duplex fire lights no spot fire in 120 s.
11. **Grid dependence of the Flamescapes fire.**
    - **The finding:** peak heat release moves by two orders of magnitude between 0.20 and 0.34 m cells, and not in one direction. The 0.23 m run also settles to a constant 460 kW tail.
    - **Before any quantitative use:** a grid convergence study (same scene, 0.15 to 0.4 m, CPU and GPU), and a check of the tail (the pilot, or a steady burn of the contents).
    - **Speed-sized grid noise:** sizing the GPU grid by speed makes the grid device-dependent; the stored speed keeps it fixed per browser. Fresh browsers picked 110k to 200k cells in three sessions, because the speed measurement is noisy while other jobs run.
12. **Ventilation is a bound, not a model.**
    - **Missing inputs:** system membership (IfcRelAssignsToGroup with IfcDistributionSystem) and supply, return and exhaust roles are not read; duct geometry is unused; 100 L/s a terminal is an assumption.
    - **Next:** read the systems where the IFC has them (DS3 has 4), make exhaust terminals a sink, and source the flows.
13. **Doors in the graph fire models,** SBC egress checks, fire hose coverage and firefighter access: see `wiki/research/sbc-firefighting-future-notes-2026-11-05.md`. Suppression: `fire-suppression-note-2026-11-05.md`.
14. **"Describe this model":** not built. Plan in Resume here, decision 2.
15. **IFCNet:**
    - the weights' licence (ask the authors);
    - the retraining study (agent phase 1 pilot; the full campaign needs approval);
    - the shape classifier adds almost nothing to the rules today.
16. **Learned escape:** done.
    - **The new method:** the shielded learner (`rlSafe.js`, `SafeEscape.lean`, `wiki/research/rl-egress-study-2026-11-05.md`) is the default.
    - **Still open:**
      - The shield uses the RL step's own hop-front fire. Covering the temperature, zone and Flamescapes forecasts needs a seconds-based version (deadline.js has one).
      - Certification with many exits can take up to 2.6 s.
      - rl.js still draws its policy with a no-revisit walk, which caused most of its misses. Drawing the policy walk instead cuts them from 15.0% to 2.6%.
    - **Item 8:** its per-room, any-exit policy now exists as `escapeField`.
17. **The CPU solver's advection** is half of each step, at about 730k trilinear samples. Any speed-up must stay bit-identical: vorticity reads the faces between solids, so no face can be skipped. Micro-optimisations, such as the exact own-component shortcut, are possible.
18. **The guided run "A fire on the route"** uses heat diffusion, which reads faintly. Spread-along-links, or the forecast plan, may show the rerouting better.
