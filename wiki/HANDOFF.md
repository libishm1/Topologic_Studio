# Handoff: Topologic Studio Next (2026-11-10)

## State

- **Live:** https://libishm1.github.io/Topologic_Studio/ is `origin/main` at 7200efd (released 2026-11-10 at the user's "release it if it doesnt break the existing site"): a failed human check says why, from Cloudflare's error code (lib/turnstile.js `humanCheckFailure`). The user had seen "The human check failed. Try again." on the describer; it was Turnstile's widget refusing in their browser, and it then worked for them. Gates: 405 unit tests, lint, the Pages build, the browser suite desktop 129 and phone 128 steps, server-race 7 of 7, model-race 5 of 5; live check desktop 129 and phone 128 steps, 0 failed, 0 console errors. **The Render backend runs the audit's code**: its `/health` gives counts only (B11 live; the user deployed it by hand). Roll back the frontend by pushing 66e3315 to main.
- **Built, not released (2026-11-10, next/browser-core):** people leaving now take the quickest exit by default, the nearest an option (the user: "Make sure the fire egress people escaping use all the doors as an option by default so theres more chance of escape"; asked "is it a bug or a feature": a design choice, every exit out of the building was already used, but each person took the nearest and queued there while other doors stood free). `core/crowd.js` `choice` "quickest": a walk tree to each exit; each person weighs every exit by the walk there or the people ahead of them for it at the flow seen through it, whichever is longer, every 2 s; turns only where the two ways part and when it saves 5 s and 20 % (SWITCH_GAIN, SWITCH_SHARE); the flow is the door's capacity until someone heading for it is held up, then an average over 20 s of what passes (a stair feeding an exit slower than its door makes it look as slow as it is); with a fire, only exits whose walk the fire has not cut. Precedent: Pathfinder's "locally quickest" door choice (Technical Reference, Path Planning: place in the queue and the door's observed flow, a preference for the current door); Duives and Mahmassani 2012 (doi:10.3141/2316-10, checked on doi.org and Crossref) not cited, its abstract unread. Above 2e6 exits x nodes it falls back to the nearest and says so. Measured with no fire, last one out (nearest to quickest): NIBS Clinic 459 to 333 s, NIBS Office 433 to 350 s, Office 76 to 73 s, Duplex, Vienna (one stairwell, both exits at its foot) and the Courtyard unchanged; with a stand-in fire (nodes closing 3 s a metre from a start), Clinic 627 to 556 s, Office 110 to 112 s. Found on the way: counting those ahead in 1 m bands made people in a dense queue turn back and forth (0.1 m now); the door's nominal flow made the Clinic slower than the nearest (495 s) since its stairs feed exits slower than their doors. Panel: "Each person takes" (quickest, queues counted / nearest), the notice says how many turned to another exit. Gates: 408 unit tests, lint, the Pages build, browser suite desktop 131 and phone 130 steps (0 failed, 0 console errors), server-race 7 of 7, model-race 5 of 5. Release only on the user's word.
- **Released before it:** 66e3315 (2026-11-10 at the user's "Ok do it now"): the state-machine audit, all three areas (016a002, 4f7d57b, 66e3315; see [architecture/state-machines.md](architecture/state-machines.md)). Live check 129 steps, 0 failed, 0 console errors.
- **Released before it:** 713c697 (released 2026-11-09 at the user's "Release it"): people leaving starts only when switched on for the graph on screen. The user saw people start as soon as a graph was built: the switch was stored with the settings (on after the Vienna demo or by hand). Now it lasts for the visit only (core/settingsMigration.js SESSION_ONLY) and a new graph switches it off; a graph is told by its nodes, so the server upload's copy (graphId added) neither switches it off nor restarts the walk. Checked live: a stored switch starts nothing, on through the upload, off after a rebuild; live check 121 steps, 0 failed, 0 console errors.
- **Released before it:** 228dc6d (2026-11-09 at the user's "Release it"): 4a01f78, the people options, Record GIF, guided origins and the "Vienna on the GPU" demo (see "Built after it" below), then 228dc6d, deploy-pages waiting up to 20 minutes: the first deploy of 4a01f78 built but timed out at GitHub Pages' default 10 minutes ("Timeout reached, aborting!", no GitHub incident; the deploy before took 16 s), and the change's push ran it again. Live check 120 steps, 0 failed, 0 console errors, Render awake; the Vienna demo ran live start to end in 23 s on the GPU, no errors. The previous release is tagged `live-2026-11-09-people`. The user declined a chat on the describer.
- **Released before it:** 0f256f0 (2026-11-09 at the user's "the gpu works now lets add the agents to the deployment and test it"): people leaving, the optional egress animation (see "Built: people leaving" below). Live check 117 steps, 0 failed, 0 console errors, Render awake; the user confirmed WebGPU works on their Quadro RTX 4000 laptop. The backend did not change.
- **Released before it:** d0a1ffc (2026-11-09 at the user's "build the web deployment so I can test on my Quadro laptop", choosing "release now, crowd later"): fede128, 39c276a, e3effe6 and d0a1ffc, i.e. the CPU and GPU agreeing step by step, the CPU 12 to 26 % faster, the standard grid by default and repeat runs on the GPU (off by default). main, next/browser-core and next/calibration all point at it. Live check 112 steps, 0 failed, 0 console errors, Render awake (TopologicPy 0.9.86). The backend did not change.
- **Released before it:** 7a8079d (2026-11-09, the user's "test why gpu releases twice the flame output and then release the fix"): the design-fire item's flame drawn as a flame (narrowing, flickering, at its cells' gas temperatures) instead of the "pillar of light" the user saw. Gates: 371 unit tests, lint, the Pages build, the browser suite desktop and mobile (0 failed), every phase of the GPU check; live check 0 failed steps, Render awake. The backend did not change.
- **Branch `next/calibration` (not on main; merge only on the user's word, "only if we are confident we merge with main"):**
  - fede128: the CPU and GPU fires agree step by step. The CPU's radiation emits from the stage's starting temperatures (it depended on cell numbering), flame particles beyond their share are refused by the same hash on both paths (cell order on the CPU, thread order on the GPU before), and the CPU casts the GPU's rays (`cellStream`, `pcgHash`). RUN_CACHE_VERSION 8. Open: over whole runs the 200 kW test room's fire held after its flare in 16 of 16 GPU runs and 8 of 16 CPU runs (p = 0.002); no shared-state window shows the bias. `wiki/research/gpu-cpu-fire-gap-2026-11-09.md`.
  - The benchmark commit after it: `tools/backend-bench.mjs` and `tools/flamescapes-fingerprint.mjs` (new); on the same grid the CPU reproduces the GPU in Duplex, Office, Clinic and Vienna at about a ninth of its speed, but the app gives the GPU a grid ten times larger, which changes the answer more than the backend (Vienna 380 against 262 nodes untenable); two bit-identical CPU changes (advection, pressure solve) cut the CPU's time 12 to 26 %. Of the four proposals the user said "build 1 to 3" (the same grid on every device by default, the GPU's speed spent on repeat runs shown as a range, the finest grid kept as an option): built, see "Resume here (2026-11-09)". Proposal 4 (slow CPUs sized by measured speed) was not asked for. `wiki/research/backend-benchmark-2026-11-09.md`.
  - The user's stance (2026-11-09): WebGPU is the reference, fast and used for big fires; the CPU must reproduce it and must not be slow or laggy; fires honest but visible. The solver is not calibrated; the Worker and Methods keep saying so.
- **Released before it:** 41a1426 (2026-11-08 night, the user's "release it after fixes"): layered walls, roofs and slabs (IfcBuildingElementPart) are read into the fire grid and the walkable graph (`ViewerManager.extractEgressGeometry`, `tools/ifc-extract.mjs`), and the graph's "Reachable" skips links blocked by walls (`core/graph.js`), so Vienna reads 90% (95% before both). The user accepted that part of Vienna's garage ramp stands apart behind basement walls with no opening.
  - **Measured (Node, Vienna):** rooms that keep their air in with every door closed 0 of 82 before, 8 of 76 at 0.27 m and 4 of 78 at 0.34 m after; the remaining leaks are door meshes that are not closed solids (24), slab edges and stair openings (17), layer junctions (12) and windows (11). Fire set-up 0.11 to 0.19 s at 30k cells and 0.17 to 0.48 s at 160k; solver steps unchanged; graph build 183 to 201 ms. All 92 openings in layered walls are cut through the layers; rooms reaching the main graph 106 to 91 (15 shafts and the refuse room, whose only door is an outside door).
  - **Gates:** 370 unit tests, lint, the Pages build, the browser suite on the Pages build desktop (110 steps) and mobile (109), 0 failed; live check 110 steps, 0 failed, 0 console errors, Render awake. The backend did not change.
  - **Web-ifc sessions:** `COORDINATE_TO_ORIGIN` shifts each session differently, so meshes compared or merged must come from one session (`smoke-path-check.mjs --add` uses a second one and can misplace them).
  - **IFCNet in the app:** Model step, "What the elements are", "Identify elements", then "Identify N by shape (IFCNet)"; the button shows only on models with generic elements the rules placed by size or not at all (Clinic 88, NIBS Office 18, buildingSMART scene 5; none on Vienna, the Duplex and the two CC0 examples).
- **Released before it:** c272fbd (2026-11-08 evening at the user's request: the egress-by-storey facts, after 2032e73 Turnstile and the site rebuilt with `DESCRIBE_URL` and `TURNSTILE_SITE_KEY`). The Cloudflare describer works on the live site: the user's phone passed the human check and got a description (after d74723b). Live check of c272fbd: 110 steps, 0 failed, Render awake; Cloudflare Turnstile's own frames log in a headless browser (a 401 token probe, CSP refusals in its iframe, hidden logs), which the browser test now counts apart from the app's errors.
- **Released before it:** 58dc781 (the design-fire item's flame drawn; live check 109 steps, 0 failed, 0 console errors).
- **Released before it:** 17f1889 (2026-11-08 afternoon, the user's "yes to all": 1f21594 (handoff), 696683a (the sprinkler activation fix) and 17f1889 (the first item as a design fire; the workflow, open office and server room GIFs re-recorded, the 2026-11-07 media in `public/docs/media/2026-11-07/`). The backend did not change, so Render was not redeployed.
  - **Gates before it:** 365 unit tests, lint, the Pages build, the browser suite on the Pages build desktop (108 steps) and mobile (107), 0 failed, 0 console errors.
  - **Live check of 17f1889:** 108 steps, 0 failed, 0 console errors, with Render awake; the new GIFs and the archive serve from the live site.
  - **Build the Pages site from PowerShell** (`$env:BASE_PATH = "/Topologic_Studio/"`, then `node node_modules/vite/bin/vite.js build`) and restart `vite preview` after a rebuild: from Git Bash the base path becomes a Windows path and every browser-test step fails (it happened again on 2026-11-08).
- **Released before it:** 977978f (2026-11-08 morning, "Release it and update readme"): fc71f77 (the interface: hints as one line, the phone layout), c880a4b (the second round of the sprinkler work), 8b3b646 (sprinkler fixes; the Office and Clinic examples), 3144a8e (README and the /docs/ guide), de743ad and 6d03f01 (suppression systems by room use; the spray drawn as a cone), 1619bec (the suppression GIFs and video) and 977978f (README: what's new). The backend did not change, so Render was not redeployed.
  - **Gates before it:** 358 unit tests, lint, the Pages build, the browser suite on the Pages build desktop and mobile (0 failed steps, 0 console errors).
  - **Live check of 977978f:** 0 failed steps, 0 console errors, with Render awake.
  - **Local tooling:** `topologicpy-web-frontend/node_modules/.bin` was found empty on 2026-11-08 (OneDrive), so `npx eslint` and `npm run build` fail with "not recognized"; run `node node_modules/eslint/bin/eslint.js .` and `node node_modules/vite/bin/vite.js build` instead, or `npm rebuild` to relink.
  - **Live check of ea2ec33** (the release before): 0 failed steps, 0 console errors, with Render awake.
  - **Live check of ba32e6d:** 0 failed steps, with Render awake; the replay (4.0 s) and the as-built doors (14 of 81) pass live. Before it, the suite passed on the dev server and on the Pages build, desktop and mobile.
  - **Live check of 7e009e6:** 0 failed steps, with Render awake; before it, the suite passed on the dev server and on the Pages build, desktop and mobile.
  - **Live check of 3f9a4f2:** 0 failed steps, with Render awake.
  - **Deployed 2026-11-06:**
    - 34d036b, the TU Wien Vienna example;
    - 17cf2ff, the fix for the blank band;
    - d0d1749, field notes and hints, and the doors switch;
    - 8852fe2, the human check (Turnstile), which stays off until its keys are set;
    - 3f9a4f2, the measures.
  - **Live check of 4cfcefb:** 0 failed steps.
    - The rooms (LOD1) step passed live: the plan opened in 0.3 s and its graph joined both storeys.
    - **The first attempt** failed only the TopologicPy cross-check: the Render backend did not answer `/health` for about ten minutes.
    - **Once it answered,** the rerun passed in full; the cross-check took 69 s.
  - **Deployed this evening:**
    - da7064f, "Describe this model";
    - d9afc1c, the code check (live check 0 failed steps);
    - 96170e4, the Worker's room suggestions (the page's client only; the Worker is not deployed);
    - eea6d1d, rooms (LOD1) models;
    - 4cfcefb, the drawing core (no screen yet).
  - **Deploy route:** fast-forward `origin/main` to `next/browser-core` (`git merge-base --is-ancestor origin/main HEAD`, then `git push origin HEAD:main`). The local `main` is a stale ref.
  - **Watching the deploy:** `gh` is not installed. Read the runs from the public API at `https://api.github.com/repos/libishm1/Topologic_Studio/actions/runs?per_page=4`.
- **Uncommitted on purpose:**
  - `wiki/research/free-inference-gpu-options-2026-11-05.md`: its drafts speak in the user's voice and the repository is public. The user decides.
- **The drawing-to-LOD1 core is committed but has no screen yet:**
  - `src/core/drawingToLod1.js`, its 9 tests, `tools/dxf-lod1-check.mjs`;
  - the method, results and failure modes: `wiki/research/dxf-floor-plans-to-lod1-2026-11-05.md`.
- **Gates before any deploy:**
  - `npm test` in `topologicpy-web-frontend`: 370 tests;
  - `npx eslint .`;
  - the Pages build: `MSYS_NO_PATHCONV=1 BASE_PATH=/Topologic_Studio/ npm run build` from Git Bash;
  - `node tools/browser-test.mjs --url http://localhost:4173/Topologic_Studio/`, desktop then `--mobile`, against `vite preview` on that sub-path;
  - the live check after the deploy;
  - the Worker's own tests: `node --test test/index.test.mjs` in `cloudflare/describe-worker` (8 tests);
  - after any solver change, `node tools/flamescapes-gpu-check.mjs --phase full --quick` (CPU and GPU agreement, now including soot, CO2 and O2) and `--phase timing`.
- **Lean:** 38 modules, 1569 theorems (2026-11-08: `SprinklerActivation.lean`, 35; `axioms-all-2026-11-08.txt`, 547 prints); on 2026-11-07 `Sprinklers.lean` (28), `Performance.lean` (14), `DoorQueue.lean` (16), `PreMovement.lean` (10), `Suppression.lean` (58), then `Federation.lean` (16), `Obstructions.lean` (36), `Extinguishers.lean` (47) and `Wetting.lean` (12), standard axioms only (`formal/fireegress/axioms-all-2026-11-07.txt`, 512 prints). `Extinguishers.lean` found the count rule's comment wrong for default ratings (`count_default_rating_counterexample`); fixed.

## Resume here (2026-11-09)

The session stopped at the user's word (usage at 97 %), then went on ("continue"). Main is 0f256f0; the worktree has next/browser-core checked out; next/calibration is behind it at d0a1ffc. Still open: the state-machine audit, area 1 done on 2026-11-10 (below).

### Built after it, released as 228dc6d: the people options, recording and the demo

The user liked the live site (0f256f0, tagged `live-2026-11-09-people`; restore by pushing the tag to main) and asked that it be preserved: every new feature is an option, off by default, except the stair gap. Released at the user's word.

- **Gaps on stairs** (core/crowd.js `stairGap` 0.9 m): a person steps on to a stair node only when no one else stands within 0.9 m of it on the flight.
- **Gathering places with a capacity** ("Full gathering places clear", 10/25/50/100 m²): a place holds its area at GATHER_AREA (0.64 m², the sunflower spacing) a person; once full, each newcomer takes the earliest's place, who fades (`goneAt`, shader fade over 2.5 s).
- **Egress with the fire** (option "Egress": without / with the fire): people keep the fire's playback time (Flamescapes frame times, else steps x seconds a step, smoothed between ticks; after it has played out they carry on at their own rate; before a fire they wait). With a finished fire run, `closes` = each node's untenable time by the planners' rule (forecastFromFrames(...).untenable, or buildForecast(...).untenable for the graph models; not `closes`, which is capped at the horizon): every 2 s the ways out are worked out again with a closed place costing FIRE_PENALTY (1 km), no one steps from an open place into a closed one, those the fire reaches walk out of it, and those whose only way out is through it stay put and are counted cut off. The user asked for it as an option.
- **The runner** ("Down a stair, on a loop" as before, or "The forecast's way out, on its timeline": core/crowd.js `forecastRunner` on the plan's earliest route and times).
- **Record GIF** (viewport button; lib/recorder.js reads the canvas just after each draw via ViewerManager.onRendered, workers/gif.worker.js encodes with gifenc 1.0.3, MIT, frame by frame): desktop 640 px, 10 fps, 30 s; phone 360 px, 8 fps, 15 s.
- **Guided runs set the fire origin themselves** (core/guided.js `guidedOrigin`): the likeliest ignition point that snaps to a node reaching an exit, nearest the route and its node, 3 m clear of the start and the exits; else a node part-way along the route. Restores the "ignition" field note.
- **Demo: "Vienna on the GPU"** (welcome screen guided run): open, graph, start, exit, route, origin, people (with the fire, forecast runner, dots on a phone), Flamescapes 120 s (60 s on a phone), the forecast plan, then a tour (ViewerManager.tour: one turn round the building, ended by touching the camera) while it all plays again. About 25 s on this PC's Intel GPU.
- **The describer's facts** (core/describeFacts.js) say what the animation showed when it was run, labelled an illustration.
- **Phones**: at most 1,500 people, dots in the demo, a smaller and shorter GIF.

### Built: people leaving (the user's egress animation)

The user asked for an optional agent simulation: simple but convincing human figures on the floors, people as dots flowing along the graph to their nearest exits and gathering there, a single person running down a stair on a loop, on the GPU, fast but grounded in the literature, using their animation folder (two agent-based egress papers and five Mixamo FBX clips).

- `core/crowd.js` (new): `simulateCrowd`, a floor-field walk on the graph (one person a node, Burstedde et al. 2001) to the nearest exit by the performance check's own `walkTree` (Carrera et al. 2019, Park et al. 2019), speeds from the SFPE hydraulic model (0.85 k = 1.19 m/s, k (1 - 0.266 D), stairs k = 1.08) and door flow `doorCapacity(0.9)` (0.79 p/s); out, people gather a few metres past the exit at ground level. `stairRunner`: one walk down the most stair nodes, looped. 8 tests in `core/crowd.test.mjs`.
- `viewer/crowd.js` (new): each walk uploaded once as float textures; the vertex shader bisects for the playback time, so the GPU moves everyone (one draw call, no work per frame on the page). Figures (shirt in the status colour: grey, blue walking, amber held up, green out, red still inside; trousers; head; swinging limbs) or dots; drawn through the building after a depth clear, clipped by a storey section. Not WebGPU compute: positions would have to come back to the WebGL view each frame.
- Route step, "People leaving" (off by default; zero cost off: a 4 KB chunk loads when switched on): look, playback 1, 5, 10 (default) or 30 times real time, play, pause, from the start, the runner switch, and the GPU drawing the view (WebGL's unmasked renderer) so the user can see whether Chrome chose the Quadro. Duplex 44 people, out by 35 s; Vienna 197, out by 305 s (189 through one exit, the single stairwell).
- Not used: the Mixamo FBX files. Their terms allow use in a project but not redistributing the files, and the repo is public; the figures are procedural instead. Offer the Mixamo runner if the user accepts that.
- A bug of the audit's kind was found and fixed while building it: the effect that worked out the walk depended on its own state, so marking itself busy re-ran it and its cleanup cancelled the walk.

### Built: "build 1 to 3" (on `next/calibration`)

The standard grid by default on every device, repeat runs on the GPU shown as a range (off by default: the user did not answer whether to default it on, and heavy features are off by default), the finer GPU grids kept as options. Details in `wiki/research/backend-benchmark-2026-11-09.md` section 5. Gates: 379 unit tests (new `core/gridPolicy.test.mjs`), lint, the Pages build, the browser suite desktop on the real GPU (two new steps: the standard grid on the card; 4 repeat runs gave a range in 9 s) and mobile. Open for the user: repeat runs on by default on the GPU or not.

### Asked alongside (2026-11-09): OpenIFCViewer and the GPU

The user shared ODA's OpenIFCViewer documentation (`reference/*.pdf`, ODA MCP Servers) and its installer, and asked what to learn and to decompile the installer with ILSpy. Declined: the viewer is native C++ (Qt 6), not .NET, and ODA's licence will not allow reverse engineering. Only the installer's file table was read (Windows Installer database, read-only): Qt 6.9.3; render devices for DirectX, OpenGL, GLES2 and GDI; a thread pool; partial viewing and a spatial index; IFC2X3, IFC4, IFC4X3 and IFC4X4; BCF 2.1 and 3.0; IDS and SDAI validation; COBie; HDF5; an embedded MCP server (`OdMcp`, `IfcMcpToolset`) with Python geometry helpers. Ideas for us, none started: BCF export of the code check's findings, IDS checks, a tool interface (MCP) on the backend, discovery-first model summaries.

### The state-machine audit (asked 2026-11-09; all three areas done 2026-11-10, on `next/browser-core`, not released)

The user asked: find the app's explicit and implicit state machines, document their states, invariants and transition tables in one file; identify invalid states or transitions the code can actually reach, with evidence, including through retries and concurrency; prioritise confirmed bugs by impact, group related fixes into issues, fix them with regression tests, and keep the document updated. On 2026-11-10 the user said to do it **one area at a time, for critical infrastructure**.

- The document: [architecture/state-machines.md](architecture/state-machines.md) (linked from the wiki index). Fixed issues also in [issues/known-bugs-fixed.md](issues/known-bugs-fixed.md); the open one in [issues/open-risks.md](issues/open-risks.md).
- **Area 1, fire runs and compute workers: done.** One read-only agent mapped 10 machines and 12 findings; each was traced again by hand: 11 confirmed, 1 suspected (A9, open). Fixed as three issues: SM-1 (a fire run outliving Stop, replacement or a restart: A1, A3, A4, A8), SM-2 (one forecast worker shared by the plan, the two-zone fire and the report, now one lane each, and a "starting" state for the two-zone fire: A2, A7), SM-3 (async actions without a starting state or a check: A5, A6, A10, A11, A12). A11's fix goes a little past the finding: moving a point drops the agent's finished answer too, as it drops a plan.
- Regression tests: unit (`core/workerJob.test.mjs` new, `core/tenability.test.mjs`, `core/guided.test.mjs`; 400 unit tests pass) and nine browser checks in `tools/browser-test.mjs` (it now counts the workers the page starts and the jobs sent to them). All nine fail on the build before the fixes and pass after; the identify-by-shape check cannot run on the Duplex (nothing left to identify by shape). Desktop 129 steps and phone 128 steps, 0 failed, 0 console errors, on the fixed build at `vite preview` 4173.
- **Area 2, network, backend and storage: done (2026-11-10).** 18 findings: 13 confirmed, 5 suspected; 11 fixed as SM-4 (the graph's copy on the server: id kept beside the graph, uploaded on demand, sent again if lost; engine with no server), SM-5 (answers for changed inputs; Stop during the human check; a description tied to its model), SM-6 (graph ids no longer listed by `/health` and `/api/graphs`; a stale tab says to reload; a timeout is not "offline"). Open: B8, B15 (confirmed, low) and five suspected. New checks: `tools/server-race-test.mjs` (7, against the real local backend, holding back or refusing chosen requests; all fail before, pass after) and `tools/describer-test.mjs` (needs a build with `VITE_DESCRIBE_URL`; its `--check model` did not reach its assertion and was set aside). The backend change (B11) is live on Render (deployed by hand; `/health` gives counts only, checked 2026-11-10). Build only from PowerShell: Git Bash rewrites `BASE_PATH=/Topologic_Studio/` into a Windows path.
- **Area 3, model, graph and workflow: done (2026-11-10).** 16 findings: 15 confirmed, 1 suspected; 11 fixed as SM-7 (a build overtaken by a model load, the b key, a fire start surviving a model change or rebuild, a rebuild keeping the last fire and the agent, Topology JSON and back, stuck progress cards, a second model left in the scene, markers, late identification). Open: C10, C11, C12, C13, C16. New check: `tools/model-race-test.mjs` (5; all fail before, pass after). **The audit is complete**: all three areas mapped and fixed, released at 66e3315 (backend on Render too); gates for a release: unit tests, lint, the Pages build, `browser-test.mjs` desktop and `--mobile`, `server-race-test.mjs`, `model-race-test.mjs`, backend pytest, then a manual Render deploy for the backend change (B11).
- Release: the fixes are on `next/browser-core` only; main (713c697) is unchanged until the user says to release.

## Resume here (2026-11-07)

00. **Asked for on 2026-11-07:**
   - **Sprinklers and suppression: released (ea2ec33); the second round committed, not deployed.** The user asked for the five items the first round left: a second (MEP or structural) model (`core/federation.js`, `ViewerManager.loadLinkedIfc`, Model step), beams and ducts as obstructions (`sprinklers.js` `checkObstructions`), water wetting the fuel (sprinkler mode "wetting", `WETTED_FRACTION` 0.52 from `tools/wetting-calibrate.mjs`), validation against Vettori, Bittern and UL/NFPRF (`tools/sprinkler-validation.mjs`: the estimate is late in rooms, mostly from the solver's 200 W/(m² K) surface coefficient, which the user had left open; changed on 2026-11-08, see This round), and extinguishers (`core/extinguishers.js`, Simulate step). Also water by fire class (`fireClasses.js` `waterResponse`; the user asked how the flame reacts by its type), the spray drawn under open heads, and the doors and windows wording (closed is airtight; `tools/envelope-check.mjs`). Read the "Second round" section of `wiki/research/fire-suppression-sprinklers-2026-11-07.md` first, then `sprinkler-validation-data-2026-11-07.md` (R1 to R9), `extinguishers-2026-11-07.md` and `water-by-fire-class-2026-11-07.md`.
   - **Branch tidy-up, waiting on the user.** All four GitHub branches were fully merged into `main` and no PR was open. Deleting `next/studio-2` (tip 0f68573) and `codex/wiki-hitl-testing-docs` (tip d56730e) on GitHub, and the merged local branches, was refused by the permission system as destructive. The user can run: `git push origin --delete next/studio-2 codex/wiki-hitl-testing-docs`, then `git branch -d codex/wiki-hitl-testing-docs next/studio-2 worktree-agent-a5564f61ecc07c56c worktree-agent-a6e3e35eb51c00dc1 worktree-agent-ab4bdc24cb476c1b2 worktree-agent-ab4f7303ef3afd806` and `git branch -f main origin/main`. Restore a deleted branch with `git push origin <tip>:refs/heads/<name>`. The worktrees under the Classic checkout (`TopologicStudio/.claude/worktrees`) and `TopologicStudio_push_20260311_100658` were left alone (Classic is not to be modified); their branches are merged too.
   - **Outdoor assembly points:** a check is proposed in section 7 of the sprinkler note; the user decides.
   - **The user's decisions (2026-11-07):**
     - release the sprinkler and suppression work as a whole once it is built, not step by step;
     - the university building's egress waits for a future session;
     - codes: pass SBC 801-CR-2018, with IBC 2021 where stricter, until the user obtains SBC 2024. Done for the code check (`EGRESS_B` in `core/codeCheck.js`, both tables kept: `SBC2018_B`, `IBC2021_B`). Committed to `next/browser-core` but **not deployed**: main stays at the last release until the sprinkler work is built and released as a whole.
   - **Simulated-fire (performance-based) compliance check: built, committed, not deployed.** Route step, "Performance check (with the fire)", beside the code check. It reads the forecast's escape-deadline map (Plan with the forecast, map on) and judges each storey against detection, alarm, pre-movement and the queues at doors (`core/performanceCheck.js`, `core/rsetPresets.js`, `core/perfReport.js`, `Performance.lean`, `DoorQueue.lean`, `PreMovement.lean`; values and sources in `wiki/research/rset-components-2026-11-07.md`, whose "As implemented" note lists what each part does). The user's references (SinaPf's BIM-based fire code checker, IAAC's IFCore checker, and Mohammadi, Abolkhair and Altassan 2026, Sustainability 18(16) 8023, doi:10.3390/su18168023, checked at Crossref: a Solibri rule check of SBC 201-CR-2024 egress on a three-storey R-1 hotel) are rule checkers on the IFC; this check is the performance-based side they do not cover. The five follow-ups the user asked for are built: virtual ceiling detectors (up to 16 round the fire at C/VM2's 7 m or as far as its room allows, CPU and GPU), the measured Lovreglio 2019 Weibull beside the design log-normal (chart and per-storey verdict, not in the verdict), the ABCB no-response share, queues at doors (Lindley's recursion, C/VM2 door flow, IFC widths), and a report of several fires (the exit nearest the fire blocked and the first detector failing as sensitivity cases; two fires a storey added automatically for graph and two-zone fires; one HTML page to download). **Left:** the M3 refusal, C/VM2's cap of 50 persons a minute through a self-closing leaf, stair flow, speed classes and the ABCB disability increments, a faster or smouldering fire as an automatic case, and Flamescapes fires added automatically (a run takes minutes; today the user moves the fire and plans again).
   - **Smoke transport (report in `wiki/research/smoke-transport-2026-11-07.md`).** Three solver gaps were confirmed in `flamescapes.js`:
     - no thermal expansion in the projection;
     - one 200 W/(m² K) gas-solid coefficient for fuel and walls alike;
     - open grid sides, so a clipped model leaks.

     The README and the in-app limits now state the first and third. The checks to run first are C1 (closed box, NIST/NRC Test 1), C3 (Steckler) and C4 (NBS multi-room, ATF corridors); each needs a prescribed-heat-release burner first.
   - **SBC documents:**
     - The current edition is SBC 801-CR-2024. Its open table of contents is on sbc.gov.sa (`/ar/BC/Documents/tableofcontent2024/SBC 801/`); the full text is sold and was not obtained.
     - The user's Google Drive holds SBC 801-CR-2018, an 848-page licensed copy stamped for internal use only. It was read locally for Section 903, and values from it are in the sprinkler note. **Never commit it or quote its text**; cite clause numbers and values only.
     - No SBC 201 text has been found. The only SBC file on disk is SBC 601 (energy), in the Shoaib references folder.

0. **In order, as the user asked on 2026-11-06:**
   1. **Cloudflare: the user's steps, then the deploy.**
      - **Plugin:** Cloudflare's Claude Code plugin is installed for the user (`claude plugin marketplace add cloudflare/skills`, `claude plugin install cloudflare@cloudflare`), as the user asked from developers.cloudflare.com/agent-setup. It was reviewed first: it is from the Cloudflare organisation, has no hooks, and has one MCP server, `https://mcp.cloudflare.com/mcp`. The optional `cf` CLI was not installed: the page says to ask first.
      - **Signed in (2026-11-08):** the user ran `npx wrangler login` (wrangler 4.158.0, one account). Never sign in for them.
      - **Worker deployed (2026-11-08, at the user's request):** `topologic-describe` on the user's workers.dev subdomain (the address is in the user's chat, not committed here); its 8 tests passed first; `/health` 200 and one test `/describe` from the live site's origin answered (CORS allowed). No `TURNSTILE_SECRET` yet, so it serves any allowed origin without the human check.
      - **Turnstile (2026-11-08):** the user made the widget in the dashboard (site key `0x4AAAAAAF4v1X9WmvwLdAJU`, Managed, domains `libishm1.github.io`, `localhost`, `127.0.0.1`) and pasted Cloudflare's Turnstile Spin prompt (existing-widget flow, `developers.cloudflare.com/turnstile/spin/prompt.md`, read as data). Done to its contract, each step approved by the user:
        - the Worker accepts a token only on Cloudflare's success, for the route's action (`ACTIONS`: `describe`, `rooms`), from a hostname in `TURNSTILE_HOSTNAMES` (`wrangler.toml`: the live site only; a secret with no hostnames refuses all), with a 10 s timeout; the page's check renders action `describe`; 8 Worker tests; the Worker redeployed;
        - the secret fetched and stored by the guarded flow, never shown: jq 1.8.2 (official binary, SHA-256 checked) and launchers `python3` (Python 3.13) and `wrangler` in `C:\Users\lmurugesan\tools\bin`, wrangler 4.158.0 installed with `npm install -g` (esbuild and workerd install scripts left off); widget metadata checked first, the secret checked with siteverify's dummy token, then `wrangler secret put TURNSTILE_SECRET`; a request without a token now gets 403;
        - Cloudflare's test keys report hostname `example.com` and no action, so they cannot pass the Worker; local tests use the real widget with `wrangler dev` and a git-ignored `.dev.vars` (README).
      - **Then (the user):**
        - set the repository variables `DESCRIBE_URL` and `TURNSTILE_SITE_KEY` (the GitHub CLI is not installed here, so the user sets them in Settings, Secrets and variables, Actions, Variables), then the site is rebuilt (a push to `main` does it) and a live request is tried with a fresh token, and the token replayed must be refused. Until then the live site does not offer Cloudflare.

        The Worker's README has the steps.
   2. **Smoke and gas measures:** done (3f9a4f2). See This round and `wiki/research/tenability-index-2026-11-06.md`.
   2b. **Asked for during the video (2026-11-06), not built:**
      - **Shelter areas outside the building:** identify safe assembly areas outside when there is a fire. Read the guidance first (distance from the facade, clear of exits and fire-service access, upwind), then propose the method before building it.
      - **"Nearest exit" stopped at a flat's own door in the Vienna model:** fixed (7e009e6); the code check now counts the exits out of the building and the stairs (0ba7352).
      - **Ignition by context:** done (0ba7352), the Cause choice.
   3. **Done on 2026-11-07 (0ba7352):** kept runs (replay from IndexedDB) and door states (as built, random). Left: a door ensemble (several seeds, the spread of results).
   4. **Training** (IFCNet retraining, on Nebius, Vast.ai or a free offer): later, by the user's choice.
      - **Wanted in the data:** the user asked to use "all of these models", the list on github.com/bimdata/BIMData-Research-and-Development, page `pages/IFC_FILES.md`. It holds 100 IFC files (3.7 GB, 40 models) from seven sources: DURAARK at TIB, the Open IFC Model Repository, BIMcollab examples, Schependomlaan, IFCCompressor, NIBS common BIM files and markusplassen.no. The repository itself has no licence and holds screenshots only.
      - **Check each source's licence first.**
        - `NBS_Lakeside` is an NBS model: excluded by the user's standing rule.
        - The `Autodesk_*` models need Autodesk's terms read first.
        - The Open IFC Model Repository states no licences.
        - DURAARK is licensed dataset by dataset.
        - Several models are already in the local corpus: Duplex, FZK-Haus, Smiley West, HiTOS.
      - **Also wanted: Sand3D** (Sandflow, sandflow.ai/sand3d; the user asked 2026-11-06).
        - **What it is:** about 100K single 3D objects, each written as a Python program in the Sand3D language that builds a mesh (for glTF, Blender, three.js): furniture, lighting, vehicles, toys, garden and outdoor, sports, articulated objects (10K+ with 36K joints) and mechanical parts. No buildings, rooms or IFC. Counts are Sandflow's own.
        - **Where:** five Hugging Face datasets, `Sandflow/sand3d-objaverse` (66,887), `-cad` (24,495), `-articraft` (9,209), `-simready` (2,334), `-automate` (3,169), all tagged CC BY 4.0; the library is Apache-2.0 (github.com/Sandflow-ai/sand3d, created 2026-11-03). A model is "release soon" per the page.
        - **Licence, read on all five cards:** the programs, renders and USD files are CC BY 4.0. Each record is derived from a third-party object whose own licence also applies, and its per-row attribution must be kept (`source_license`, `attribution` columns). Upstream: Objaverse CC BY 4.0 or CC0 per object (Objaverse as a whole ODC-By), NVIDIA SimReady CC BY 4.0, Articraft CC BY 4.0 ("Articraft Data, Matt Zhou et al."), CADFS CC BY 4.0, AutoMate CC0. None is non-commercial or share-alike.
        - **Autodesk:** 16,766 of the CAD objects derive from Zero-To-CAD (ADSKAILab, Autodesk's AI lab). Its card was read: Apache-2.0, not gated, no further terms, so it passes the user's rule; keep the Apache notice and a statement of changes.
        - **Use here:** furniture and fixtures for the ignition-relevant furniture classes (IFCNet phase 1), as extra meshes or point clouds beside the IFC elements. Filter on each row's `f_score`, the pipeline's own measure of how close a program's shape is to its source. Keep the attribution columns with any derived set.

Items 1 to 4 below are unchanged from the night before.


The user accepted three steps for floor-plan drawings to LOD1. Step 1 is half-built, step 2 is built but not on screen, and step 3 is deferred.

1. **Drawings to LOD1 (step 1).**
   - **Done:**
     - the rooms model and its import, live (eea6d1d);
     - the drafting core, with its match rates checked here (This round).
   - **Next: the review screen.** Pipeline: `decodeDxfBytes`, `readDxf`, `suggestLayers`, `draftRooms`, in a worker or a dynamic import.
   - **What the user confirms per storey, in order:**
     1. the sheet region and scale (one DXF often holds several plans);
     2. the wall layers (suggestions pre-ticked);
     3. a common point to stack the storeys;
     4. floor level and height.
   - **Room review works from the draft's issues list.** The edits needed, most needed first:
     1. merge;
     2. split by a line snapped to wall ends;
     3. rename and set class;
     4. mark exits (the hotel's sheets have no entrance door blocks);
     5. delete or draw rooms, and add or remove doors.
   - **Keying edits:** key them by room position, so a re-draft keeps them. `keepLines: true` returns the linework to draw under the draft.
   - **End:** `toLod1Model([{level, draft}], levels)`, then `openLod1(model, {name})` in App.jsx, the entry point the JSON import uses.
   - **Merging needs a polygon union.** It is simple where drafted rooms share edges, since they are faces of one planar arrangement. It is not simple after the rooms have grown to the wall centre lines.
2. **AI room suggestions (step 2).**
   - **Built:** the Worker endpoint and the page's `suggestRoomsInCloud` (96170e4).
   - **To do:** show them in the review screen.
   - **Deploying the Worker needs the user's own Cloudflare account:** `npx wrangler login`, then `npx wrangler deploy` in `cloudflare/describe-worker`, then the repository variable `DESCRIBE_URL`. See that folder's README. Do not create accounts or sign in for the user.
3. **A fine-tuned vision model for scanned plans (step 3).** Deferred by agreement. It is worth doing only if many scanned plans must be handled. Ask the user before starting.
4. **IFCNet retraining:** paused by the user ("use the IFCNet weights themselves"). The shipped weights stay.
   - **To resume:** start from `STATUS.md` in the IFCNet working folder outside OneDrive, and the phase 1 plan in `wiki/research/ifcnet-furniture-study-plan-2026-11-05.md`.
   - **Before a paper:** read Utkucu et al. 2024 (doi:10.1016/j.aei.2024.102503) in full. It is the closest precedent for fusing names, rules and geometry, so the claim must stay narrow: ignition-relevant furniture sub-types, leave-buildings-out, in the browser, open weights with provenance.

Before any deploy: the gates above, then the live check, then update this file.

## This round (2026-11-08)

- **Released 977978f** (the user: "Release it and update readme"): see State.
- **Sprinkler activation fix** (the user: "plan, ideate, fix the heads starting late and the large room problem, check Lean, derivations, literature, metrics, and test across IFCs"; this was the user's go-ahead on the surface coefficient). The plan, diagnosis, results and what is left: `wiki/research/sprinkler-activation-fix-2026-11-08.md`; sources: `wiki/research/sprinkler-activation-literature-2026-11-08.md` (an agent; FDS medians recomputed and three DOIs checked by hand). Four causes, three fixed:
  - suspended ceilings (`IfcCovering` CEILING) were dropped at import, so the fire grid had none and heads hung at the slab: now a `ceilings` geometry group (`ViewerManager.extractEgressGeometry`, `tools/ifc-extract.mjs`), solid in the fire grid (`buildScene`), and `roomsFromSpaces(…, { ceilings })` lowers a room's ceiling to them (they never make a space inside; in a model with ceilings, a room without one is open to the slab within 2 m); a thin element split between two cells now seals (`voxelize.js occupancy`);
  - walls took the ceiling jet's heat at the fuel's 200 W/(m² K): `wallConvection` 10 (CPU `conduct`, GPU `conductGas`, a new uniform);
  - the estimate read the grid's plume column and cell-centre distances: each head's own `H` and exact `r` (`sprinklerSetup`; `ceilingAbove` on a record overrides the 0.13 m link offset);
  - the fire grew to about 500 kW within 15 s (faster than an ultra-fast t-squared fire), so heads opened at 250 to 600 kW: fixed by the design-fire first item below, at the user's word.
  - Results: against NIST every measured head opens within twice the measured time, first heads at medians 0.78 to 1.26 (FDS 0.73 to 1.13); in the app's fires the Office's 610 m² open office (no head in 120 s before) opens at 26 s and the Clinic's waiting area at 22 s (65 s at 4.5 MW before). Sprinklers now act at 16 to 39 s in the systems check (39 to 57 before): the Clinic's server room 61 % less heat on sprinklers (28 before), its waiting area 79 % with wetting (no mode helped before). Left: the Clinic's mechanical room (no ceiling over the fire; heads at 47 s against about 1 MW, no mode helped) and Vettori's 1998 room (a quarter late: idea I4, a layer from more than the top cell). Results in `benchmark/results/2026-11-08-*`. `RUN_CACHE_VERSION` 4. 362 unit tests.
  - Side effect: with walls at 10 a fire in the 5 x 4 m test room releases less heat in 120 s (41 to 48 MJ against 57 to 73 at 200), with an occasional flash of unburnt pyrolysis gas (224 MJ in one of five seeds). CPU and GPU agree as before (one step from one state: the same 0.022 K as at 200; ensembles agree).
  - The GIFs and video of 1619bec were recorded before the fix; they are kept in `public/docs/media/2026-11-07/` (below).
- **The user's answers ("yes to all"; then "rerecord just important cases, keep the website up to date and mention where old simulations are kept"):**
  - **The first item as a design fire** (`fireClasses.js designFireFor`, `flamescapesRun.js itemBurner` and `itemSuppression`, `flamescapes.js burnerHeat`, `burnerCells`, the GPU `burner` kernel; "How it grows" in the panel, default its class's growth; `RUN_CACHE_VERSION` 5). Classes A and C only; pools, metals and packs keep their own burning. Water over the item decays it as Evans measured, clean agent stops it. In 12 of 16 fires in eight models the first head now opens at 33 to 319 kW (was mostly 0.5 to 1 MW); under high or open ceilings the contents still catch first (0.45 to 1.6 MW). Section 8 of the fix note; `benchmark/results/2026-11-08-design-fire-*`. 365 unit tests; CPU and GPU identical on the item (`flamescapes-gpu-check.mjs --phase design`); no measurable cost per step.
  - **Recordings:** the 2026-11-07 GIFs and video moved to `public/docs/media/2026-11-07/` (git mv), named in the README and the guide ("Earlier recordings"). Only the important cases were re-recorded, at 180 s with the design fire, from a production build: the workflow (Duplex, wetting: the first head at 132 s and 287 kW, the item down to 2 kW by 180 s), the open office (no sprinklers 1.8 MW at 180 s; wetting: the first head at 142 s and 371 kW, 394 kW at 180 s, as the contents it had lit go on burning) and the server room (the first head at 95 s and 109 kW; the clean agent stops the flames at 100 s). The four-mode comparison, the fire classes and the video were not re-recorded and are linked from the archive. The compose label now says "1 head open".
  - **Commit 5fb678a reworded** to 1f21594 ("Handoff: released 977978f, live check passed") and `next/browser-core` force-pushed with a lease; the tree was byte-identical. A local branch `backup/before-reword-2026-11-08` keeps the old commits.
  - **Released** as 17f1889, live check passed (see State).
- **The design-fire item showed no flame** (the user, with two screenshots of a class C fire in Vienna at 63 kW: "for electrical fire the flame animation doesn't show up in its new version"). Flame-front particles spawn only in cells the solver's own combustion marks `burning`; the item's burner puts heat and products into its cells directly, so nothing was drawn at the item until the room's contents caught (every class with a design fire, A as well as C). Fixed for drawing only: `flamescapesRun.js withBurnerFlame` appends six points per cell the step's flame reached (`burnerCells`; the burner keeps `flameCells`, `flameRate`, CPU `burnerStep` and GPU `step`) to the frame's particles, coloured by McCaffrey's mean centreline temperature (`flamescapes.js plumeRise`, his Table 1: 797 K, 62.9 / z*, 21.6 z*^-5/3; NBSIR 79-1910, doi:10.6028/NBS.IR.79-1910, checked on Crossref and read in the report). The solver never sees them: a run's heat release and temperatures are identical with and without (test). `RUN_CACHE_VERSION` 6, so kept runs are computed again with the flame. Browser test step "the burning first item is drawn as flames" (90 points on the GPU). Not changed: the item's flame does not pilot nearby fuel vapour as the solver's own flame-front particles do; the contents catch from its heat alone, as in the validated runs. The re-recorded GIFs predate this (no flame at the item until the contents catch); re-recording waits for the user.
- **"Describe this model" through Cloudflare said "The model returned no text"** (the user, on a phone, Vienna; the human check passed). gpt-oss reasons before it writes and the reasoning counts against `max_tokens`: at 700 it stopped at the limit with no answer in 4 runs of 4 on the Vienna facts (550 characters), and a reasoning effort in the messages format changed nothing (measured with a throwaway `wrangler dev --remote` probe in the scratchpad, never deployed). Now 2500, as `/rooms` has; a reply cut off before any text says so (d74723b, the Worker redeployed).
- **IFCNet instead of gpt-oss?** (the user asked) No: IFCNet classifies elements by shape and writes no text; it already runs on the device; Workers AI runs only Cloudflare's catalogue. Other Workers AI text models are a one-line change of `MODEL` in `wrangler.toml`.
- **Graphify?** (the user asked whether graphify helps the model reason) Graphify (Graphify-Labs, safishamsi/graphify) maps code and documents for coding agents and does not read IFC. What helps is giving the model the graph's own answers, short: at the user's word ("Ok that sounds good"), the facts now have a line per storey (`describeFacts.js` from `codeCheck.js` on the graph): its ways out (doors out, or stairs), its longest walk to one, places that reach none, and the height of the highest occupied floor; the code check's verdicts and its basis (office, Group B, the stricter of SBC 801-CR-2018 and IBC 2021) only once the user ran it, since it judges every building as an office. Storeys are named by the IFC's storeys after taking off the model's coordination shift (`levelShift` in App.jsx; viewer height = IFC + shift: about 3 m in Vienna, -1 m in the Clinic); twelve storey lines at most, the rest summed up. Computed only while the Describe panel is open (the code check takes 2 to 15 ms on the examples, the Clinic's 33,654 nodes the most). The instructions (page and Worker, kept equal) now put the verdicts under "how people get out" and say when no fire was run: with them gpt-oss described the Clinic storey by storey correctly in 2 runs of 2 (37 to 49 neurons). `codeCheck` storeys now carry `unreached`. 369 unit tests.

## Earlier round (2026-11-07)

- **Suppression systems by room use** (the user: water in apartments, powder at strategic places, FM-200 in electrical and server rooms, foam in generator rooms, tested on the commercial types; de743ad engine by an agent, reviewed; 6d03f01 interface). `core/suppressionSystems.js` assigns each room water, clean agent (HFC-227ea at 7 %, sprinklers kept unless "Agent only"), foam or none by name and the performance check's occupancy (Group R: water only), sizes the agent and lists each special room's own extinguisher (CO2, class B foam, class K); the solver (CPU and WGSL) floods a room after cross-zoned detection and 30 s, or blankets a pool with foam. Zero cost off (bit-identical, measured on both backends). `tools/systems-check.mjs` (`benchmark/results/2026-11-07-systems-check.txt`): server rooms 65 and 28 % less heat with sprinklers, 95 and 93 % with the agent; a heptane spill 26 % with sprinklers, 41 % with foam; checked again by hand on one seed (58 and 94 %). Sources in `wiki/research/suppression-systems-by-room-2026-11-07.md` (SBC cited by number and value only). Not modelled: agent cooling, oxygen dilution, HF, leakage, doors closing on release.
- **GIFs and video** (`tools/suppression-media.mjs` captures the app per fire step from a production build, never the dev server; `tools/suppression-compose.mjs` composes GIFs and an MP4 with MediaRecorder; no ffmpeg). Found while making them: the spray was drawn as one helix per head (all three droplet seeds came from one golden-ratio step; now Roberts's R3), the capture's first frame read the last run's card and its final frame overwrote step 0, the page kept the last case's settings, and the video crashed the page when all frames were sent at once. The headless page crashed at the end of 2 of 3 early spray runs and not since; the capture retries a crashed page once.

- **Sprinkler testing on several models and fires** (the user: "test the sprinklers for different IFC models and fires; why are the sprinklers on the roof above the Duplex?"; 8b3b646). Found and fixed: heads hung above roofs and terraces (every IfcSpace was a room with its top as the ceiling: the Duplex's "Roof" space, Vienna's terraces and balconies, the Clinic's six "ROOF" spaces; `roomsFromSpaces` now looks up for the real slab and leaves out spaces outside the building); the NIBS Office never burned (no floor slabs; `core/spaceFloors.js` gives the fire grid the spaces' bottoms). Measured (`tools/sprinkler-check.mjs`, `tools/sprinkler-seeds.mjs`): wetting cut most fires by 20 to 79 %, Evans's cap and the spray ranged from 40 % less to 78 % more, and a fire's own scatter is large (the same Duplex fire released 137 to 272 MJ with three contents layouts; a Clinic room 138 to 547 MJ). In the Clinic's large rooms heads opened late (63 to 68 s) and no mode helped: the late activation the validation found. Wetting is now the default mode. The NIBS Office and Clinic are site examples. The remaining merged branches were pruned one at a time: six local ones and `codex/wiki-hitl-testing-docs` on GitHub; `next/studio-2` on GitHub is left for a later pass, as the user asked.

- **Released ea2ec33** (the user: "release it now"): unit tests and lint, then `main` fast-forwarded; Pages built in about 3 minutes; the live check had 0 failed steps.
- **Performance of the release** (`benchmark/results/2026-11-07*.json`): the corpus benchmark with five repeats on the models that looked slower in the quick run is the same as or faster than 2026-10-28 (Yen on the largest model 52 to 40 ms); `tools/feature-bench.mjs` (new) times the plan's deadline maps (Vienna 1.43 s, the slow part), the performance check (1.9 ms; 11.8 ms with door queues) and the sprinkler layout (22 ms); the spray stage costs 0.01 ms a step.
- **Interface** (fc71f77; the user: hints "should not be in the way"): hints and found field notes are one line (47 to 58 px, were 200), wait 2.5 s, never offer the step on screen; the phone page was 412 px wide on a 390 px screen (the header's minimum width set the shell's grid); the graph-parts warning is one sentence outside the Graph step.
- **Second round of the sprinkler work** (the user's five items, then two questions mid-way): see Resume here. Agents wrote the obstruction rules, the validation harness and its burner, the extinguisher module, the Lean modules and three research notes; each diff was reviewed. Measured: NIBS Office with its HVAC model, 292 heads 5 cm under the architecture's ceilings; Simple-Scene plumbing through two map conversions, checked by hand to 0.1 m; wetting CPU 54.3 and GPU 52.9 MJ against 72.5 and 66.7 with no water; Duplex extinguishers 4 placed in 215 ms.
- **Asked mid-way:** a sprinkler animation (built: droplets in a cone under each open head, moved by the shader); whether the water wets the flame (no: drops cool the fuel's surface, not the flame; now by fire class); whether smoke leaks with doors and windows closed (no: closed is airtight; measured 0.0 g of 138 g of soot outside on the Duplex after 60 s; with the doors open, the default, 24 %; a grid clipped round the fire lets smoke out at its sides, which the run card now says).

- **Sprinklers and suppression, steps 1 to 4 (the user: "work through 1 to 4"):**
  - **Placement and IFC** (an agent; reviewed): `core/sprinklers.js` and its 19 tests; IFC4 and IFC2x3 sprinklers with the Revit sets; per-room NFPA 13 layout with extra heads where a bay centre is not clear of the walls, both orientations, merges under 1.8 m, coverage on a 0.25 m sample; storey lattice without rooms; requirement report (903.2.8, 903.2.11.3 at 16.764 m, high-rise, A-3, E). Panels count rooms by `placement` (obstructions are always "cannot tell").
  - **Activation** (`core/suppression.js`): per frame, gas held at the mean of the frame's readings, exact link step, crossing inside the frame; estimate = Alpert under the fire's ceiling (line of sight from the plume's ceiling cell) with the ignition cell as the axis, late bound = the top cell. Sensors read after the ceiling detectors; the GPU's `gather` adds their speed (`speedFrom`, `ioSpeed`).
  - **Evans and the spray**: `sup` buffer (binding 14, 16 bytes when off); `chemistry` caps the fuel (`evans`), the `spray` kernel runs one thread per column after `chemistry`. CPU and GPU agree on activation (32.2 against 31.4 to 31.9 s in `tools/flamescapes-gpu-check.mjs --phase sprinklers`) and water (35.4 kg).
  - **Found while building**: the forced flame cull removed (it, not the water, was putting the fire out); a cold covered cell now reads rate 0 and never ignites (it used to stay armed); the solver's pyrolysis gas can gather and flash, with or without sprinklers, so tests check fuel, energy and the per-cell cap, not a frame's peak; `sim.suppression` declared in `createSolver` so the engine keeps one object shape.
  - **Lean**: `Suppression.lean` (58), an agent; items 8 to 15; X30 is the case e = 0, not T' = T_B.
  - **Tools**: `fire-bench.mjs --sprinklers`; the GPU check's `sprinklers` phase and detector comparison; the browser suite's sprinkler step (46 heads in the Duplex's 21 rooms; the first opens at 21 s by the estimate in its 30 s run). Prepared examples must be rebuilt after `IMPORT_SETTINGS` changes (`npm run prepare-examples`; the build does it).

- **Performance check, second round (the user's five follow-ups):**
  - **Ceiling detectors:** `flamescapesRun.js` `ceilingDetectors`: 16 rays from the ignition cell at 1.5 m above its floor, each stopping at 7.0 m, before a wall, or where the ceiling steps by more than a cell (a lintel); the detector is the top open cell under the ceiling there, and only the rays that reach the farthest distance (within a cell) keep one. The first version took every ray's end, so a wall beside the fire put a detector over the plume and the Duplex run detected at 4 s; the browser suite caught it. Read with the nodes each frame on the CPU and the GPU (`withDetectors`), forwarded by the worker, shown on the run card ("Smoke detectors"), and timed by `forecast.js` `ceilingDetection` (first and slowest). `RUN_CACHE_VERSION` is 2, since frames carry the detectors.
  - **Measured pre-movement:** `rsetPresets.js` `MEASURED_PRE` (Lovreglio et al. 2019 Weibull clusters; each preset has a default). Judged per storey and drawn beside the design log-normal; the verdict stays on the design values.
  - **No response:** each preset's ABCB share (1 % for apartments and hotels). Never counted as starting in time; the percentiles are the responders'.
  - **Queues:** `performanceCheck.js` `doorQueues`: occupants at the area per occupant on every place, the walk tree to the nearest exit with no fire, Lindley's recursion at each door at 1.32 persons/s/m of effective width, doors upstream first. The 99th-percentile starter's wait W gives `required(tau + W) + W`; the share that starts in time is found by bisection on the exit time. Timed at 59 ms on Vienna and 67 to 81 ms on a 21,000-node local model, only when the check is run.
  - **Report:** `perfReport.js` (`planCases`, `reportHtml`). The forecast worker makes a second deadline map without the exit nearest the fire when "Also with the exit nearest the fire blocked" is on (off by default: it about doubles the map's time). "Add fires on every storey" plans up to eight fires in the worker. "Download the report" saves one HTML page.
  - **Lean:** `DoorQueue.lean` (16) and `PreMovement.lean` (10); `weibull_nonneg` needed a non-negative scale.
  - **Browser suite:** step 5c checks the queues, the measured line, the chart, the report with added fires and its download; step 5e checks the run card's detector line. 0 failed steps on the dev server and on the Pages build, desktop and mobile; the Flamescapes run still takes 15.6 to 18.0 s.
  - **GPU:** `tools/flamescapes-gpu-check.mjs --phase full` now compares the detectors: 9 on its test room, visibility correlation 0.994 between CPU and GPU, first alarm at 40 s on both.
- **Performance check (with the fire):**
  - **Rule:** a place passes when its escape deadline L (the dose-limited map, never later than the exact deadline) allows detection, the alarm and the 99th percentile of pre-movement (a log-normal through PD 7974-6's two percentiles), with the criterion's factor on RSET and margin. The criteria are: the base case, ASET/RSET at least 2 and at least 30 s over (SFEG 2025 Table 10.4, Italian code M.3.2.2; the default); a sensitivity case, 1.2; or the 99th percentile alone.
  - **Forecast too short:** where the deadline is bound by the end of the forecast rather than the fire, the check says "cannot tell" and how long a forecast it needs.
  - **Presets:** office (the default), classroom, apartments, the flat of origin, hotel and assembly.
  - **Detection:** taken from a Flamescapes run when there is one (`forecast.js smokeDetection`: OD 0.097/m, read at head height, the frame's end); otherwise the time set.
  - **Duplex probe** (graph fire, placeholder times): storey 1 had 304 fail and 600 cannot tell (it needs a 368 s forecast); storey 2 failed with deadlines near 281 s.
- **Pending tasks built (0ba7352):**
  - **Cause** in Likely ignition points (`core/ignitionContexts.js`): a gas fire at a cooktop starts as class B; "deliberate" brings in the accelerant scenario.
  - **Code check exits:** the exits out of the building plus the stairs. A storey with exit doors whose places reach none on that storey leave by a stair (a second pass), so Vienna's storey 1 has 315 places with no exit (325 before) rather than 1,910. Vienna's single stairwell now fails the two-exit rule from the third storey up (Group B values); the README says so.
  - **Kept runs** (`core/runCache.js`): IndexedDB, keyed by `RUN_CACHE_VERSION`, the converter id, fingerprints of the graph and the scene, and the run's inputs; 256 MB, least recently used first; runs over 96 MB not kept. **Bump `RUN_CACHE_VERSION` with any solver change.** The browser suite checks a replay (4 s).
  - **Doors** (`core/doorStates.js`): as built (SelfClosing, or -C in an EN 13501-2 class: 14 of Vienna's 81, rated EI2 30-C or EI2 30-C-Sm) and random with a seed. `IMPORT_SETTINGS` is 4.
- **Sprinkler research (ba32e6d and the summary note):** three agents (codes, physics, open IFC MEP models) and a fourth (smoke transport) at the user's request. Key points:
  - the SBC 2024 text could not be read (protected viewer; the English text is sold), so the triggers are IBC/IFC 2021 values to be confirmed;
  - the coarse grid delays activation, so activation is to be read two ways (top cell, and CFAST's layer plus Alpert);
  - Evans's correlation is the first suppression model; Fire-X-style evaporation second;
  - IFC gives no RTI or C for a sprinkler; defaults by class are needed.

  DOIs in the two reports were checked at Crossref by the agents and again here; Flamescapes' DOI (10.1145/3842518) still returns 404 at doi.org.
- **Lean:** `Sprinklers.lean`, 28 theorems (layout, the link's step and its exact form, activation time, Evans), all on the standard axioms.

## This round (2026-11-06, evening)

- **README animations (fbdaed5, live):** two GIFs of the Vienna runs (doors open with the heat view; the gas fire with the tenability index), made from the video's frames with `tools/frames-gif.mjs`. The encoder moved to `tools/gif.mjs`, shared with `tools/fire-gif.mjs`. A doors-closed GIF was dropped: filmed from below, it did not read.
- **Exits out of the building, and rerouting to any of them** (the user asked for the route to try the next feasible exit when a stairwell is untenable, using fire escapes but "not every external door", so that the egress makes sense).
  - **The rule** (`exitDoors` in `core/ifcSemantics.js`, field `final`):
    - where the model says which doors are external (`external`: IsExternal TRUE or an EXTERNAL space boundary, read whatever FireExit says), the exits out are the external doors at street level (within 1.5 m of the lowest), and the fire exits at any level that are external and open onto no walkable floor;
    - an external door above street level with floor beyond it (a balcony or terrace) is left out, and so is a flat's front door flagged FireExit;
    - parts of the graph none of these reaches keep their exits; a model with no external doors keeps all its exit doors.
  - **Vienna:** 35 exit doors, 21 external, two exits out: the street door, and a FireExit door at the back of the first floor that opens onto no floor (most likely the hillside at grade; not checked against drawings). From the top floor the nearest is now that fire exit (33.6 m); the street door is 53.8 m.
  - **Vienna has one stairwell.** Its four IfcStairs are the four flights, and both exits are at its foot, 1.5 m apart, so no fire there makes a sensible switch between them from the upper floors. The user was told; the switching GIF uses the Office example (two stairs, an exit at each end).
  - **Used by:** the nearest exit, hazard-aware rerouting (`runFire({exits})`, new setting **Reroute towards**, default any exit), the forecast's "All N exits out of the building" and the agent's "any exit". The code check is unchanged.
  - **Search:** `astarToAny` in `core/route.js`, A* to the cheapest of several ends (heuristic: distance to the nearest). `astar` itself is untouched: a version that took a list was about 7% slower on one end (median 0.439 against 0.409 ms on a 150×150 lattice), so the two are kept apart.
  - **On screen:** orange rings at the exits (`ViewerManager.setExitDoors`), a legend entry, and an "Exit: another exit" row on the reroute card.
  - **Cache:** `IMPORT_SETTINGS` is 3, so cached models are read again for the new door field.
  - **Tests:** 259 unit tests (new: `astarToAny`, rerouting to another exit, the exits rule, the door's external flag), and three new browser-suite checks on Vienna (two exits out, two rings, the nearest from the top floor is the fire exit).
- **The rerouting GIF** (`docs/images/office-reroute.gif`, 1.3 MB): the Office example, a person upstairs at the east end, a Flamescapes class B fire at the foot of the east stair (8.6 MW peak), **Impassable above** 60 °C. The route takes the east stair until step 20 (100 s), then turns to the west stair and exit. Captured from the dev app with a caption injected into the page; the scripts are in the session scratchpad (`reroute/capture-reroute.mjs`). The switch came at step 20 in all five runs; the peak varied from 8.6 to 9.0 MW, as the GPU sometimes restarts on a coarser grid. The README quotes no single run's MW (Open 11): its Vienna captions were changed to match after fbdaed5 had quoted 13.3, 3.0 and 10.9 MW.

## This round (2026-11-06, afternoon)

- **A video of the Vienna model** (the user's request): 3 min 22 s, captioned, 1280×720 H.264.
  - **Where:** in OneDrive under `CM-iTAD/topologic_webapp/videos/2026-11-06 Vienna fire escape/`, full quality (48 MB) and phone (21 MB). Not in the repository.
  - **What it shows:**
    - the model, the graph and the ranked ignition points;
    - an electrical fault (class C) in a first-floor living-kitchen, 300 s with smoke and embers. Doors open: 13.3 MW peak, about 5 000 embers, 13 spot fires, and the route replanned at step 55. Doors closed: 3.0 MW;
    - a gas fire (class B) at another flat's hob: 10.9 MW;
    - escapes from the top floor (53.8 m down the stairwell to the street door) and from the fire's floor (29.5 m);
    - the heat, smoke and tenability views.
  - **How it was made:** Playwright driving the dev app, then a canvas page recorded with MediaRecorder (MP4). The scripts are in the session scratchpad: `video/capture.mjs`, `compose.mjs`, `probe.mjs`. A temporary dev-only hook exposing the exits was removed afterwards; the tree is clean.
  - **What it taught about the fires:** an electrical or gas fire in the services rooms stays small (26 to 122 kW), as there is little to burn. A fire in a furnished living-kitchen grows to megawatts within 300 s.

- **The human check** (8852fe2): Cloudflare Turnstile in front of the Worker.
  - **On the page:** `lib/turnstile.js` loads Turnstile only when "Describe this model" is set to Cloudflare and the site has a site key. It shows one slim strip, the panel's width and in the app's theme, under the send button. It checks at once and gives a fresh token for each request.
  - **In the Worker:** it checks the token with `siteverify` when it has `TURNSTILE_SECRET`, and refuses with 403 otherwise.
  - **Tested:** with a stand-in for `siteverify`, and end to end on localhost with Cloudflare's test keys.
- **Smoke, gas and tenability measures** (3f9a4f2):
  - **Readout:** the solver reads soot (mg/m³), CO2 and O2 (%) at every node, on the CPU and the GPU.
    - **Agreement:** CPU and GPU correlate 0.90 to 0.93 on them.
    - **Speed:** unchanged. On the GPU, 3.04 against 3.06 ms a step, and the worker run takes 1.9 s both times. The CPU is within noise.
    - **Memory switch:** "Record soot, CO₂ and O₂ too". A run past 64 MB of frames keeps the basic three.
  - **Show on the graph:**
    - Heat;
    - Smoke: visibility, soot, extinction, obscuration or optical density per metre;
    - Gases: CO, CO2, O2;
    - Tenability.
  - **The tenability index** is the largest share of a limit reached, after ISO 13571 cl. 4.1. Its four parts:
    - gas temperature;
    - visibility, 0 in clear air;
    - the asphyxiant dose: CO / 35 000 a minute, times exp(CO2/5) above 2 % CO2, plus Purser's low-O2 dose as in FDS;
    - the convected-heat dose, T^3.4 / 5e7 a minute.

    Both doses are measured against FED 0.3.
  - **The planner** closes nodes on the same four. No closure changed on the five examples within 30 s.
  - **Exports:** the TopologicPy JSON carries the series, the index, its doses and the time each node reached 1. A new CSV gives one row per node per frame, gzipped.
  - **Sources and what was read directly:** in the research note and on the Methods page. The ISO 13571:2012 constants were read in the 2007 text; confirm them against a full 2012 copy before a publication relies on them.

## This round (2026-11-06, morning)

- **Field notes and hints** (d0d1749, built to the plan and the user's answers):
  - **Field notes:** 12 questions marked found from the app's own results; "Field notes N/12" in the header; the list in a dialog.
  - **Hints:** one at a time at the foot of the panel. They lead past the model to the graph, route, alternatives, code check, fire, Flamescapes, closed doors, forecast plan and agent. On by default, with a switch and "Show closed hints again".
  - **Guided runs:** each ends on its note. "The way out" reports this run's own walk.
  - **Files:** `src/core/fieldNotes.js` (data, tested), `src/state/useFieldNotes.js` (localStorage), `src/ui/FieldNotes.jsx`.
- **The doors switch** (d0d1749):
  - **The problem:** it was locked for a whole fire run and its playback, with no explanation.
  - **The fix:** it now always sets the next run's doors. While a Flamescapes fire plays with the other setting, it offers "Restart the fire with the doors ...". The fire model choices still lock, and now say so.

- **The TU Wien escape-route model is a site example** (34d036b), by the user's decision:
  - **Name:** "Vienna apartment building (TU Wien)". The user called it the office building, but its record describes a residential building; the mixed-use custom model, which has offices, stays local.
  - **Licence:** CC BY 4.0, credited in `public/examples/index.json`, on the welcome screen and in the README.
  - **Unchanged file:** SHA-256 759c5e5f… as in the record, kept byte for byte by a `.gitattributes` rule (`-text`).
  - **Opening:** the build converts its 13.5 MB to 862 KB of fragments, so it opens in about a second with no IFC reader.
  - **In the app:**
    - graph: 6,795 nodes, 95% in one part;
    - exit doors read from the file: 35 of 81;
    - the code check passes on every storey.
  - **Browser suite:** section 8 checks it opens and reads its exits, and that its graph's largest part holds at least 90%.
- **The blank band after clicking a switch** (17cf2ff, reported by the user):
  - **Cause:** a switch's hidden checkbox was positioned against the page, not its switch. For switches low in a panel (Flamescapes, the code check's sprinklers) it sat below the window. Focusing it on a click scrolled the whole page up by as much as 340 px, though `body` has `overflow: hidden`.
  - **Fix:**
    - the checkbox is inside its switch;
    - `#root` is `position: relative; overflow: clip`, so the page has nothing to scroll.
  - **Browser suite:** flips the sprinklers switch and checks the page has not moved.
  - **The same band** showed in the 2026-11-05 screenshots of the hotel fire run and was wrongly blamed on the test script.

## This round (2026-11-05, late evening)

- **"Describe this model"** (da7064f, live):
  - **What it does:** describes the model from the app's own computed facts (`src/core/describeFacts.js`).
  - **Where it runs:**
    - on the device with Chrome's Prompt API (Chrome 148 or later, desktop);
    - or through the Cloudflare Worker in `cloudflare/describe-worker` (Workers AI, `@cf/openai/gpt-oss-20b`, free allowance, no retention by Cloudflare's terms). The Worker is offered only when the site is built with `DESCRIBE_URL`.
  - **Logo:** the engine's logo is off.
- **Code check** (d9afc1c, live; `src/core/codeCheck.js`; Route step, "Code check"):
  - **What it checks, per storey:**
    - the longest walk to an exit or stair;
    - exits against the occupant load;
    - exit separation;
    - exit door widths;
    - high-rise.
  - **Values:** IBC 2021, occupancy group B (offices), each labelled to be confirmed against SBC 201. The SBC's own text is not open. Sources are in `wiki/research/tall-office-sbc-egress-2026-11-05.md`.
  - **Display:** a toggle colours the graph by the distance to spare.
  - **On the TU Wien test models** (CC BY 4.0, local only): the real-world model passes; the custom model fails on exits and their separation.
- **The Worker's room suggestions** (96170e4): `POST /rooms`.
  - **Input:** one storey's draft summary.
  - **Output:** suggested merges, and names with classes.
  - **Checks:** only suggestions that name known rooms survive. Merged rooms must touch, each room gets at most one merge and one name, and classes come from the LOD1 set.
  - **Cost:** about 100 to 200 neurons a storey.
- **Rooms (LOD1) models in the app** (eea6d1d, live):
  - **The core** (`src/core/lod1.js`, written by an agent, checked here) reads:
    - a TopologicPy LOD1 export with its circulation and adjacency graphs;
    - a GeoJSON plan, whose schema is in `wiki/concepts/lod1-models.md`;
    - IfcSpace volumes.

    It converts them to the egress geometry an IFC gives, and also writes the GeoJSON plan.
  - **The app** (Model step, "Rooms models (LOD1)"; or drop the JSON files together on the viewport):
    - opens the model with no IFC;
    - draws it as rooms tinted by class, with walls, slabs, stairs, and doors (exits green) (`ViewerManager.loadLod1`);
    - lists rooms, storeys and warnings;
    - exports the GeoJSON plan.

    The graph, routes, code check, both fire models and the TopologicPy export run on it unchanged. Identification and the TopologicPy cross-check need an IFC and say so.
  - **The lab's hotel model** (private, local only; not in the repository): 586 rooms on 15 storeys.
    - In the dev app it opened in 0.5 s, with 1 warning (one opening whose rooms do not touch).
    - Graph: 47,305 nodes in 0.6 s, 100% reachable, every storey joined.
    - A Flamescapes fire ran 180 s in 58.5 s on the GPU, with no console errors.
    - Headless (`tools/lod1-check.mjs`, rerun here): the route from the top-floor guest room is 143.1 m, and the GeoJSON round trip is identical.
  - **The new block** (also private): 54 rooms on 4 storeys, all joined.
  - **IfcSpace fast path:** within about 6% of the full IFC's route on four public models, with 3 to 5 times faster extraction.
  - **Rooms without IfcSpace:** traced from the graph's regions, they matched too few rooms to ship (12 of 21 on the Duplex).
  - **The browser suite** gains step 9: a two-storey GeoJSON plan opens, lists its rooms, exports, and builds a graph joined through its stair.
    - **A false failure, fixed:** the step's first version failed because its wait function used a closure variable. The function runs inside the page, where the variable does not exist, so the wait failed at once. The step now passes the value in.
- **Drawings to LOD1, the core** (4cfcefb; `src/core/drawingToLod1.js`, written by an agent, checked here):
  - **Reader:** its own DXF reader, with no dependency. It explodes nested blocks and handles mirrored entities, code pages, and Rhino's arcs exported as polylines.
  - **Pipeline:** it suggests layers, bridges door openings, builds the planar arrangement, cuts rooms by their names and grows them to the walls' centre lines. `toLod1Model` and `draftSummary` (for the Worker) are included.
  - **Results against the lab's checked rooms**, rerun here: rooms found at IoU 0.8 or more, then matched at 0.5 or more.

    | building | rooms | lab's drafting | with the lab's layer settings | layers suggested by the core |
    |---|---|---|---|---|
    | residence (new block) | 54 | 30 found, 43 matched | 42 found, 49 matched, 6 extra | 38 found, 45 matched |
    | hotel, 7 sheets | 194 | 20 found, 65 matched | 57 found, 135 matched, 147 extra | 53 found, 135 matched, 193 extra |

  - **Speed:** each sheet is read in 0.1 to 0.4 s and a storey is drafted in 12 to 280 ms.
  - **Failure modes:**
    - partitions the drawing omits;
    - spaces bounded only by columns;
    - site walls on wall layers;
    - no entrance doors on the hotel's sheets;
    - hatch-only walls.

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
    - **The calibration plan** (2026-11-08, the user asked for it): `wiki/research/flamescapes-calibration-plan-2026-11-08.md`. Stage 0 is this grid study (hypotheses H1 to H4); then transport with prescribed heat release, one item burning alone, and spread in rooms, with the FDS Validation Guide's statistics. Nothing run yet; five decisions for the user in its section 11.
    - **Speed-sized grid noise:** sizing the GPU grid by speed makes the grid device-dependent; the stored speed keeps it fixed per browser. Fresh browsers picked 110k to 200k cells in three sessions, because the speed measurement is noisy while other jobs run.
12. **Ventilation is a bound, not a model.**
    - **Missing inputs:** system membership (IfcRelAssignsToGroup with IfcDistributionSystem) and supply, return and exhaust roles are not read; duct geometry is unused; 100 L/s a terminal is an assumption.
    - **Next:** read the systems where the IFC has them (DS3 has 4), make exhaust terminals a sink, and source the flows.
13. **Doors in the graph fire models,** SBC egress checks, fire hose coverage and firefighter access: see `wiki/research/sbc-firefighting-future-notes-2026-11-05.md`. Suppression: `fire-suppression-note-2026-11-05.md`.
14. **"Describe this model":** built and live (da7064f). The Worker is not deployed until the user sets it up with their own Cloudflare account.
15. **IFCNet:** retraining paused by the user (late evening).
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
19. **Resolved, test only:** step 9's first version failed on a broken wait in the test itself (This round). The app opened the plan every time.
20. **The code check covers offices only** (IBC group B), so a hotel (group R-1) or a mixed building needs its own values.
    - **On the hotel:** it reports exits as close as 2.3 to 5.8 m apart on several storeys. The cause is not yet established, either the stairs' real layout or how stair entrances are grouped. Check before quoting it.
    - **Not checked:** the common path of travel, dead ends, stair widths and capacity, enclosure.
21. **The IfcSpace fast path is not in the app.** `lod1FromIfcSpaces` works headless.
    - **What it would give:** for very large IFCs it could skip fragments conversion: read spaces, doors and stairs in the import worker and draw prisms, as in `tools/lod1-check.mjs`.
    - **Frame:** do not mix its frame with a fragments model of the same file.
22. **Drawings to LOD1:** see Resume here, item 1. Even the improved core finds only 57 of the hotel's 194 rooms at IoU 0.8, with 147 extra rooms, so the review screen is not optional.
