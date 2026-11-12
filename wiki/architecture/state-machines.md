# State machines

The app's explicit and implicit state machines: their states, transitions and invariants, the invalid states or transitions the code can reach (with the event sequence that reaches them), and the fixes. Kept current as the code changes: a fix updates its machine's table and its issue's status here.

Paths are relative to `topologicpy-web-frontend/src/` unless they start with `tools/`. Line numbers in the evidence are at commit 979d759, before the fixes; functions are named so they can be found after.

## How this was done

One area at a time, most critical first. For each area a read-only pass maps the machines and lists findings; every finding is then traced again by hand before it is called confirmed. Confirmed bugs are grouped into issues (SM-n below), fixed on `next/browser-core` with a regression test that fails on the code before the fix and passes after it, and recorded in [../issues/known-bugs-fixed.md](../issues/known-bugs-fixed.md); open ones go to [../issues/open-risks.md](../issues/open-risks.md).

A finding is **confirmed** when the full event sequence is traced line by line in the shipped code, **suspected** when a step is unproven (what would settle it is stated), and **refuted** when tracing shows the state cannot be reached.

| Area | Scope | Status |
|---|---|---|
| 1. Fire runs and compute workers | `lib/workerJob.js`, `workers/*` (not the IFC import), `core/flamescapesRun.js`, GPU init and device loss, `lib/recorder.js`, the run, plan, people and recording state in `App.jsx` | mapped; 11 confirmed, 1 suspected; fixed (SM-1 to SM-3) |
| 2. Network, backend and storage | `lib/api.js`, `lib/turnstile.js`, `lib/describe.js`, `core/runCache.js`, `lib/fragmentCache.js`, `state/useFieldNotes.js`, the backend's `app/store.py` and routers, `cloudflare/` | mapped; 13 confirmed, 5 suspected; 11 fixed (SM-4 to SM-6), 2 confirmed low ones open |
| 3. Model, graph and workflow | `state/useStudio.js`, `ui/Stepper.jsx`, model loading and the IFC import worker, `core/federation.js`, graph and route invalidation, `viewer/ViewerManager.js`, `viewer/TopologyViewer.jsx` | mapped; 15 confirmed (one by tracing That Open's code), 1 suspected; 11 fixed (SM-7), 4 confirmed lower ones open |

## Issues

Most severe first. Each groups findings with one cause.

### SM-1. A fire run outlives Stop, replacement or a restart

Findings A1, A3, A4, A8. A Flamescapes run is "current" from the moment it is made, but its worker job starts only after the kept runs are read, and its repeat runs keep the worker busy after it is done. Every place that ends a run asked only "is it done?" and cancelled the worker, so a run waiting for its lookup, or done but repeating, was not ended.

| Finding | Impact | Fix |
|---|---|---|
| A1. Stopped or replaced while its kept run is read, a run starts computing anyway. After Stop the card comes back and nothing can end it; replaced, it computes in the same worker as the newer run and its peak, first detection and first sprinkler can land on the newer run's card. | High | `runCombustion`: the lookup's continuation goes on only if the run is still current, else it ends as cancelled; `onProgress` writes nothing for a run that is not current. |
| A3. Start fire while a plan's Flamescapes run computes threw that run away and computed the same inputs again from zero. | Medium | `startFire` drops the plan with `keepRun`; `runCombustion` then reuses the run, or ends it if the inputs differ. |
| A4. Repeat runs survived Stop, a new graph and a new model, holding the GPU. | Medium | `run.repeating` while they run; Stop and a new graph cancel the worker when the run is not done or is repeating. |
| A8. After a GPU-to-CPU restart the tenability index kept the abandoned frames' doses. | Low | `core/tenability.js runIndex` starts the index again when the run's first frame is a different one. |

Regression tests: `core/tenability.test.mjs` (a restarted run's index equals a fresh one; fails on the old reset rule), `core/workerJob.test.mjs` (a cancelled job's late messages reach nobody); browser suite 5e: "a fire stopped while its kept run is looked up starts no solver" (the kept runs are held busy in IndexedDB), "Start fire while a plan's run computes plays that run" (solver workers counted), "Stop ends the repeat runs too" (only where the browser offers WebGPU; headless Chrome on the development PC uses its Intel GPU). Status: fixed.

### SM-2. One forecast worker shared by the plan, the two-zone fire and the report

Findings A2, A7. The plan, the two-zone fire's frames and the report's "Add fires" batch were jobs in one worker, and `cancel()` ends every job in a worker. The two-zone fire also had no "starting" state while its frames computed.

| Finding | Impact | Fix |
|---|---|---|
| A2 (i). The report's Stop ended a plan running beside it; the plan's catch returned on a cancel without resetting, so Plan spun, disabled, until a point moved. | High | Each caller has its own lane (`forecastJob(payload, transfers, lane)`: plan, zone, report), so a cancel ends only its own; a plan that is still current ends on any error, a cancel included. |
| A2 (ii, iii). A second Plan click lost a starting two-zone fire with no message; a plan or a point move ended a running report batch with fewer fires than it said. | Medium | The lanes. |
| A7. A two-zone fire had no "starting" state: Start and Plan stayed enabled and Stop disabled while its frames computed, so a double click started it twice and a Plan landed mid-fire. | Low to medium | `fire.starting` while the frames compute (Start, Plan and the fire settings wait; Stop ends it); a start goes on only if no stop or start came after it (`fireStartRef`). |

Regression tests: `core/workerJob.test.mjs` (ending one lane leaves another's job running); browser suite 5c "stopping the report's fires leaves a plan beside them to finish", 5d "a double click on Start fire starts the two-zone fire once" (two-zone jobs counted). Status: fixed.

### SM-3. Async actions with no "starting" state or no check that they are still wanted

Findings A5, A6, A10, A11, A12. Each awaits something (a module, a timer, a worker) and then acts as if nothing changed meanwhile.

| Finding | Impact | Fix |
|---|---|---|
| A5. People leaving: a key flipping A to B to A within the walk's 30 ms pause (a graph fire started again with no step delay) left people switched on with nobody drawn and no message. | Medium | The finished walk is kept with its key and shown again when its key comes back. |
| A6. A double click on Record GIF while the recorder's module loaded made two recorders; one recorded on, out of reach of Stop. | Medium | A "starting" sentinel in `recordRef` while the module loads; clicks then do nothing. |
| A10. Identify by shape: a module that failed to load left the button spinning for good. | Low | The import moved inside the `try`. |
| A11. Moving the start, exit or fire while the agent trained drew its answer for the old points. | Low | A move drops the training and its answer, as it drops a plan (the fire only when the agent uses it). |
| A12. Guided run: Stop in the fire stage counted as the fire ending, and the plan stage computed the stopped run again. | Low | `core/guided.js fireStageNext`: a stopped fire ends the guided run. |

Regression tests: `core/guided.test.mjs` (the stopped fire stage); browser suite 4b' "a double click on Record GIF makes one recording" (the recorder's module held back 0.8 s, recorders counted), 5b "moving the fire while the agent trains drops its answer", 5d' "people leaving with the fire are drawn after the fire is started again", Model step "identifying by shape ends cleanly when its module cannot load" (runs only on a model with elements left unidentified by the rules; the Duplex has none, so it is skipped there). Status: fixed.

### Verification of the area 1 fixes (2026-11-10)

Each regression check was run on the build before the fixes (979d759) and after them, both on the Duplex model in headless Chrome, which used the development PC's Intel GPU for WebGPU. After the fixes the whole suite passed on a desktop (129 steps) and an emulated phone (128 steps), with no console errors.

| Check | Before the fixes | After |
|---|---|---|
| A double click on Record GIF makes one recording | fail: 2 recorders | pass: 1 |
| Moving the fire while the agent trains drops its answer | fail: the old points' answer shown | pass |
| Stopping the report's fires leaves a plan beside them to finish | fail: Plan stuck busy (and the next step's Plan then could not be clicked) | pass |
| A double click on Start fire starts the two-zone fire once | fail: 2 starts | pass: 1 |
| People leaving with the fire are drawn after the fire is started again | fail: nobody drawn | pass |
| Stop ends the repeat runs too | fail: they ran on | pass |
| Start fire while a plan's run computes plays that run | fail: 2 solver runs | pass: 1 |
| A fire stopped while its kept run is looked up starts no solver | fail: 1 solver run after Stop | pass: 0 |
| A restarted run's tenability index (unit test) | fail on the old reset rule | pass |
| Identifying by shape ends cleanly when its module cannot load | not run: the Duplex leaves no element to identify by shape | not run |

### SM-4. The graph's copy on the server

Findings B1, B2, B3, B5. The server's id was written into the graph object when the upload answered, the upload happened only if the backend was "online" at the moment of the build, and a copy the server lost was never sent again. Since Next, routes, fires and plans run in the browser, so the free-plan server is usually asleep, and its store empty, by the time a TopologicPy feature is asked for.

| Finding | Impact | Fix |
|---|---|---|
| B1. The upload's answer replaced the graph with a new object for the same graph; everything kept "with the graph it was made on" took it for a new graph: a Flamescapes fire or plan running then was cancelled with no message, and the code check, the performance report and the sprinkler, systems and extinguisher plans vanished or were made again. Reached when the upload is slow, as it is after the server's 15-minute sleep (about 33 s), or when a guided run reaches its fire first. | High | The id is kept beside the graph (`serverGraphRef` in `App.jsx`, by the graph's `nodes`), so the graph object changes only on a build or a model load. |
| B2. A graph built while the backend was checking, waking or offline, or whose upload failed, was never uploaded; the TopologicPy features then refused with "not connected" under a "backend up" badge. | Medium | The graph is uploaded when a TopologicPy feature first asks for it (`serverGraphId`, one upload in flight per graph), not at the build; while the backend is waking the message says so. |
| B3. A copy the server lost (its 15-minute sleep, the 1 h limit, eight graphs shared by every user) was never sent again: every TopologicPy feature failed until a rebuild, which also dropped the route, plan and fire. | Medium | `onServerGraph`: on the server's "no graph" refusal the graph is sent again once and the request asked again; the copy of a graph replaced on screen is deleted. |
| B5. A TopologicPy engine saved in an earlier visit was used while the server's engines were unknown (asleep, unreachable, or a site with no backend): Find route refused, and a guided run stopped at its route stage. | Medium | `core/session.js effectiveEngine`: with no engine list the browser's engine runs; the saved choice applies again when the server lists it. |

Regression tests: `core/session.test.mjs` (the engine rule); `tools/server-race-test.mjs`, which stands between the page and the real local backend and holds back or refuses chosen requests: "the code check stays while the graph's upload is still out", "a graph built while the backend woke is sent when TopologicPy routes it", "a graph the server lost is sent again and the route asked again", "with no backend, a saved TopologicPy engine gives way to the browser's". Status: fixed.

### SM-5. Answers written for inputs that changed while they were awaited

Findings B4, B16, B6, B7. The server and describer answers were written whatever had changed meanwhile, and two waits could not be ended.

| Finding | Impact | Fix |
|---|---|---|
| B4. A TopologicPy route, alternatives or engine comparison for the old start, exit or graph was drawn after the points moved or the graph was rebuilt; a Navigation cross-check of the old model was drawn over a newly opened one. | Medium (the TopologicPy engine is a choice; the browser engine answers at once) | Each answer is written only if the graph and points (route answers) or the graph and model (the cross-check) are still the ones it was asked for (`askedRef`, `sameRoute`, `sameModel`). |
| B16. The engine comparison stayed after the points moved or the graph was rebuilt, and Compare had no busy state, so a double click sent two requests. | Low | The comparison is cleared wherever the route is; Compare is busy while a route request is out. |
| B6. Stop while the describer waited for the human check did nothing: it stayed busy for up to two minutes. | Low to medium | `humanToken({ signal })`: Stop ends the wait. |
| B7. A model's description stayed, or landed if still being written, under another model opened after it. | Low to medium | Opening another model ends a description still being written and clears the text. |

Regression tests: `tools/server-race-test.mjs`: "a route that comes after the start moved is not drawn", "the engines' comparison goes when the start moves"; `tools/describer-test.mjs --check stop` (B6), run on a build made with `VITE_DESCRIBE_URL` and Cloudflare's always-interactive Turnstile test key. B7's check (`--check model`, the always-pass key) did not reach its assertion on either build (it timed out waiting for the describer to be ready) and was set aside; B7 is checked by reading only. Status: fixed.

### SM-6. The backend's state and the site's own updates

Findings B10, B9, B11.

| Finding | Impact | Fix |
|---|---|---|
| B11. `/health` and `/api/graphs` listed every stored graph's id, and an id is the only key to a graph: any visitor could download another's walkable graph or delete it. | Medium | Both routes report counts only (`store.summary()`); the full listing stays for the server's own use. Live on Render since 2026-11-10 (deployed by hand; `/health` checked). |
| B9. A tab open across a deploy lost every part of the site it had not loaded yet, with a raw "Failed to fetch dynamically imported module" and no word about reloading. | Medium to low | `core/session.js isStaleSite`: the message says the site was updated and to reload. |
| B10. A request that timed out or was cancelled marked the backend offline. | Low | `lib/api.js`: kinds "timeout" and "aborted" apart from "network"; only "network" flips the status. |

Regression tests: `core/session.test.mjs` (the request kinds with a stubbed `fetch`, and the stale-site messages of Chrome, Firefox and Safari); backend `tests/test_api.py::test_graph_ids_are_not_listed`; `tools/server-race-test.mjs` "a part of the site gone after a deploy says to reload". Status: fixed.

### Verification of the area 2 fixes (2026-11-10)

Run on the build before the area 2 fixes (016a002) and after them, on the Duplex, against the real local backend.

| Check | Before | After |
|---|---|---|
| The code check stays while the graph's upload is still out | fail: gone when the upload answered | pass |
| A graph built while the backend woke is sent when TopologicPy routes it | fail: no route, 0 uploads | pass: 1 upload |
| A graph the server lost is sent again and the route asked again | fail: no route, 1 route request | pass: 2 uploads, 2 requests |
| With no backend, a saved TopologicPy engine gives way to the browser's | fail: no route | pass |
| A route that comes after the start moved is not drawn | fail: drawn | pass |
| The engines' comparison goes when the start moves | fail: it stayed | pass |
| A part of the site gone after a deploy says to reload | fail: the raw message | pass |
| Graph ids are not listed (pytest) | fail | pass |
| Request kinds, engine rule, stale-site messages (unit) | fail: `lib/api.js` could not be imported by node, and the old rules | pass |
| Stop ends a wait for the human check (describer build) | fail: still waiting | pass |
| A description goes when another model is opened (describer build) | not reached: the check timed out on both builds | not reached |

After the fixes the whole suite passed on a desktop (129 steps) and an emulated phone (128 steps), with no console errors, and `tools/server-race-test.mjs` passed 7 of 7. Backend: 131 tests pass.

### SM-7. The model and the graph changing under work in progress

Findings C1 to C9, C14, C15. The model and the graph live both in React state and in the viewer's scene, kept in step by hand. Only the model load checked that it was still the latest; the graph build, a fire still starting, the second models, the markers and identification did not, and a rebuild left the last fire behind.

| Finding | Impact | Fix |
|---|---|---|
| C1. A graph build still running when another model was opened set the old model's graph on the new one, said "Graph ready" and moved to Route; routes, fires and the code check then ran on the old graph under the new model. Its end also cleared the new load's progress card. | High | `buildGraph` goes on only for the load it started on (`loadSeqRef`), keeps its extracted geometry only then, and leaves the busy flags to a newer build or load; opening a model cancels the build's worker. |
| C7. The `b` key started a build while one was running, and once per key repeat, each with copies of the geometry. | Medium | One build at a time per model (`buildingRef`); key repeats ignored. |
| C2. A two-zone fire still starting survived a new model or a rebuild, then played on the old graph. | Medium | `endFireStart()`, used by Stop, the model opens, a rebuild and the viewer switch. |
| C6, C14. A rebuild kept the last fire's step, model and burning nodes, so Export for TopologicPy wrote them onto the new graph's vertices; and the agent's answer for the old graph. | Medium, low | The rebuild resets the fire, its burning nodes and temperatures, and the agent. |
| C3. Viewer mode Topology JSON and back gave a new, empty viewer while the model, graph and points stayed; Rebuild graph then said "Load an IFC model first". | Medium | `switchMode`: leaving the model's viewer closes the model with it. |
| C4. A rooms model opened while an IFC loaded left the progress card for good. | Medium | `openLod1` ends the card with the load it overtakes. |
| C5. A second model still being added when the main model was replaced stayed in the scene until a reload. | Medium | `loadLinkedIfc` removes and disposes it when the owner changed after it was drawn. |
| C8. The start, exit and fire markers stayed drawn after another model was opened. | Medium to low | The markers follow `points` in an effect. |
| C9. An example clicked before the viewer was ready left "Downloading" spinning. | Medium to low | The not-ready path ends the card. |
| C15. A late identification answer for an older model overwrote the newer model's. | Low | Written only if the same model is still open. |

Regression tests: `tools/model-race-test.mjs` (new): "the build key pressed during a build starts no second one" (builds sent to the sampler counted), "a build still running when another model opens does not land on it" (the sampler worker's script held back), "a rebuilt graph drops the last fire and its export", "the start marker goes when another model is opened", "back from the topology viewer, no graph is offered for a model no longer drawn". C2, C4, C5, C9 and C15 are checked by reading only: each needs a timing the scripts do not yet hold (a two-zone start, a rooms model during a load, a second IFC, the viewer still starting, two identifications). Status: fixed.

### Verification of the area 3 fixes (2026-11-10)

| Check | Before (4f7d57b) | After |
|---|---|---|
| The build key pressed during a build starts no second one | fail: 3 builds sent | pass: 1 |
| A build still running when another model opens does not land on it | fail: the old model's graph set on the new one | pass |
| A rebuilt graph drops the last fire and its export | fail: the export still offered | pass |
| The start marker goes when another model is opened | fail: still drawn | pass |
| Back from the topology viewer, no graph is offered for a model no longer drawn | fail: the old graph offered | pass |

### Open

- A9 (suspected, low): when the GPU solver falls back to the CPU on its own (`flamescapesGpu.js runSceneGpu`: device already lost, scene unsupported, or a buffer setup error), the run keeps the GPU label and the GPU-sized grid, and with the "speed" grid a CPU-measured rate could be stored as that GPU's. To settle: force a buffer setup error in a WebGPU browser and read the card's backend line and `info.gpuRate`. Suggested fix: let `runSceneGpu` throw so the worker's labelled, re-sized CPU restart runs. In [../issues/open-risks.md](../issues/open-risks.md).
- Not reachable today, noted: a throw inside `repeat()` after a successful run would reject `run.promise` with `failed` still false, and that run would be reused for its key. No call in `repeat()` throws in practice.
- B8 (confirmed, low, not fixed): a theme change while Turnstile's script loads leaves no human check widget (two mounts of the same element await the script; the first's late cleanup removes the widget the second relied on), so the next cloud description fails with "choose Cloudflare again". Fix when touched: a mount counter in `lib/turnstile.js`, and the cleanup removing only its own widget. Needs a site key to test, which local builds lack.
- B15 (confirmed, low, not fixed): two tabs overwrite each other's field notes and settings in `localStorage` (each writes its whole object; no `storage` listener). Fix when touched: merge on write and refresh on the `storage` event.
- Suspected, low: B12 (near the storage quota a failed write also rolls back its eviction, so nothing new is kept); B13 (IndexedDB opens have no timeout or `onblocked`/`onversionchange`; latent until a database version bump with an old tab open); B14 (a cached fragments buffer that fails to load is not dropped); B17 (TopologicPy TGraph builds are not serialised, though Navigation builds are locked as "not known to be thread-safe"); B18 (the Worker's `/rooms` wants a Turnstile action the page never issues, latent; a siteverify outage reads as a failed human check). What would settle each is in the area 2 findings below.
- The server still answers a request with no graph id on the latest graph of its kind, whoever made it (`store.get(None, mode)`). Next always sends an id; a client that does not would reach another visitor's graph. Not changed: other clients of the API may rely on it.
- C10 (confirmed, medium to low, not fixed): a second model added before the first model's semantics arrive is placed and checked with none (`addLinkedModel` reads `studio.semantics` at the click), so a pair aligned by map conversions is placed by placements, and nothing places it again. Fix: offer "Add a model" only once the semantics are in, or read them when placing.
- C11 (confirmed by tracing That Open's code, not run; low to medium): after a WebGL context loss and restore, That Open makes a new renderer without local clipping, so the storey cut stops clipping (and a phone's pixel ratio is lost). Fix: a `webglcontextrestored` listener in `ViewerManager.init` setting `localClippingEnabled` and the pixel ratio again.
- C12 (suspected, low to medium): a code check run before the IFC's semantics arrive keeps the fallback exits after they arrive. To settle: hold the semantics pass and run the check.
- C13 (confirmed, low): the overhead meshes (beams, ducts) are keyed by file name, so a revised file of the same name briefly uses the old model's. Fix: key by the model.
- C16 (confirmed, low): one abort slot for imports in the viewer, so a second model's load displaces the main model's semantics job, which can then no longer be stopped.

## 1. Fire runs and compute workers

### Machines

#### M1. Worker job client (`lib/workerJob.js`)

One client per worker; the forecast worker has one per lane (plan, zone, report).

| State | Representation |
|---|---|
| No worker | `worker === null`, no jobs |
| Idle | `worker` set, no jobs |
| Busy(n) | `worker` set, n jobs pending |

| Event | From | To |
|---|---|---|
| `run()` | any | Busy(n+1); a worker is made when there is none |
| reply `{id, progress}` for a pending job | Busy | Busy; `onProgress` runs synchronously |
| reply `{id, ok}` | Busy(n) | Busy(n-1) or Idle |
| reply for an id not pending (after a cancel) | any | unchanged; dropped |
| `error` or `messageerror` from the current worker | Idle or Busy | No worker; every job rejected with the client's message |
| `cancel()` | Idle or Busy | No worker; every job rejected "cancelled" |
| `postMessage` throws | any | that job rejected; worker kept |

Invariants: every job settles once; after `cancel()` no handler of a job that was pending runs again (the job table is cleared synchronously and every message is looked up in it); a crashed worker is never reused; **`cancel()` ends every job of its worker**, so callers that must not end each other need separate clients (SM-2).

#### M2. The fire solver's job (`workers/flamescapes.worker.js`)

Choosing a backend, then the scene posted, then running (GPU, measuring its rate), possibly restarted coarser (speed grid) or on the CPU after a GPU throw (a second scene with `restarted`), then answered. Repeat jobs post one summary per member, or answer `{skipped}` without a GPU. Jobs in one worker interleave at every `await`; a CPU run is synchronous. Invariant: every frame after a `restarted` scene is the new run's (message order). The invariant "the backend label names the backend that ran" is suspect (A9).

#### M3. GPU device (`core/flamescapesGpu.js`)

Per job: no device, device ok, device lost or in error. `readFrame` throws on a lost or erroring device, so the worker's CPU restart runs; each job asks for a new adapter and device.

#### M4. Fire run (`App.jsx runCombustion`, `combustionRef`)

| State | Representation |
|---|---|
| None | `combustionRef.current === null` |
| Looking up | run current; `promise` waiting on `readRun` (IndexedDB); no worker job |
| Computing | worker job pending; `done` false |
| Done | `done`, not `failed`; `result` set (computed or kept) |
| Repeating | Done, and `repeating`: a repeat job pending in the same worker |
| Failed or cancelled | `done` and `failed` |

| Event | From | To |
|---|---|---|
| `runCombustion()`, same key and graph, not failed | any | the same run |
| `runCombustion()`, otherwise | any | worker cancelled; a new run, Looking up |
| lookup ends, run no longer current | Looking up | Failed (cancelled); no work starts *(SM-1, A1)* |
| lookup hit | Looking up | Done (kept frames and card), then Repeating if asked |
| lookup miss | Looking up | Computing |
| scene or frame, run current | Computing | Computing (card, draw rate, frames; a restart drops the frames) |
| scene or frame, run not current | any | ignored *(SM-1, A1)* |
| job resolves / rejects | Computing | Done (kept when idle, repeats if asked) / Failed |
| Stop | Looking up or Computing | None; worker cancelled |
| Stop | Repeating | Done; worker cancelled, repeats "stopped" *(SM-1, A4)* |
| plan dropped, its run not playing and not done | Looking up or Computing | None; worker cancelled; kept when the fire starting now plays it *(SM-1, A3)* |
| graph changes | any | None; worker cancelled unless Done and not repeating *(SM-1, A4)* |

Invariants: only the current run writes the card and the draw rate; a run that is not current starts no work; at most one fire job runs, the current run's; Stop frees the CPU core or the GPU.

#### M5. Repeat runs (`App.jsx runCombustion` `repeat`)

Started after the run is Done, on the GPU, on the same model, while the run is current. `run.repeats.status`: running, then done or stopped; the card is written only while the run is current.

#### M6. Forecast plan (`App.jsx planForecast`, `planGenRef`, `planRunningRef`, `forecastPlan`)

Idle, waiting for the fire run (Flamescapes), planning (a job in the plan lane), then shown or dropped. `planGenRef` counts plans and the changes that make a plan stale; an answer for an older count is ignored. Invariant: `busy` ends when the plan ends, whatever ended it *(SM-2, A2)*.

#### M7. Fire start and playback (`App.jsx startFire`, `stopFire`; `state/useStudio.js stopFire`)

| State | Representation |
|---|---|
| Idle | `fire.running` and `fire.starting` false |
| Starting (two-zone) | `fire.starting`: its frames compute in the zone lane *(SM-2, A7)* |
| Playing | `fire.running`; a fire handle set |
| Ended / Stopped | `fire.running` false; `fireStoppedRef` tells which |

Start is enabled only in Idle (or after an end); Plan and the fire settings wait while Starting or Playing; Stop is enabled while Starting or Playing. A start continues after its frames only if no stop or newer start came (`fireStartRef`). Run again in the view (`App.jsx replay`, 2026-11-12) calls the same `startFire`; it is offered only in Ended or Stopped after a Flamescapes fire, with a Flamescapes model chosen, and not while a plan's run or repeat runs compute, a guided run is on screen or a pick waits. During a guided run the playback bar's Stop fire and the panel's Start and Stop are not shown: the run's own Stop (`stopGuided`) stops a fire playing or starting as Stop fire does, then ends the run and gives back the settings it changed (2026-11-12).

#### M8. People leaving (`App.jsx`: `crowdKey`, `crowdState`, `crowdDoneRef`, the walk effect)

| Event | From | To |
|---|---|---|
| key set, not the last walk worked out | any | Busy for the key, then after 30 ms the walk |
| key is the last walk worked out | any | that walk shown again *(SM-3, A5)* |
| walk ends | Busy | Shown or Error; kept as the last walk |
| key changes before the walk ends | Busy | the walk dropped (`live` false, timer cleared) |
| new graph | any | switched off |

Invariant: switched on, the panel shows exactly one of busy, an error, or the people for the current key.

#### M9. GIF recording (`App.jsx toggleRecording`, `lib/recorder.js`, `workers/gif.worker.js`)

Idle, starting (the module loading; `recordRef` holds a sentinel and clicks do nothing) *(SM-3, A6)*, recording, encoding, idle with the GIF downloaded. The button is disabled while encoding. Invariant: at most one recorder, and `recordRef` holds it.

#### M10. Shape identification (`App.jsx identifyByShape`)

Busy from the click, the module loaded and a classifier made inside the `try` *(SM-3, A10)*, chunks classified, idle; the classifier ended in `finally`.

### Findings

| Id | Finding | Status | Impact | Issue |
|---|---|---|---|---|
| A1 | A run stopped or replaced during its kept-run lookup starts computing anyway; its frames write the card | Confirmed | High | SM-1 |
| A2 | One forecast worker for the plan, the two-zone frames and the report; the report's Stop leaves Plan busy for good | Confirmed | High | SM-2 |
| A3 | Start fire during a plan's run recomputes that run from zero | Confirmed | Medium | SM-1 |
| A4 | Repeat runs survive Stop, a new graph and a new model | Confirmed | Medium | SM-1 |
| A5 | People leaving: a key flipping A, B, A within 30 ms leaves nobody drawn | Confirmed | Medium | SM-3 |
| A6 | Double click on Record GIF while its module loads starts two recorders | Confirmed | Medium | SM-3 |
| A7 | The two-zone fire has no starting state | Confirmed | Low to medium | SM-2 |
| A8 | The tenability index keeps an abandoned run's doses after a restart | Confirmed | Low | SM-1 |
| A9 | The GPU solver's own CPU fallbacks keep the GPU label and grid | Suspected | Low | open |
| A10 | Identify by shape stays busy when its module fails to load | Confirmed | Low | SM-3 |
| A11 | Training is not dropped when the start, exit or fire moves | Confirmed | Low | SM-3 |
| A12 | Guided run: Stop in the fire stage counts as the fire ending | Confirmed | Low | SM-3 |

Evidence, as traced (lines at 979d759):

- **A1.** `runCombustion` sets `combustionRef.current = run` (App.jsx:1795) but starts the job in `readRun(storeKey).then(...)` (:1778-1784) with no check that the run is still current. Every way of ending a run cancels the worker (:1601, :2187-2190, :359-363, :3344-3346), and during the lookup the run has no job in it. `readRun` opens a readwrite transaction (core/runCache.js:129), so it waits behind the previous run's `writeRun` of up to 96 MB, started when the browser is idle up to 5 s after a run ends (App.jsx:1786-1789). Sequence: start, Stop during the lookup, the lookup misses, `compute()` makes a new worker; `onProgress` writes the card and the GPU draw rate with no currency check (:1669-1712), and Stop is disabled. Replaced instead, both runs' jobs go to one worker and, on the GPU, interleave at every frame read; the card's running maximum, first detection and first sprinkler keep values from either run (:1683, :1696, :1710).
- **The open lead from the first reading** (a frame queued when the worker is terminated reaching `studio.setFire`) is **refuted**: `cancel()` clears the job table synchronously (lib/workerJob.js:28-29), every message is looked up in it when it arrives (:36-37), and `onProgress` runs inside that listener (:39).
- **A2.** `forecastJob` (App.jsx:293-299) was the only client for the plan (:3273), the two-zone frames (:2033) and the report's fires (:2817). The report's Stop (`stopPerfFires`, :2848-2851) cancelled it; the plan's catch returned on a cancelled error without resetting (:3334), leaving `planRunningRef` and `forecastPlan.busy` true; the Plan button is `busy` and so disabled (ui/Panels.jsx:2343, ui/primitives.jsx:33). A second Plan click (:3208) or `dropPlan` (:356) during the two-zone computation rejected the fire's frames as cancelled, and `startFire` returned silently (:2045-2048).
- **A3.** `startFire` called `dropPlan()` (:2021); `dropPlan` cancelled a run that is not done and not playing (:359-363), and `playingRunRef` is set only later in `startFire` (:2068). `runCombustion` (:2060) then made a run with the same key; a cancelled run is never kept (:1786), so it started from zero.
- **A4.** Stop, a new graph and `dropPlan` cancelled only `!done` runs (:2187, :3345, :360); repeat runs start after the run is done (:1768, :1791).
- **A5.** The walk effect returned early when the key matched the last finished one (:2444) without putting that walk back into `crowdState`; a walk for another key started and dropped in between left `crowdState` on that key with `busy` true (:2452, :2478-2481), and `crowdResult` needs `crowdState.key === crowdKey` (:2402). Reached with "Egress: with the fire", a graph fire and a step delay under about 30 ms: `startFire` sets step 0 (key without the fire), and step 1 comes 0 ms later (key with the fire).
- **A6.** `toggleRecording` set `recordRef` only after `await import("./lib/recorder.js")` (:2552-2558); the button is disabled only while encoding (ui/Viewport.jsx:509).
- **A7.** `startFire` awaited the two-zone frames (:2029-2050) before setting `running` (:2085); Start, Stop and Plan read only `fire.running` (ui/Panels.jsx:2870, :2875, :2343).
- **A8.** On a restart the frames are dropped (:1666), but the index was reset only when it was longer than the frames (:381); playback resumes at the frame it waited for, by then the new frames outnumber the old index, and the doses carried over.
- **A10.** The `import()` and `createGeometryIdentifier` ran before the `try` (:1920-1921).
- **A11.** `handlePick` (:963-979) dropped the plan and the route, not the training; `trainRl` drew its answer with no check (:2247-2255).
- **A12.** The fire stage advanced on `sawRunning && !fire.running && fire.step > 0` (:3113), which Stop also satisfies; the plan stage's `planForecast()` then found no run and started one (:3115, :3222).

### Checked and sound

- `lib/workerJob.js`: every job settles once; a crash rejects that worker's jobs and drops it; an error from an older worker is ignored; a throwing `create()` or `postMessage` rejects the job. Unmount cancels every client. There is no React StrictMode (main.jsx), so no development-only double effects.
- The forecast, sampler, RL and identify workers answer every job with its id, as `{ok: false}` on a throw; the identify client waits for `ready` before classifying.
- Fire run: same key and graph reuse the run, a failed run is not reused; a second Flamescapes Start cannot happen (`running` is set in the same click, before any await). A replaced or stopped run leaves the newer run's draw rate alone. Repeat runs start only for the current run on the same model. A kept run is written only after success. GPU loss mid-run restarts on the CPU with a re-sized grid and no stale GPU frame after the restart scene; no retry loop, no stuck state (apart from A8 and A9).
- Plan and fire share runs safely: the plan awaits the same `run.promise`, its generation check drops stale answers, and a cancelled run ends the plan. Plan is disabled while a fire plays.
- People leaving: the walk's `finish` is guarded by `live`, its timer cleared on cleanup, a failed import or walk ends in an error, a new graph switches people off. The walk runs on the main thread inside the timer (a cost, not a state bug).
- Recording: a second `stop()` is guarded; a GIF worker error resets to idle; unmount cancels; late ticks after idle are ignored; recording across a model load keeps the canvas.
- RL: Train is disabled while busy; the worker is cancelled on a model load and a graph build; the catch resets busy.
- Shape identification: a re-identify mid-run stops the old loop. A model change does not stop it, but its answers go to a state whose model no longer matches, so nothing wrong is shown (wasted compute only).

## 2. Network, backend and storage

Since Next, routes, fires, plans and RL run in the browser, so the backend sees a request only when a TopologicPy feature is used. On the free Render plan it sleeps after 15 idle minutes and restarts with an empty graph store (render.yaml), so it is usually asleep, and its store empty, when a TopologicPy feature is asked for. Every machine below lives with that.

### Machines

#### N1. Backend status (`state/useStudio.js`: `serverStatus`, `capabilities`, `refreshCapabilities`, `reportError`)

| State | Representation |
|---|---|
| Checking | the first probe pending |
| Waking | still checking after 2.5 s ("Waking backend…") |
| Online | a probe answered; `capabilities` set |
| Offline | a probe failed (`capabilities` null), or a request could not reach the server |

| Event | From | To |
|---|---|---|
| mount | - | Checking; one probe with a 90 s timeout |
| 2.5 s, probe pending | Checking | Waking |
| a probe answers | any | Online |
| a probe fails | any | Offline |
| a request fails as `network` (unreachable) | any | Offline *(a timeout or a cancel no longer does: SM-6, B10)* |
| Offline, tab visible | Offline | probes after 4, 8, 16, 32 then 60 s |
| the server sleeps or restarts | Online | Online (not probed again while Online) |

Invariants: Offline is left only by a probe that answers; Online means "answered once", not "has my graph" (N3). Derived: `effectiveEngine` (core/session.js) is the saved engine if the server lists it, the server's first otherwise, and the browser's when no list is known *(SM-4, B5)*.

#### N2. One request (`lib/api.js request`)

Waiting for headers (timer armed: 120 s by default, 8 or 90 s for probes, 300 s for the cross-check), reading the body, resolved, or rejected as an `ApiError` of kind `http` (the server's `detail`), `network` (unreachable), `timeout` or `aborted` *(SM-6, B10)*. Every request settles once; the timeout covers the wait for headers only.

#### N3. The graph's copy on the server (`App.jsx`: `serverGraphRef`, `serverGraphId`, `onServerGraph`)

| State | Representation |
|---|---|
| None | `serverGraphRef.nodes` is not the graph on screen's |
| Uploading | `pending` set for the graph on screen |
| Held | `id` set for the graph on screen |
| Lost | `id` set, but the server no longer has it (known only when a request is refused) |

| Event | From | To |
|---|---|---|
| graph built or model opened | any | None for the new graph (the graph object itself is not touched again) *(SM-4, B1)* |
| a TopologicPy feature asks, backend Online | None | Uploading; the previous graph's copy deleted *(SM-4, B2)* |
| a second feature asks meanwhile | Uploading | Uploading (the same upload awaited) |
| upload answers | Uploading | Held |
| upload fails | Uploading | None (the feature reports the error; the next ask uploads again) |
| the server sleeps, its 1 h limit passes, eight newer graphs, a DELETE | Held | Lost |
| a request refused "no graph" | Lost | Uploading, then the request asked once more *(SM-4, B3)* |
| a feature asks while the backend is not Online | any | refused: "not connected", or "still waking up" |

Invariants: a server request names only the graph on screen; the graph object stands for one graph for as long as it is on screen; a lost copy is sent again.

#### N4. Server graph store (`backend/app/store.py GraphStore`)

Per id: absent, live (`_items[gid]`, touched on every `get`), gone (TTL of an hour idle, the oldest beyond eight, `DELETE`, a process restart). Every access holds the lock; a request holding a graph keeps it after eviction; one uvicorn process, so one store. The store is shared by every visitor; the public routes now report counts only *(SM-6, B11)*. Still: `get(None, mode)` answers with the latest graph of a kind whoever made it (Open).

#### N5. Server answers in `App.jsx` (route, compare, alternatives, Navigation cross-check)

Idle, busy (`pathBusy` or `navBusy`; Compare now shows it too), shown (`path`, `comparison`, `alternatives` or `navCheck`). An answer is written only if the graph and the points (route answers) or the graph and the model (the cross-check) are still those it was asked for *(SM-5, B4)*. Moving a point, clearing them, rebuilding the graph or opening a model clears the route, the alternatives and the comparison *(SM-5, B16)*. The browser engines answer within the click, so only the server paths have a window.

#### N6. Describer (`App.jsx`: `describe`, `runDescribe`, `stopDescribe`; `lib/describe.js`)

| State | Representation |
|---|---|
| Ready | `busy` false |
| Waiting for the human check | `busy`; cloud; `humanToken` pending (up to 120 s) |
| Writing | `busy`; the cloud request or the on-device model running |
| Written / Error | `text` or `error` set |

| Event | From | To |
|---|---|---|
| Describe | Ready, Written, Error | Waiting (cloud) or Writing (on the device) |
| token, answer, failure | Waiting, Writing | Writing, Written, Error |
| Stop | Waiting or Writing | Ready *(SM-5, B6: Waiting used to stay)* |
| another model opened | any | Ready, text cleared *(SM-5, B7)* |
| the panel closed while Waiting | Waiting | Error "The human check was closed." |

The Describe button and the engine switch are disabled while busy, so two runs never overlap.

#### N7. Human check (`lib/turnstile.js`)

The script not loaded, loading or loaded; no widget, a widget with no token (requests wait), a widget with a token (taken once, then the widget resets). Mounting a new element replaces the widget; unmounting rejects the waiters. Invariant: while the cloud engine is chosen, one widget is in the panel. Broken by a theme change during the script's load (B8, open).

#### N8. The describer Worker (`cloudflare/describe-worker/src/index.js`)

Per request: origin (403 with no CORS header), size (413), JSON (400), the human check (no secret: pass; else siteverify with a 10 s timeout, checking success, action and hostname), facts (400), Workers AI (503 past the daily allowance, 502 for an empty answer), then `{text, model}`. A siteverify outage is answered as a failed human check (B18, open).

#### N9. Kept runs (`core/runCache.js`) and N10. Fragment cache (`lib/fragmentCache.js`)

One IndexedDB connection each, opened on first use and kept; a failed open is retried on the next call. Kept runs: read and touch in one readwrite transaction; write only after a run succeeded, over 96 MB not at all, LRU eviction and the put in one transaction; every failure means "compute as before". Fragments: buffer and record in one transaction, keyed by the IFC's hash and the converter's versions, LRU eviction after a successful write. Open: B12, B13, B14.

#### N11. Field notes (`state/useFieldNotes.js`) and settings (`state/useStudio.js`)

Read once at mount from `localStorage`, written whole on every change, errors swallowed (the session keeps its state). Settings drop the session-only keys on write and on read. Two tabs overwrite each other (B15, open).

#### N12. The server's Navigation cache (`backend/app/graphs/navigation_check.py`)

Parsed models and Navigation graphs by the IFC's hash and the spacing, LRU, every build under one lock, run in the threadpool after the upload is read. Sound.

### Findings

| Id | Finding | Status | Impact | Issue |
|---|---|---|---|---|
| B1 | The upload's answer made a new graph object; a fire or plan running then ended, and the code check, report and plans vanished | Confirmed | High | SM-4 |
| B4 | Server answers written whatever changed meanwhile | Confirmed | Medium (TopologicPy engine) | SM-5 |
| B3 | A lost server copy never sent again | Confirmed | Medium | SM-4 |
| B2 | No upload unless "online" at the build, and no retry | Confirmed | Medium | SM-4 |
| B5 | A saved TopologicPy engine refused routing while the server's engines were unknown | Confirmed | Medium | SM-4 |
| B11 | Graph ids listed publicly; any client could read or delete any graph | Confirmed | Medium | SM-6 |
| B9 | A tab open across a deploy fails with a raw message | Confirmed | Medium to low | SM-6 |
| B6 | Stop did not end a wait for the human check | Confirmed | Low to medium | SM-5 |
| B7 | A description stayed under another model | Confirmed | Low to medium | SM-5 |
| B10 | A timeout or a cancel marked the backend offline | Confirmed | Low | SM-6 |
| B16 | The comparison outlived its inputs; Compare had no busy state | Confirmed | Low | SM-5 |
| B8 | A theme change while Turnstile loads leaves no widget | Confirmed | Low | open |
| B15 | Two tabs overwrite each other's notes and settings | Confirmed | Low | open |
| B12 | Near the quota, eviction never happens | Suspected | Low | open |
| B13 | IndexedDB opens can wait for good | Suspected, latent | Low | open |
| B14 | A cached fragments buffer that fails to load is served again | Suspected | Low | open |
| B17 | TopologicPy TGraph builds are not serialised | Suspected | Low | open |
| B18 | The Worker's `/rooms` action; a siteverify outage read as a failed check | Suspected, latent | Low | open |

Evidence, as traced (lines at 016a002):

- **B1.** `buildGraph` uploaded only when `serverStatus === "online"` (App.jsx:879) and on the answer set the graph to `{ ...current, graphId, bounds }` (:884-887), a new object. The effect on `[graph]` (:3409-3418) cancels a run whose `run.graph !== graph`; the code check (`codeState.graph === graph`, :2343), the report (:2378), and the sprinkler, systems and extinguisher plans (:1318, :1432, :2651, :2712) compare the graph by object. The people-leaving code already compared `nodes` for this reason (:2422-2431).
- **B2.** With the status checking, waking or offline at the build, nothing uploaded later; `needsServerGraph` (:919-929) then refused because `graph.graphId` was null, under a "backend up" badge once the probe answered.
- **B3.** The store keeps a graph for an hour idle and at most eight, shared by every visitor (backend/app/config.py, store.py); the free plan's sleep empties it. A request then got 400 "No navigation graph available…" (routers/ifc.py `_require_graph`), `reportError` said to build the graph again (useStudio.js:366-380), and `graph.graphId` stayed, so every later request failed the same way.
- **B4.** `findPath` (:1036-1046), `comparePath` (:1066-1075), `findAlternatives` (:1102-1126) and `runNavigationCheck` (:1156-1164) wrote their answers with no check; the pick buttons, Rebuild graph and Open IFC stay enabled while they wait.
- **B5.** `effectiveEngine` returned the saved engine when `capabilities` was null (useStudio.js:489-493); `settings.engine` is saved across visits.
- **B6.** `humanToken()` took no signal (lib/turnstile.js:85-100); Stop aborted only the later fetch.
- **B7.** Nothing cleared `describe.text` when a model was opened; the answer was written with no check (App.jsx:3060).
- **B9.** No handling of a chunk that a deploy removed; `reportError` showed the browser's message as it was.
- **B10.** `request` mapped every `AbortError`, its own timeout and a caller's cancel alike, to kind `network` (api.js:50-53), and `reportError` set Offline for that kind (useStudio.js:362-364).
- **B11.** `GET /health` returned `store.stats()` with every id (main.py:69-77, store.py:235-253), as did `GET /api/graphs`; `GET` and `DELETE /api/ifc/graph/{id}` need only the id.
- **B16.** `setComparison(null)` was not called by `handlePick`, `clearPoints` or `buildGraph`; the Compare button had no `busy`.

### Checked and sound

- `request`: every call settles once; an HTTP error carries the server's `detail`; the timer is always cleared; an already-aborted signal aborts at once.
- The probes: 90 s for the first; the offline loop backs off to a minute, skips while hidden and probes at once on becoming visible.
- A server request always names the graph on screen; the server's "latest graph" fallback is never reached from Next.
- Server: the store takes its lock on every access and keeps its "latest" map consistent on eviction; the graph's lazy caches are idempotent; `import_graph` validates lengths, ranges and finiteness; CORS lists the Pages origin and the local ports. The fire stream and the server's RL and timeline routes are not used by Next.
- Topology inspector: its three server actions are disabled while busy.
- Describer: one run at a time; an abort during a fetch or on-device streaming ends busy without an error; a token is taken before any await, so two callers never share one; an expired token resets the check.
- Worker: other origins are refused before any work; the body size is checked before parsing; a token is checked for success, action and hostname; Workers AI failures are 503 with the reason.
- Kept runs and fragments: written only after success; every failure falls back to computing or converting; a version or converter change changes the key, so a stale entry is never served.

## 3. Model, graph and workflow

The model and the graph live both in React state (`useStudio`) and in the viewer's scene (`ViewerManager`), kept in step by hand. Most findings are a place where one of them moved on and the other did not.

### Machines

#### L1. Model load (`App.jsx`: `handleFile`, `openExample`, `openLod1`, `loadSeqRef`; `useStudio`: `beginLoad`, `loadState`, `file`, `modelInfo`, `semantics`)

| State | Representation |
|---|---|
| None | `file` and `modelInfo` null |
| Downloading (example) | `loadState` "downloading" |
| Loading | `loadState` busy, `file` set, `modelInfo` null; `loadSeqRef` counts loads |
| Shown | `modelInfo` set; the semantics may still be on their way |

| Event | From | To |
|---|---|---|
| a file opened | any | Loading for a new `seq`; the downstream reset (`beginLoad`); a graph build and a fire start for the old model end *(SM-7, C1, C2)* |
| an example clicked | any | Downloading, then a file opened; a file opened meanwhile wins |
| the viewer not ready yet | Downloading | None, the card ended *(SM-7, C9)* |
| a rooms model opened | any | Shown at once; an IFC still loading dropped with its card *(SM-7, C4)* |
| the load answers, `seq` still latest | Loading | Shown |
| the load answers for an older `seq` | any | ignored |
| the viewer mode switched away | any | None *(SM-7, C3)* |
| Guided runs, back to the welcome screen (`backToWelcome`, 2026-11-12) | any | None: the guided run ends as its Stop ends it; a fire playing or starting is stopped as Stop fire stops it (M7) and a recording saved as its Stop saves it (M9); then as for a file opened (the load's `seq` bumped, the sampler, RL, plan, report, egress and slime workers ended) with no new file; the viewer drops a load still running and its model and layers (`ViewerManager.unloadModel`); a Flamescapes run still computing or repeating is ended by the graph's effect (M4) |

Invariants: only the latest load writes the page; `loadState` belongs to the latest load or build; the viewer shows the model `modelInfo` describes.

#### L2. The viewer's load (`ViewerManager.loadIfc`, `loadLod1`, `clearModel`) and L3. the import job (`_importJob`, `workers/ifc-import.worker.js`)

Every await in `loadIfc` is followed by a check of the load's token, its signal and disposal; a model loaded for an overtaken load is disposed, not adopted; `clearModel` removes the second models, the rooms meshes, overlays and the graph's pick nodes. The import job settles once and its worker is ended on every path. One abort slot is shared by the main and second models' jobs (C16, open).

#### L4. Second (linked) models (`App.jsx addLinkedModel`; `ViewerManager.loadLinkedIfc`, `linked`)

Adding, added, removed. Kept per main model (`linked.owner === modelInfo`). The owner is checked after each await; a model drawn for a replaced owner is now removed and disposed *(SM-7, C5)*. Placed with the main model's semantics as they were at the click (C10, open).

#### G1. Graph build (`App.jsx buildGraph`, `buildingRef`, `graphBusy`, the sampler worker)

| State | Representation |
|---|---|
| Idle | `buildingRef` null |
| Extracting, sampling | `buildingRef` holds the load's `seq`; `graphBusy`; `loadState` "extracting", "sampling" |
| Built | `graph` set and drawn |

| Event | From | To |
|---|---|---|
| Build (button or `b`) | Idle, or a build for an older model | Extracting *(a second for the same model is ignored, as are key repeats: SM-7, C7)* |
| extraction or sampling answers, same model | Extracting, sampling | Built: the route, alternatives, comparison, cross-check, plan, fire (step, burning nodes, a fire still starting) and agent reset *(SM-7, C6, C14, C2)* |
| another model opened | Extracting, sampling | the worker cancelled; the answer dropped; the busy flags left to the new load *(SM-7, C1)* |

Invariants: the graph on screen was built from the model on screen; one build at a time; a rebuild leaves nothing computed on the old graph on screen. The Flamescapes run, code check, plans, report and people leaving are kept with the graph object and drop themselves.

#### G2. Points and markers (`useStudio.points`, `pickMode`; `handlePick`, `clearPoints`; `ViewerManager.markers`)

The points are set by a pick, the keyboard path, a guided run, or cleared by a new model or Clear; the markers follow the points in an effect *(SM-7, C8)*. `pickMode` is reset on a new model and after each pick.

#### W1. Workflow and stepper (`tab`, `setTab`, step status; `ui/Stepper.jsx`)

Presentational: the active tab is always valid, the status of each step is derived, not stored; `busy` (a load, a build, a fire running or starting, a guided run) is shown, not enforced.

#### V1. Viewer lifecycle (`ui/Viewport.jsx`; `ViewerManager.init`, `dispose`; Viewer mode)

Starting, ready, disposed. The topology viewer's mode unmounts the model's viewer and a new one is made on return; the model's state now goes with the old viewer *(SM-7, C3)*. A manager whose `init` ends after unmount is disposed. No handling of a WebGL context loss of its own (C11, open).

#### I1. Identification (`identifyState`, `identifyDataRef`, `modelKey`)

Kept per model (`modelKey === modelInfo`); an answer for a model no longer open is dropped *(SM-7, C15)*.

#### T1. Topology inspector (`topology`, `topologyBusy`; `viewer/TopologyViewer.jsx`)

Its server actions are disabled while busy; a new payload resets the selection; unmounting removes its listeners and stops its loop. Sound.

### Findings

| Id | Finding | Status | Impact | Issue |
|---|---|---|---|---|
| C1 | A build overtaken by a model load lands on the new model | Confirmed | High | SM-7 |
| C2 | A two-zone fire still starting survives a model change or a rebuild | Confirmed | Medium | SM-7 |
| C3 | Topology JSON and back leaves the state on an empty viewer | Confirmed | Medium | SM-7 |
| C4 | A rooms model opened during an IFC load leaves the progress card | Confirmed | Medium | SM-7 |
| C5 | A second model being added when the first is replaced stays in the scene | Confirmed | Medium | SM-7 |
| C6 | A rebuild keeps the last fire; the export writes it onto the new graph | Confirmed | Medium | SM-7 |
| C7 | The `b` key builds while a build runs, once per key repeat | Confirmed | Medium | SM-7 |
| C8 | Markers stay after another model is opened | Confirmed | Medium to low | SM-7 |
| C9 | An example clicked before the viewer is ready leaves "Downloading" | Confirmed | Medium to low | SM-7 |
| C10 | A second model added before the first's semantics is placed without them | Confirmed | Medium to low | open |
| C11 | After a WebGL context loss the storey cut no longer clips | Confirmed by trace | Low to medium | open |
| C12 | A code check run before the semantics keeps the fallback exits | Suspected | Low to medium | open |
| C13 | Overhead meshes keyed by file name | Confirmed | Low | open |
| C14 | A rebuild keeps the trained agent's result | Confirmed | Low | SM-7 |
| C15 | A late identification answer overwrites the newer model's | Confirmed | Low | SM-7 |
| C16 | One abort slot for imports | Confirmed | Low | open |

Evidence, as traced (lines at 4f7d57b):

- **C1.** `buildGraph` (App.jsx:750-898) awaited the extraction and the sampler worker with no check of `loadSeqRef`, then set and drew the graph, said "Graph ready" and moved to Route; its `finally` set `loadState` idle under the new load. `handleFile` did not cancel the sampler.
- **C2.** `buildGraph` called the hook's `stopFire`, which does not bump `fireStartRef` or cancel the zone lane; the model opens did neither, so `startFire`'s continuation after the two-zone frames went on.
- **C3.** The header's Viewer mode unmounts `Viewport` (and its `ViewerManager`) in Topology JSON mode; nothing reset `file`, `modelInfo`, `graph` or `points`.
- **C4.** `openLod1` bumped `loadSeqRef` but never reset `loadState`, which the overtaken load would have reset only if still latest.
- **C5.** `loadLinkedIfc` added the object to the scene (ViewerManager.js:808), then awaited and checked the owner (:810-813), throwing with the object still in the scene and not in `linked`.
- **C6.** A rebuild left `fire.step`, `fire.model` and `burningRef`; `exportTopologic` writes the burning flags when `fire.step > 0` (App.jsx:1916, :1922).
- **C7.** The `b` hotkey called `buildGraph` with no busy check and on every key repeat (App.jsx:3572).
- **C8.** `beginLoad` cleared `points` but nothing cleared the markers in the viewer (`setMarker` was called only by picks and Clear).
- **C9.** `openExample` set "downloading" and `handleFile`'s "viewer still starting" path returned without resetting it.
- **C15.** `identifyElements` wrote `identifyDataRef` and `identifyState` after its awaits with no check.

### Checked and sound

- `ViewerManager.loadIfc`: every await followed by a check of token, signal and disposal; an overtaken load's model disposed; a prepared model's id carries the load's token; `clearModel` removes second models, rooms meshes, overlays and the graph's pick nodes.
- `handleFile`: progress, model info, semantics, storeys, extraction and the card's reset are all behind `latest()`; `beginLoad` resets points, pick mode, semantics and everything downstream.
- `openExample`: a file opened during the download wins; a double click downloads twice and the second wins (wasted, not wrong).
- The import job: job and semantics settle once; the worker is ended on every path; a job aborted before the worker is ready never posts.
- `loadLinkedIfc` up to the scene: the owner checked after each await; adding is disabled while busy; the list is kept per main model.
- `core/federation.js`: pure; matrices validated.
- Picking: `pickMode` reset on a new model and after each pick; picks snap to the graph's nodes when there is one.
- Viewport mount: a manager whose `init` ends after unmount is disposed; `dispose` aborts the import and ends the warm worker.
- Viewer settings pushed by effects (graph shown, appearance, theme, pick mode, exit rings, sprinkler heads, the reference graph) follow state on each change.
