# State machines

The app's explicit and implicit state machines: their states, transitions and invariants, the invalid states or transitions the code can reach (with the event sequence that reaches them), and the fixes. Kept current as the code changes: a fix updates its machine's table and its issue's status here.

Paths are relative to `topologicpy-web-frontend/src/` unless they start with `tools/`. Line numbers in the evidence are at commit 979d759, before the fixes; functions are named so they can be found after.

## How this was done

One area at a time, most critical first. For each area a read-only pass maps the machines and lists findings; every finding is then traced again by hand before it is called confirmed. Confirmed bugs are grouped into issues (SM-n below), fixed on `next/browser-core` with a regression test that fails on the code before the fix and passes after it, and recorded in [../issues/known-bugs-fixed.md](../issues/known-bugs-fixed.md); open ones go to [../issues/open-risks.md](../issues/open-risks.md).

A finding is **confirmed** when the full event sequence is traced line by line in the shipped code, **suspected** when a step is unproven (what would settle it is stated), and **refuted** when tracing shows the state cannot be reached.

| Area | Scope | Status |
|---|---|---|
| 1. Fire runs and compute workers | `lib/workerJob.js`, `workers/*` (not the IFC import), `core/flamescapesRun.js`, GPU init and device loss, `lib/recorder.js`, the run, plan, people and recording state in `App.jsx` | mapped; 11 confirmed, 1 suspected; fixed (SM-1 to SM-3) |
| 2. Network, backend and storage | `lib/api.js`, `lib/turnstile.js`, `lib/describe.js`, `core/runCache.js`, `lib/fragmentCache.js`, `state/useFieldNotes.js`, the backend's `app/store.py` and routers, `cloudflare/` | not started |
| 3. Model, graph and workflow | `state/useStudio.js`, `ui/Stepper.jsx`, model loading and the IFC import worker, `core/federation.js`, graph and route invalidation, `viewer/ViewerManager.js`, `viewer/TopologyViewer.jsx` | not started |

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

### Open

- A9 (suspected, low): when the GPU solver falls back to the CPU on its own (`flamescapesGpu.js runSceneGpu`: device already lost, scene unsupported, or a buffer setup error), the run keeps the GPU label and the GPU-sized grid, and with the "speed" grid a CPU-measured rate could be stored as that GPU's. To settle: force a buffer setup error in a WebGPU browser and read the card's backend line and `info.gpuRate`. Suggested fix: let `runSceneGpu` throw so the worker's labelled, re-sized CPU restart runs. In [../issues/open-risks.md](../issues/open-risks.md).
- Not reachable today, noted: a throw inside `repeat()` after a successful run would reject `run.promise` with `failed` still false, and that run would be reused for its key. No call in `repeat()` throws in practice.

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

Start is enabled only in Idle (or after an end); Plan and the fire settings wait while Starting or Playing; Stop is enabled while Starting or Playing. A start continues after its frames only if no stop or newer start came (`fireStartRef`).

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

Pending.

## 3. Model, graph and workflow

Pending.
