# The Flamescapes backends side by side (2026-11-09)

The user asked to benchmark the WebGPU implementation across cases and to make the CPU fallback reproduce its results without slow, laggy runs for CPU users ("we know WebGPU works accurate and fast and CPU reproduces results"). This note records how the two backends compare as the app runs them, what the CPU fallback costs, and what was changed to make it faster. It follows `gpu-cpu-fire-gap-2026-11-09.md`, which made the two paths solve the same steps.

## 1. Method

`node tools/backend-bench.mjs` (new) runs each case three ways, each four times with the room 1 mK warmer per run (the outcome of a room fire is partly chance, so single runs say little):

- **cpu**: the CPU on the grid the app gives it on this machine (`lib/hardware.js cpuBudget`: 30,000 cells; 15,000 on phones and machines with 4 cores or 4 GB or less);
- **gpu-same**: the GPU on that same grid, which tests whether the CPU reproduces the GPU;
- **gpu-app**: the GPU on the grid the app's worker gives it here: its memory's budget capped by the level its measured speed keeps to 33 ms a step (`flamescapes.worker.js gpuCells`).

The fire is the app's default: a class A first item as a t-squared design fire ("class" growth), 780 MJ/m² of contents, 180 s, frames every 5 s; `room-solver` lets the solver burn the item instead (its own ignition, the case where flares occur). The cases are the GPU check's 5 x 4 x 2.7 m test room and four of the site's examples (Duplex, the NIBS Office and Clinic, Vienna), with tools/fire-bench.mjs's fire origin (the lowest floor's node nearest its centre). Recorded per run: wall time, ms a step and the multiple of real time; energy released, peak heat release and fuel burnt; from the forecast (60 °C, visibility 10 m) the untenable nodes and when the first becomes so; the mean node temperature at the end.

Machine: the development laptop (Intel UHD 770 integrated GPU through D3D12, Chrome 156, 28 logical cores; the CPU path uses one).

## 2. Results

Medians of four runs (range in brackets), 180 s, `benchmark/results/2026-11-09-backends.json`. Released energy includes the design fire's own: 22.7 MJ by 180 s, at 369 kW then, so where a case releases 22.7 MJ the room's contents did not catch.

| Case | Set | Grid (cells, cell) | Wall s | x real time | Released MJ | Peak kW | Untenable nodes | First untenable s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Duplex | cpu | 26,208, 0.370 m, clipped | 49.5 | 3.7 | 26.6 | 373 | 12 | 35 |
| | gpu-same | the same | 5.6 | 32.0 | 26.6 | 372 | 12 | 35 |
| | gpu-app | 245,745, 0.216 m | 61.6 | 2.9 | 27.4 | 383 | 12 | 40 |
| Office | cpu | 27,744, 0.400 m, clipped | 53.4 | 3.4 | 22.8 | 369 | 83 (62-86) | 70 |
| | gpu-same | the same | 5.9 | 30.4 | 22.8 | 369 | 75 (56-80) | 70 |
| | gpu-app | 351,520, 0.370 m, clipped | 55.2 | 3.3 | 23.3 | 407 | 103 (85-116) | 60 |
| Clinic | cpu | 28,512, 0.370 m, clipped | 57.1 | 3.2 | 22.7 | 369 | 46 (40-49) | 90 |
| | gpu-same | the same | 6.2 | 29.2 | 22.7 | 369 | 45 (38-55) | 105 (85-110) |
| | gpu-app | 294,872, 0.317 m, clipped | 69.1 | 2.6 | 22.8 | 369 | 37 (1-46) | 90 |
| Vienna | cpu | 27,744, 0.400 m, clipped | 58.1 | 3.1 | 40.7 | 812 | 380 (358-383) | 20 |
| | gpu-same | the same | 6.1 | 29.7 | 40.6 | 786 | 372 (338-388) | 25 |
| | gpu-app | 279,585, 0.343 m, clipped | 70.7 | 2.6 | 26.4 | 392 | 262 (236-273) | 35 |
| Test room | cpu | 17,732, 0.200 m | 48.4 | 3.8 | 47.9 (37.2-59.4) | 3315 (1348-3452) | 80 | 50 |
| | gpu-same | the same | 5.1 | 35.4 | 36.8 (22.8-38.1) | 1111 (369-1115) | 80 | 55 |
| | gpu-app | the same (its finest cell) | 5.0 | 35.8 | 38.8 (22.8-46.7) | 1287 (369-3421) | 80 | 55 |
| Test room, solver's item | cpu | 17,732, 0.200 m | 35.1 | 5.1 | 0.9 | 12 | 80 | 0 |
| | gpu-same, gpu-app | the same | 4.5 | 40.1 | 1.0 | 11 | 80 | 0 |

What they show:

1. **On the same grid the CPU reproduces the GPU** in all four buildings: the same energy to within 0.5 % and peak to within 3.5 %, and untenable nodes and times within each other's run-to-run range. The GPU is 7.8 to 9.6 times faster there (29 to 40 times real time against 3.1 to 5.1).
2. **The exception is a room where the contents catch.** In the test room the design fire lights the contents; both backends then scatter widely (CPU 37 to 59 MJ, GPU 23 to 38 MJ in four runs each) and their ranges barely touch, this time with the CPU burning more. It is the open gap of `gpu-cpu-fire-gap-2026-11-09.md` (there the GPU's fire held more often); four runs cannot settle it. In the same room with the solver burning the item, it never sustains itself on 0.20 m cells on either backend (as the grid study found).
3. **The device, not the backend, changes the answer.** The app gives the GPU a grid ten times larger: the whole storey at 0.22 to 0.37 m where the CPU gets a box of about 28,000 cells round the fire at 0.37 to 0.40 m. In Vienna the CPU's coarse box burnt twice the contents (peak 812 against 392 kW) and made 380 nodes untenable from 20 s, against 262 from 35 s on the GPU's grid; in the Office the GPU's whole-storey domain reached more nodes (103 against 83). A CPU user and a GPU user see different fires for the same building, and which grid is nearer the truth has not been measured (calibration plan, Stage 0).
4. **The app spends the GPU's speed on resolution.** Sized to 33 ms a step, the GPU's own grid runs at 2.6 to 3.3 times real time, as slow as the CPU (1.03 to 1.24 times its wall time).

## 3. The CPU made faster without changing its results

Profiled on the examples (`tools/fire-bench.mjs`), a CPU step is about half advection and a quarter the pressure solve. Two changes in `flamescapes.js`, each held to a fingerprint of the whole solver state after 300 steps (T, Ts, Y, u, v, w, rho, p, fuel, moisture, particles, energy) that had to stay the same bit for bit (`tools/flamescapes-fingerprint.mjs`, new: f0946dee and 640d9003 before and after):

- **Advection.** Every back-trace starts at a face or a cell centre, where the trilinear weights are 0, 1/2 or 1. Away from the domain's sides the start's velocity is now read from its one to four faces with `sampleFace`'s own products and sums, less the terms weighted 0; three of the seven samples a face takes.
- **The pressure solve.** A cell with all six neighbours open (most of them) sums them unbranched, in the same order.

On the 17,732-cell test room, the commit before (fede128) and after run back to back in Node, twice each: advection 10.0 to 7.5 ms a step (25 % less), the pressure solve 4.7 to 4.0 ms (15 %), the whole step 19.1 to 16.0 ms (16 %).

The CPU being deterministic since the paths share their random streams, the check was repeated on the benchmark's own runs: all 24 CPU runs gave the same answers as before to the last digit (energy, peak, fuel, untenable nodes and times, node temperatures, the whole heat release curve), in 12 to 26 % less time in the buildings (Clinic 57.1 to 42.1 s, Vienna 58.1 to 44.6 s, Office 53.4 to 44.1 s, Duplex 49.5 to 42.9 s; 4.0 to 4.3 times real time instead of 3.1 to 3.6). The test room where the contents flare gained nothing (48.4 against 50.0 s): there particles, radiation and chemistry take more of the step. `benchmark/results/2026-11-09-backends-cpu-optimised.json`.

What was not done, and why: with SharedArrayBuffer one run could use several cores (three to six times on an eight-core machine), but that needs cross-origin isolation, which GitHub Pages cannot set and which would likely block the Turnstile check; WebAssembly for advection and the pressure solve might give 1.5 to 2.5 times, with a new build step. Neither is needed for the CPU to keep ahead of real time on this machine.

## 4. What it means for the app

The benchmark supports the user's premise in part. The GPU is fast (about nine times the CPU on the same grid) and the CPU reproduces it there. But the app does not give them the same grid, so a CPU user does not see the GPU user's fire, and the GPU's speed goes into a finer grid whose accuracy has not been measured. Proposed, for the user to decide:

1. **The same grid on every device by default.** The fire's grid is chosen from the model and the fire, not the device: the CPU's box of about 28,000 cells at the calibrated ladder step. Every device then gives the same answer, the CPU at about 4 times real time and the GPU at about 30.
2. **The GPU's speed spent on repeat runs.** In the CPU's time for one run the GPU makes about eight; the app shows the median and range of what matters (untenable nodes and times, peak heat release) instead of one run. Where the contents catch, the spread is wide (the test room: 23 to 47 MJ on the GPU), and showing it is the honest reading. On the CPU, independent runs can go to other cores in separate workers (no shared memory needed), so a four-core machine gives four runs in about the time of one.
3. **The finest grid kept as an option** ("Grid: finest" already exists), labelled as differing from the default and from the CPU.
4. **Slow CPUs kept responsive.** The CPU's budget is a fixed rule today (15,000 cells on phones and small machines, 30,000 otherwise); measuring the first frame's speed, as the GPU's worker does, would keep slower machines ahead of real time, at the cost of a coarser grid there (and so of reproducing the default). Playback already starts with the first frame.

Which grid is the more accurate is the calibration plan's Stage 0 and Stage 1, not something this benchmark can say.

## 5. Built (2026-11-09, the user's "build 1 to 3")

- **1. The standard grid.** `settings.flamescapesGridPolicy` (was `flamescapesGrid`, migrated by `core/settingsMigration.js`: a stored "finest" stays, the old default "speed" gives way) defaults to `"standard"`: the worker caps every backend at `STANDARD_CELLS` (30,000, `lib/hardware.js standardCells`), so the GPU builds the desktop CPU's grid. A phone's or small machine's CPU keeps `cpuBudget`'s 15,000 so it is not slow, and its run card says the grid is lighter and can differ. The GPU's measured speed is kept only from `"speed"` runs, so a small grid does not size later finer grids too coarse.
- **2. Repeat runs.** `settings.flamescapesRepeats` (1, 4 or 8; one by default, as heavy features are off by default): after a GPU run the page sends its inputs again to the same worker with the run's grid budget, and the worker runs the rest with the room k mK warmer, each answered with `core/forecast.js runSummary` (energy, peak, untenable nodes and first time by the planners' own `forecastFromFrames`). The run card gives the median and range (`runRange`), filled in as they arrive; a newer run ends them; none on the CPU. They start only if the run is still the current one on the same model.
- **3. The finer grids** stay as options ("Finer on the GPU", "Finest on the GPU"), and the card says their fire differs from the standard grid's.
- Proposal 4 (slow CPUs sized by measured speed) was not built.
