# Why the GPU released more heat than the CPU (2026-11-09)

The user asked why, in the sprinkler phase of `tools/flamescapes-gpu-check.mjs`, the GPU released about twice the CPU's heat (42.9 against 18.7 MJ), and which backend is the more accurate. The fire is the phase's test room (5 x 4 x 2.7 m, 0.25 m cells, 14,784 cells) with a 200 kW pilot under the first item for 60 s and no heads.

## 1. The figure

The 18.7 against 42.9 MJ came from a `--quick` run, which stops at 60 s. Over the full 120 s the same fire released 47.5 MJ on the CPU and 67.6 MJ on the GPU. The gap is real but smaller, and most of it is in one event: about 31 s in, the pyrolysis vapour gathered under the ceiling (about 1 kg, some 17 MJ) lights at once, the flare reaching 5 to 10 MW for a second or two. After it the fire either holds at 2.5 to 3.5 MW for some 10 s ("holds") or falls back to about 100 kW and flares again near 65 s.

Whether the room flashes over is itself a matter of chance on both paths. Six runs with the room 0.001 K warmer each released 43 to 208 MJ on the CPU and 69 to 626 MJ on the GPU.

## 2. Three differences between the paths, fixed

`--phase spread` (new) runs the fire as an ensemble on each path, hands the CPU's state at a given time to the GPU and to CPU copies, and keeps an energy budget of single steps. With radiation off the paths agreed throughout; the differences were in what radiation and the flame-front particles do.

1. Radiation's deposits. The CPU added each ray's energy as the ray went, so a cell later in the loop re-emitted, in the same step, heat it had just taken: the result depended on the cells' numbering. The GPU emits from the stage's starting temperatures and adds the deposits after. The CPU now does the same (`flamescapes.js radiation`); a test checks that the heat radiated in a step does not depend on the rays.
2. The flame-particle cap. Flame-front particles may fill 15,000 of the 20,000 slots. During a flare about 1,500 cells burn at once and spawn six each, so some 6,000 spawns a step are refused. The CPU refused the cells numbered last, always the same end of the room; the GPU refused whichever threads came last. Both now refuse by the same hash of cell and step (`particleStep`, `kernels.js spawnFlame`), so every burning cell has the same chance; a test checks the spread.
3. The rays' random numbers. The CPU drew rays from one stream and the GPU from a stream per cell. The CPU now draws from the GPU's per-cell PCG streams (`cellStream`, `pcgHash`, checked against BigInt arithmetic in a test), so both paths cast the same rays.

With the three, one step from the same state after a flare agrees on both paths to the decimal (radiated 632.0 kJ on both, released 1,443.5 kJ, the heat left in the fuel, walls and gas within 0.1 kJ), and ten seconds through a flare track each other where a CPU copy with other rays goes its own way. The sprinkler validation against NIST's tests is unchanged to two decimals (its methane burner makes no soot and spawns no flame particles).

## 3. What remains

Over whole runs the fire held after the flare in 16 of 16 GPU runs and 8 of 16 CPU runs (Fisher's exact test, two-sided p = 0.002); median release 75.2 against 72.2 MJ. Before the third fix it was 16 of 16 against 7 of 16 (p = 0.0008), and before any, 6 of 6 against 0 of 6. No window tested from a shared state shows the bias: pyrolysis per step is identical to the bit, and from the same state at 5 s the GPU's vapour follows the CPU's more closely than CPU copies with 1e-4 temperature noise. The run-up is chaotic (the vapour held at 10 s ranged 0.14 to 0.26 kg between runs 0.001 K apart), so the bias, if it is one, lies in something a whole run accumulates. Open; the spread phase is the tool to continue with.

Neither path is validated for this: both solve the same equations, and how much a room burns after a vapour flare has not been compared with any measurement.

## 4. For the calibration plan

The particle budget decides how much a room burns after a flare. With room for every particle (100,000 slots) the three seconds after the flare released 8.3 MJ on both paths, against 5.0 to 5.6 MJ at the default 20,000; over 60 s the CPU released 28.8 MJ against 16.8 MJ, at 27.2 against 19.1 ms a step during the fire. The budget is a fixed count, not a share of the grid, so a large GPU grid thins its flame front more than a small one. It belongs with the flame-front pilot among Stage 3's parameters (`flamescapes-calibration-plan-2026-11-08.md`, section 7).

## Records

The ensembles and probes are in the session's logs; `node tools/flamescapes-gpu-check.mjs --phase spread [--members 16]` reproduces them (about 9 minutes with 16 members on the development machine).
