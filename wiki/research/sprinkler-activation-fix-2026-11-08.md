# Sprinklers opening late, and the large rooms: diagnosis and fix (2026-11-08)

Status: done on `next/browser-core`, not deployed (the release waits for the user). The user asked, after the 977978f release, to "plan, ideate, fix the heads starting late and the large room problem, check Lean, derivations, literature, metrics, and test across IFCs, to make sure this bug is documented and fixed". The surface coefficient had been left to the user's decision on 2026-11-07; this request is that decision.

## 1. The problem as it stood

Two findings of 2026-11-07:

- Against NIST's full-scale tests (`wiki/research/sprinkler-validation-data-2026-11-07.md`, R4 to R8) the activation estimate opened heads late in rooms: the first head at a median 1.68 times the measured time in Vettori's 1998 room, 1.09 in his 2003 room, 1.18 in Bittern's rooms, with 8 of 25 room tests opening no head within twice the measured time, and no head behind the 2003 beams opening.
- In the app's runs (`tools/sprinkler-check.mjs`), heads in the NIBS Clinic's large rooms opened at 63 to 68 s, when the fire was already large, and no suppression mode helped.

## 2. Diagnosis

`tools/activation-diagnose.mjs` (new) runs fires in a model's largest rooms with the heads opening only, and records for the heads of the fire's room what the estimate read (the sensor cell under the ceiling and the cell below it, the layer, the plume's height on the grid), each head's distance on the grid and from its true position, when it opened by the estimate and by the late reading, and, as a reference, when the same link would open under Alpert's ceiling jet alone over ambient air for the run's own heat release at the head's true distance and the room's true height (DETACT-QS). A head under a closed ceiling sees at least that jet, so an estimate later than the reference means the grid's layer added nothing.

Four causes were found, in order of their effect on the app's runs.

D1. Suspended ceilings were not in the fire grid. The importer dropped `IfcCovering` elements typed CEILING (so that they would not be walked on), in the site (`ViewerManager.extractEgressGeometry`) and in the Node tools (`tools/ifc-extract.mjs`). The NIBS Office has 83 and the Clinic 250 of them, 52 to 57 mm thick. Without them the fire's gas rose past the heads to the slab or the roof: in the Office's 610 m² open office the grid's ceiling over the fire was 6.11 m up against the room's 1.75 m, the gas under the heads never passed 22 °C, and no head opened in 120 s, not even under Alpert's jet alone; in the Clinic's mechanical room the grid's ceiling was 3.40 m up against 2.05 m and the cells under the heads read 24 to 34 °C at 0.6 to 1 MW. The heads were also hung too high: with the ceilings, 248 of the Clinic's 269 rooms have a lower ceiling than their space's top, the central waiting area 2.68 m instead of the 4.57 m slab.

D2. The estimate's geometry came from the grid. The plume's height was the open column over the first item's cell, and a head's distance was measured between cell centres. In Vienna's garage the column stopped under a low solid 0.6 m over the fire (the room is 2.0 m high), so the estimate opened heads at 14 to 16 s where Alpert's jet on the true height never opened them; the distance was 3.58 m on the grid against 1.95 m from the head's position.

D3. One surface coefficient, 200 W/(m² K), for fuel and walls alike (validation R6.1). The value is the top of the Flamescapes paper's range for its "Convection coefficient" ([0, 200], Table 5), raised in the port so that a burning item lights its neighbours; on walls and ceilings it takes most of the ceiling jet's heat. In the app's fires its effect on the first head was a few seconds (Duplex 25.0 to 23.7 s at 25 W/(m² K), Clinic 65.7 to 61.5 s); in the validation rooms, with prescribed fires, it was the largest single cause.

D4. The fire grows faster than any design fire. In every model the first item and the contents round it reached about 500 kW within 15 s (Duplex 510 kW, Clinic 558 kW, Office 536 kW at 15 s). An ultra-fast t-squared fire (alpha 0.1876 kW/s²) reaches 500 kW at 52 s. With quick-response heads (RTI 50, C 0.4) under these ceilings, the same link driven by Alpert's jet opens at 250 to 600 kW whatever the growth class (slow 292 s at 250 kW, medium 162 s at 308 kW, fast 94 s at 415 kW, ultra-fast 57 s at 600 kW, for the Duplex living room). So a head opening at about 500 kW is what a correct activation gives on so fast a fire: "the fire was already large" is the fire's growth, not the activation. The Flamescapes fire's growth is not calibrated (README, Limits).

## 3. What was considered

| Idea | Evidence | Cost | Decision |
| --- | --- | --- | --- |
| I1. Suspended ceilings into the fire grid, and as the ceiling the heads hang from (D1) | Office and Clinic diagnostics | none at run time; one more mesh group at import | done |
| I2. Each head's own height over the fire and its distance from its position, in the estimate (D2) | Vienna garage; validation R6.5 (UL's heads 1.43 to 2.87 m on the grid, 2.15 m true) | none | done |
| I3. A walls' coefficient apart from the fuel's (D3) | validation R6.1; literature (section 5) | none (a uniform; same work per face) | done; value in section 5 |
| I4. The layer from more than the top cell: a column reduction (Janssens and Tran 1992, FDS's layer height) or the room's enthalpy in a zone-like layer | validation R6.3; literature (section 5) | a few more cells read per head, gathered on the GPU | open: the remaining quarter in Vettori's 1998 room |
| I5. Evans's (1985) hot-layer substitution in place of "layer plus Alpert" | literature (section 5) | small | not done: neither form bounds the other, and they agree within 4 K early |
| I6. A speed-dependent wall coefficient (FDS's forced and natural convection) | literature | per-face speed, about 6 reads a face | not done: the constant met the target |
| I7. A finer grid near ceilings | validation R7: a finer grid did not help at 200 | large | no |
| I8. The first item as a t-squared design fire (D4) | section 2, D4 | changes every run's fire | the user's decision |
| I9. Spray cooling below boiling (UL's head counts) | validation R6 | small | later: after activation |

## 4. Metrics

- M1, against measured tests (`tools/sprinkler-validation.mjs`): per data set the predicted over measured activation time of each measured head and of the first head per test (median and range), the share of heads opening within twice the measured time, and how many open early. The target is a first-head median within 0.85 to 1.25 in every data set with every first head opening, and no data set's median below 0.8.
- M2, in the app's fires across models (`tools/activation-diagnose.mjs`): the first head's time and the heat release then, the estimate against Alpert's jet alone on the run's own heat release (no later), the plume's height and the heads in sight.
- M3, side effects: the free fire's heat released and peak (`tools/sprinkler-check.mjs`), the CPU and GPU agreement (`tools/flamescapes-gpu-check.mjs`), the time a step takes, the unit tests and the browser suite.

## 5. Literature

The sources, with every DOI checked at Crossref and OpenAlex, are in `wiki/research/sprinkler-activation-literature-2026-11-08.md` (written by an agent; its FDS medians were recomputed here from its saved pairs and agree, and three DOIs were checked again by hand). What decided the fix:

- The walls' coefficient. FDS's default convection model, h = (k/L) max(Nu_free, 0.0296 Re^0.8 Pr^(1/3)) with L = 1 m (FDS Technical Reference, section 7.1.2), gives 4 to 14 W/(m² K) on walls and ceilings in rooms like these and about 1 on floors; Cooper's (1982, doi:10.1115/1.3245113) ceiling correlation gives 38 to 48 at the plume's axis and 7 to 13 at 2.2 m; CFAST's C |ΔT|^(1/3) gives 4 to 8. The note recommends 10 as a constant (5 to 25). No source gives a wall function for a cell that holds the whole jet.
- Evans's (1985) hot-layer substitution and the additive "layer plus Alpert's excess over ambient" bound neither each other: over 36 cases the difference ran from -11 to +19 K, within 4 K at a 35 °C layer. The form is not why heads were late; the layer temperature fed into it is.
- The benchmark. FDS (Validation Guide, NIST SP 1018-3, sixth edition, FDS 6.11.1) has a bias of 1.04 and a relative standard deviation of 0.16 over 272 activation times. On the tests Studio runs: 1.13 (Vettori 1998), 0.73 (2003 smooth), 0.90 (2003 beams), 1.06 (Bittern), 0.98 to 1.09 on UL's first heads; FDS's grids were 0.067 to 0.2 m. The target taken here: each set's median within about 0.15 of FDS's, every first head open.
- Resolution. D*/dx on Studio's 0.2 to 0.45 m cells is 0.8 to 1.9 at 100 kW and 2.1 to 4.8 at 1 MW; FDS's sprinkler runs sat at about 6 to 10. No published sub-grid ceiling jet for coarse grids was found.

## 6. Results

### 6.1 Against measured tests (M1)

The final runs, with every change in place, are `tools/sprinkler-validation.mjs --convection 10,200` (`benchmark/results/2026-11-08-sprinkler-validation.json`, 92 runs). The sweep that chose the value, at 200, 50, 25 and 10 before the thin-element seal (section 6.2), is `benchmark/results/2026-11-08-sprinkler-validation-sweep.json` (184 runs); the seal changed only Bittern's results, by 0.02. Predicted over measured activation time, tabulated or measured fire curves; "opened" counts heads with a time (the runs end at twice the test's last measured activation), the medians are over heads opening within twice their own measured time.

| Data set | Heads | Walls at 200: opened, median | First heads at 200 | Walls at 10: opened, median | First heads at 10 | FDS |
|---|---|---|---|---|---|---|
| Vettori 1998 (9 tests) | 18 | 16, 1.64 | 8 of 9, 1.64 | 18, 1.23 | 9 of 9, 1.26 (1.15 to 1.56) | 1.13 |
| Vettori 2003 smooth (4) | 12 | 12, 1.13 | 4 of 4, 1.13 | 12, 0.79 | 4 of 4, 0.78 (0.75 to 0.91) | 0.73 |
| Vettori 2003 beams (4) | 9 | 0 | 0 of 4 | 9, 1.09 | 4 of 4, 1.11 (1.07 to 1.16) | 0.90 |
| Bittern (8) | 16 | 13, 1.19 | 7 of 8, 1.18 | 16, 1.00 | 8 of 8, 0.99 (0.93 to 1.13) | 1.06 |
| UL/NFPRF (4) | 16 | 16, 1.16 | 4 of 4, 1.14 | 16, 1.10 | 4 of 4, 1.11 (1.03 to 1.20) | about 1.06 |

At 50 and 25 W/(m² K) the medians fell between (1998: 1.38 and 1.27; Bittern: 1.07 and 1.05; 2003 beams: 3 of 4 and 4 of 4 first heads). 10 meets the target in every set but the beamed ceilings (1.11 against FDS's 0.90) and opens all 71 heads; the late reading opens 46 of 71 (none at 200). The 2003 smooth set is early with FDS too.

### 6.2 In the app's fires, across models (M2)

`tools/activation-diagnose.mjs --fires 2 --horizon 120` on the four site examples and four models of the local test corpus (C20 Institute, FM_ARC DigitalHub, a small flat ISSUE_159, FZK-Haus): before the fixes (walls 200, no ceilings, the grid's geometry), and after every fix at walls 200 and 10 (`benchmark/results/2026-11-08-activation-diagnose.json`). The first head in the fire's room by the estimate, and the nearest head under Alpert's jet alone on the run's own heat release at the true height and distance:

| Fire | Before | After, walls 200 | After, walls 10 | Alpert's jet alone |
|---|---|---|---|---|
| NIBS Office, open office 610 m² | none in 120 s (grid ceiling 6.11 m over the fire) | 27.8 s at 516 kW | 26.1 s at 536 kW | 28.2 s |
| NIBS Office, open office 215 m² | 15.2 s at 624 kW (grid ceiling 3.00 m over the fire) | 7.8 s at 178 kW | 8.3 s at 154 kW | 7.8 s |
| Clinic, waiting area 178 m² | 65.1 s at 4.5 MW (heads at the 4.57 m slab) | 22.2 s at 656 kW | 21.8 s at 664 kW | 28.8 s |
| Clinic, mechanical room 226 m² | 65.7 s at 1.34 MW | 48.6 s at 813 kW | 47.1 s at 1.0 MW | 50.6 s |
| Duplex, living room | 25.0 s at 535 kW | 25.4 s at 576 kW | 21.0 s at 556 kW | 29.5 s |
| Vienna, garage | 30.5 s at 253 kW (the plume's column stopped 0.6 m over the first item) | none; the fire stays near 100 kW | none | none |

Of the 14 fires that grew (the C20's attic and Vienna's garage stayed under about 130 kW), at walls 10 the first head opened before Alpert's jet alone would open the nearest head, or with it, in 10, within 0.5 s after it in 3 (the estimate measures from the first item's cell, the reference from the fire's point), and 2.4 s after it in the FZK-Haus living room, a model with no ceilings at the heads' height whose nearest head has no line of sight to the plume. The walls' coefficient moved the app's first heads by a few seconds either way, less than the fire's own scatter between runs; the ceilings and the head geometry made the difference in large rooms. Three side effects were found and fixed on the way: counting ceilings as "something overhead" made two spaces drawn outside count as rooms (the Clinic's "ROOF", the Office's exterior vestibule), so ceilings now only lower the ceiling of rooms the slabs keep inside; a 52 mm ceiling split evenly between two 0.4 m cells (6.5 % each, under the 8 % threshold) did not seal, so the larger half of such a pair now seals (`voxelize.js occupancy`; it also moved Vienna's first item off a raised solid); and in a model with ceilings, a room with none now hangs its heads at the slab within 2 m of its space's top (one stairwell in FM_ARC).

One room is left as found: the Clinic's mechanical room has ceilings over part of it but not over the fire, so its grid ceiling over the fire is the slab (3.40 m) while the heads hang at 2.15 m. The estimate reads each head's own height and opens at 47 s, before Alpert's jet alone, but the fire is at about 1 MW by then and no suppression mode helped there (section 6.3).

### 6.3 Side effects and what the fix changes downstream (M3)

- Sprinkler modes (`tools/sprinkler-check.mjs --fires 3`, 180 s, `benchmark/results/2026-11-08-sprinkler-check.txt`): in 12 fires on the four examples heads opened by 47 s in nine; in those, wetting cut the heat released by 47 to 95 % in eight, Evans's cap by 12 to 95 % and the spray by 0 to 95 %. The exception is the Clinic's mechanical room (wetting +9 %, one run). On 2026-11-07 wetting cut 20 to 79 % in most fires and no mode helped in the Clinic's large rooms (heads at 63 to 68 s); its waiting area now opens at 21 s and wetting cuts 79 %. Wetting stays the default.
- Suppression systems (`tools/systems-check.mjs`, seeds 1 to 3, 180 s, `benchmark/results/2026-11-08-systems-check.txt`): the sprinklers now act at 16 to 39 s (39 to 57 s before). Server rooms on sprinklers: Office 67 % (65 before), Clinic 61 % (28); clean agent 91 and 90 % (95 and 93), with the sprinklers kept 92 and 91 %, the heads now opening before the agent. Class A on sprinklers: Office conference room 41 % (49; one seed flashed though its heads opened at 26 s), Clinic conference room 73 % (66), Clinic examination room 54 % (none: heads opened in one seed of three), Duplex living room 80 % (72). The heptane spill kept alight by its burner: sprinklers 72 % (26), foam 8 % (41): the sprinklers open at 29 s and wet the room's other contents, while the burner keeps the pool burning under the foam.
- The free fire. In the GPU check's 5 x 4 x 2.7 m room (0.25 m cells, a 200 kW burner under a 2 kg item, 120 s, five seeds on the CPU) the heat released was 57 to 73 MJ with walls at 200, and 41 to 48 MJ in four seeds with walls at 10 and 224 MJ in the fifth, where unburnt pyrolysis gas gathered and flashed. With the walls taking less heat the room's typical fire is smaller here, and its flashes rarer but larger. In the models, at walls 10 against 200, the heat released in 120 s moved both ways (Office's 610 m² room 190 against 177 MJ, the Clinic's mechanical room 284 against 410 MJ), within the scatter between seeds.
- CPU and GPU (`tools/flamescapes-gpu-check.mjs --quick`): one step from one state the GPU differs from the CPU by at most 0.022 K, the same as with walls at 200 (the difference predates the change); four perturbed runs each released 0.252 MJ on average on the CPU (0.236 to 0.259) and 0.248 on the GPU (0.227 to 0.267); the head over the burner opened at 32.0 s on the CPU and 31.2 to 32.1 s on the GPU, with 35.4 kg of water on both; single runs of the sprinkler scene released 17 to 19 MJ on the CPU and 42 to 44 MJ (121 in one) on the GPU, both inside the CPU's own spread over seeds above.
- Cost. The walls' coefficient is picked per solid cell on the CPU and by a `select` per face on the GPU, with no extra memory (one more scalar in the uniform block's padding); at about 27,000 cells a step took 29.1 ms on the CPU and 2.70 ms on the GPU. The ceilings add the models' ceiling meshes to the voxelisation (250 in the Clinic).
- Tests: 362 unit tests (four new: ceilings in the fire grid and the walls' coefficient acting on walls only; a 52 mm ceiling split between two cells sealing; each head's own height and distance; a suspended ceiling as the heads' ceiling, which makes no space inside, and a room open to the slab); lint clean. `RUN_CACHE_VERSION` 4, so runs kept in a browser are recomputed.
- The GIFs and video of 1619bec were recorded on 2026-11-07, before the fix; their captions say so.

## 7. Proved in Lean

`formal/fireegress/FireEgress/SprinklerActivation.lean` (35 theorems and lemmas, by an agent; checked here: no `sorry`, every axiom print on `propext`, `Classical.choice`, `Quot.sound`; 38 modules and 1,569 in all, `axioms-all-2026-11-08.txt`):

- `wall_coeff_one_exchange`, `wall_coeff_gas_hotter`: in a cell of gas over a wall, a lower walls' coefficient leaves the gas at least as hot after one exchange and at every step, for any nonnegative heat input (from `exchange_energy` and `exchange_diff`).
- `wall_coeff_activates_no_later`, `wall_coeff_first_activation`: so the link opens no later (with `Sprinklers.activates_no_later`).
- `alpertDT_anti_r`, `alpert_switch`: Alpert's excess never rises with distance, across the switch at r = 0.18 H too, since (5.38/16.9)^3 <= 0.18^2; it is antitone in H only within a branch (`alpertDT_H_switch`), and the speed jumps up 1.09 % at r = 0.15 H (`alpertU_jump`).
- `estimate_activates_no_later`: a hotter layer or more heat opens a head no later at a fixed speed; with the speed free it can open later (`faster_link_can_open_later`: a faster link cools faster once the gas falls), which corrected a claim of the plan.
- `old_distance_error`, `new_distance_error`: a distance between cell centres errs by up to a cell's diagonal, one from the head's own position by up to half of it, and the bound is reached.

The cell model compares one cell under the same heat input in both runs; in the solver the heat input depends on the temperatures, so it does not compare two whole simulations.
