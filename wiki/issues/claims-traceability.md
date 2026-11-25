# Claims and their evidence

Last updated: 2026-11-25. One row for each claim the guide, the deck or a paper draft makes about the Studio. A claim with no row here should not be made.

**How to read it.** "Verdict" uses the project's words: calibrated, indicative, not calibrated (fire, against measured tests, band 0.8 to 1.25 with scatter at most 0.35); pass or fail (people model, against published test cases); proved (Lean, logic only). "Checked by" says who looked at the numbers: the suite's own recomputation, an agent, or the coordinator by a second route. Result files are under `benchmark/results/` unless a branch is named. Cells are 0.20 / 0.27 / 0.34 m.

## Fire: the solver on the site

| # | Claim | Test | Result file | Verdict (0.20 / 0.27 / 0.34 m) | Checked by | Plan |
| --- | --- | --- | --- | --- | --- | --- |
| F1 | Smoke layer depth matches measured rooms | Eight test series, 541 pairs | `2026-11-13-validation-13b-020-sealed.json`, `-027-sealed-pockets.json`, `-034-sealed.json` | calibrated / calibrated / calibrated | suite | 13.31 |
| F2 | Height where a door's flow turns | 105 pairs, four series | same | calibrated / indicative / not calibrated | suite | 13.31 |
| F3 | Smoke layer temperature | 541 pairs | same | indicative / not calibrated / not calibrated (reads 35 to 48 % high) | suite | 13.31 |
| F4 | Gas under the ceiling | 291 pairs | same | indicative / not calibrated / not calibrated | suite | 13.31 |
| F5 | Gas temperature at head height | 1157 pairs | same | not calibrated on any cell (reads 33 to 56 % high) | suite | 13.31 |
| F6 | CO₂ under the ceiling | 20 pairs, one laboratory in band | same | indicative / indicative / not calibrated | suite | 13.31 |
| F7 | O₂, smoke obscuration, door speeds | as stored | same | not calibrated on any cell | suite | 13.31 |
| F8 | Time to 60 °C at head height | 877 measured crossings, 269 to 271 resolved | branch `flamescapes/tenability-quantities`: `2026-11-24-tenability-times.md` | not calibrated on any cell (bias 1.32 / 1.09 / 0.92, scatter 0.51 to 0.58) | agent; tool committed before results | 16.56 |
| F9 | Time to a heat dose of 0.3 at head height | 868 measured crossings | same | not calibrated on any cell | agent | 16.56 |
| F10 | O₂ at head height (1.84 m) | 360 pairs, one laboratory | branch `flamescapes/head-height-species`: `2026-11-24-o2-by-height.txt` | not calibrated (model's drop 1.48 / 1.32 / 2.00 of measured) | agent | 16.57 |
| F11 | Time to a visibility of 10 m, under the ceiling | 8 pairs, one laboratory | same branch: `2026-11-24-species-results.md` | not calibrated (0.64 to 0.69: a third too soon); 0.20 m not run | agent; coordinator recomputed | 16.57 |
| F12 | Peak CO, under the ceiling | 9 pairs, two laboratories | same | not calibrated (scatter 1.8; far too low where the fire is short of air); 0.20 m not run | agent | 16.57 |
| F13 | Smoke, CO and toxic dose at head height | none: no data set in hand measures them at 1.5 to 2.0 m | none | **no claim can be made** | n/a | 16.55 |
| F14 | First sprinkler head opens at the measured time | Sprinkler sets, rerun 2026-11-24 | `2026-11-24-sprinkler-validation.json` | median 0.96 to 1.10 of measured, by set; leave-one-laboratory-out not rerun | suite | sprinkler notes |
| F15 | How the app grows its own fire | none: the tests had their fires prescribed | none | **no claim can be made** | n/a | 17 |
| F16 | Agreement with FDS or a zone model, side by side | none | none | **no claim can be made** | n/a | n/a |

## Fire: variants not on the site

| # | Claim | Test | Result file | Verdict | Plan |
| --- | --- | --- | --- | --- | --- |
| V1 | Three radiation fixes (R8b) change the standing | Same series, three cells | solver branch: `2026-11-21-validation-r8b-027.json`, `-034.json`, `2026-11-24-validation-r8b-020.json` | no word moves on any cell | 16.52, 16.58 |
| V2 | Gas radiation (R8) brings head-height temperature toward the band | 0.27 and 0.34 m | solver branch: `2026-11-21-validation-r8-027.json`, `-034.json` | pooled bias 1.10 on 0.27 m, word still not calibrated (scatter between series); 0.20 m not run (87 h against a 72 h limit) | 16.52, 16.58 |

## People model

| # | Claim | Test | Result file | Verdict | Checked by |
| --- | --- | --- | --- | --- | --- |
| P1 | Passes published verification cases | 29 cases from RiMEA 4.1.1, NIST TN 1822, IMO MSC.1/Circ.1533 (`tools/validation/crowd-validation.mjs`; asserted in `src/core/crowdValidation.test.mjs`) | `2026-11-25-crowd-validation-passage-flow.json` and `-capacity-off.json` (PR 26, full run on main 3d7ecde); report `wiki/research/crowd-vv-report-2026-11-24.md` | 28 pass, 1 fail (RiMEA 16, dense single-file flow) with the app's default; 25 of 29 with corridor and stair capacity off | test suite on every run; coordinator ran the harness in full on 2026-11-25 |
| P2 | Everyone who can leave does, in the example buildings | Seven examples, capacity on and off, with and without a fire, in a browser | `wiki/issues/browser-session-2026-11-24.md` | pass | agent, headless Chrome |
| P3 | Walking speed in smoke follows the stated relation | 25 pairs against Korhonen and Hostikka 2009, Eq. 10 | unit test in the suite | pass (the relation itself is an extrapolation in thin smoke) | test suite |
| P4 | Dense flow matches single-file measurements | Calibration with held-out rules | `wiki/research/crowd-calibration-2026-11-24.md` | failed two of three held-out rules; not adopted | agent |
| P5 | Matches a real evacuation | none | none | **no claim can be made** | n/a |

## Algorithms

| # | Claim | Test | Where | Verdict |
| --- | --- | --- | --- | --- |
| A1 | Shortest paths, unit checks, conservation in the transport step and the other stated properties do what their definitions say | Lean 4 proofs | `formal/fireegress/` | proved in exact arithmetic; says nothing about the physics. Not rebuilt for this table. |

## Rules for this table

- A row is added before a claim is written anywhere else, and its verdict is copied from the result, never rounded up.
- "No claim can be made" rows stay in the table: they are the list of what we must not say.
- When a result is superseded, the row cites the new file and the plan section that records the change.
- Rows marked "agent" have not been recomputed by a second route; say so if one is quoted in a paper.
