# Draft for the plan: the tenability quantities on the saved results (2026-11-24)

A draft of a results section in the plan's style, for the coordinator to verify and move. It covers parts A, B and D of 16.54. Part C is another agent's. Nothing in the solver changed, and no solver ran.

### 16.5x The tenability quantities on the saved results: parts A, B and D of 16.54 (2026-11-24)

16.54's parts A, B and D ran as fixed, on branch `flamescapes/tenability-quantities` from 98ca598. No solver ran: every figure is a reduction of saved result files.

- **Tool:** `tools/calibration/lib/crossing.mjs` (the reducer and the rule), `tools/calibration/tenability-times.mjs` (the command), `tools/calibration/lib/crossing.test.mjs` (the check).
- **The check came first.** 13 tests on made-up series: a ramp, a step, a series that never crosses, one that crosses between samples, the dose against hand values, the four outcomes, the exclusions, and the statistic against `lib/stats.mjs`. All pass. They were committed with the choices (43dc492) before the tool read any result file.
- **A check on every file.** The tool's own series give the stored head-height pair for every device of every file: 882 of 882 with eight series and 974 of 974 with nine, none different. So the devices, the cells, the times and the averaging are those of `headTemperature`.
- **Files written:** `benchmark/results/2026-11-24-tenability-choices.md`, `-times.md`, `-times.json`, `-times-rows.json` (every device's two times and outcome), `-item8-loose-ends.md` and `.json`.
- **The 0.20 m step of R8b** had not ended when this was written. Nothing here is a word on R8b or R8 on 0.20 m.

**The 13.B files read.** On 0.20 and 0.34 m: the sealed file (`-13b-<cell>-sealed.json`) and the default-leak file, both. On 0.27 m: the two files with 13.22 merged (`-sealed-pockets`, `-pockets`; 13.31). 13.27's words are the strictest of the two leak readings, and 16.50's point 9 and 16.52 read 13.B sealed, whose pooled head-height δ (1.35 and 1.54) the tool reproduces from the sealed files. On 0.27 m the two files before 13.22 give the same figures as the two after it, to every digit shown: 13.22 dropped cells only in Steckler's trees and one NBS tree outside head height. The ninth series for 13.B is `2026-11-13-validation-fsri-chairs.json` (solver 550fce5), joined in memory. No word differs between the leak readings, or between eight series and nine.

**Choices where 16.54 was silent,** all fixed in `2026-11-24-tenability-choices.md` before any figure:
1. The compared times are `head.mjs`'s: measured times from 1 s to the horizon where both series have a value.
2. A series already at the level at its first compared sample crosses at that sample.
3. "The start" is the first compared sample. A pair is unresolved when the measured crossing is at or before it plus three measured intervals.
4. A miss stays a miss however early the measurement crosses. The misses' share is misses over pairs, unresolved pairs and misses.
5. σ_E is the head-height temperature's 0.07, since the guide gives none for a time.
6. A series is in band as the suite has it: its own statistic in the calibrated band and no stopped run. "Most" is more than half of the series with at least one pair.
7. The three readings. The suite's code has no reading that weights the series alike: its second reading is the guide's 10-bin weighting over the measured value. Computed: unweighted, 10-bin weighted, and series weighted alike (the suite's weighted statistic with series in place of bins), each on every run and with the 2026-11-12 rule. The strictest is taken over all six.
8. With the rule, the runs left out are those each file marks. All are Steckler's, which is steady and out of both quantities. So the rule's reading equals the every-run reading here, figure for figure.

**The result: neither quantity is calibrated, on any cell, with any solver.** The strictest word is "not calibrated" in all 22 readings (eight series and nine).

**Time to 60 °C (`headCrossing60`).** Eight series; 882 devices outside the plume in transient tests. In 877 the measurement crosses. Figures are δ / σ_M.

| Result | Cell | Pairs | Unresolved | Early closures | Misses | Neither | Misses' share | Pooled | Series alike | 10-bin weighted | Series in band | Words: pooled, series alike, rule | Strictest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13.B sealed | 0.20 | 269 | 599 | 1 | 9 | 4 | 1.0 % | 1.32 / 0.58 | 1.76 / 0.71 | 1.45 / 0.51 | 0 of 4 | not calibrated, all three | not calibrated |
| 13.B default leak | 0.20 | 269 | 599 | 1 | 9 | 4 | 1.0 % | 1.37 / 0.58 | 1.85 / 0.74 | 1.48 / 0.52 | 0 of 4 | not calibrated, all three | not calibrated |
| 13.B sealed | 0.27 | 271 | 599 | 1 | 7 | 4 | 0.8 % | 1.09 / 0.51 | 1.53 / 0.87 | 1.34 / 0.97 | 1 of 4 (UL FSRI) | not calibrated, all three | not calibrated |
| 13.B default leak | 0.27 | 271 | 599 | 1 | 7 | 4 | 0.8 % | 1.12 / 0.52 | 1.59 / 0.90 | 1.36 / 0.97 | 1 of 4 (UL FSRI) | not calibrated, all three | not calibrated |
| 13.B sealed | 0.34 | 269 | 599 | 1 | 9 | 4 | 1.0 % | 0.92 / 0.53 | 1.50 / 0.87 | 1.08 / 0.76 | 1 of 4 (NIST/NRC) | not calibrated, all three | not calibrated |
| 13.B default leak | 0.34 | 267 | 599 | 1 | 11 | 4 | 1.3 % | 0.92 / 0.49 | 1.34 / 0.80 | 1.05 / 0.73 | 1 of 4 (NIST/NRC) | not calibrated, all three | not calibrated |
| R8b | 0.27 | 271 | 599 | 1 | 7 | 4 | 0.8 % | 1.10 / 0.52 | 1.53 / 0.88 | 1.31 / 0.92 | 1 of 4 (UL FSRI) | not calibrated, all three | not calibrated |
| R8 | 0.27 | 268 | 599 | 1 | 10 | 4 | 1.1 % | 1.14 / 0.54 | 1.69 / 0.93 | 1.25 / 0.94 | 1 of 4 (UL FSRI) | not calibrated, all three | not calibrated |
| R8b | 0.34 | 271 | 599 | 1 | 7 | 4 | 0.8 % | 0.93 / 0.54 | 1.50 / 0.86 | 1.16 / 0.84 | 1 of 4 (NIST/NRC) | not calibrated, all three | not calibrated |
| R8 | 0.34 | 265 | 599 | 0 | 13 | 5 | 1.5 % | 0.91 / 0.51 | 1.47 / 0.85 | 0.99 / 0.69 | 1 of 4 (NIST/NRC) | not calibrated, all three | not calibrated |

- **The pooled bias is inside the band on 0.27 and 0.34 m. The scatter is what fails: σ_M is 0.49 to 0.58 against 0.35.** The misses pass the 5 % limit everywhere.
- **Two thirds of the devices carry no time.** In 599 of the 877 devices where the measurement crosses, it crosses within three measured intervals of the start: 551 of UL FSRI's 728, all of WTC's 24 and of UL/NIST's 16, 6 of NBS's, and 1 each in NIST/NRC and ATF. The gas burners and the spray fires reach 60 °C at head height inside the unresolved start, which is 20 s in UL FSRI, 40 s in the series measured every 10 s and 60 s in UL/NIST. So the quantity rests on 265 to 271 pairs from four series: UL FSRI 177, NIST/NRC 58, ATF 16 to 20, NBS 12 to 16. WTC and UL/NIST give no pair.
- **By series** (δ / σ_M; 13.B sealed, R8b, R8 on 0.27 m): NIST/NRC 1.27 / 0.25, 1.26 / 0.26, 1.35 / 0.25; ATF 1.26 / 0.84, 1.30 / 0.86, 1.37 / 0.93; NBS 3.90 / 1.50, 3.83 / 1.49, 4.86 / 1.56; UL FSRI 0.89 / 0.27, 0.91 / 0.29, 0.92 / 0.29, in band. On 0.34 m NIST/NRC is in band (1.06 to 1.08 / 0.27 to 0.29) and UL FSRI leaves it (0.71 / 0.28). On 0.20 m none is in band (UL FSRI 1.20 / 0.58).
- **The model is late far from the fire.** NBS's far corridor tree reaches 60 °C at 72 to 225 s in the tests and at 616 to 805 s in the model on 0.27 m. The misses are all in ATF and NBS, most of them at the trees farthest from the fire, where the model's peak rise is well below the measured.
- **The ninth series** is in band on every cell with every solver: the chairs read 1.00 to 1.03 / 0.07 to 0.08 over 92 pairs, none unresolved. With it the pooled figures are 0.94 to 1.27 / 0.43 to 0.51, 1 or 2 series in band of 5, and the words do not change.

**Time to a heat dose of 0.3 (`headHeatDose`).** The measurement crosses in 868 devices; 50 are unresolved.

| Result | Cell | Pairs | Early closures | Misses | Neither | Misses' share | Pooled | Series alike | 10-bin weighted | Series in band | Words: pooled, series alike, rule | Strictest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13.B sealed | 0.20 | 782 | 5 | 36 | 9 | 4.2 % | 0.97 / 0.41 | 1.03 / 0.68 | 1.23 / 0.40 | 0 of 6 | not calibrated, all three | not calibrated |
| 13.B default leak | 0.20 | 781 | 5 | 37 | 9 | 4.3 % | 0.99 / 0.42 | 1.05 / 0.69 | 1.23 / 0.40 | 0 of 6 | not calibrated, all three | not calibrated |
| 13.B sealed | 0.27 | 806 | 6 | 12 | 8 | 1.4 % | 0.98 / 0.32 | 0.78 / 0.58 | 0.90 / 0.49 | 1 of 6 (UL FSRI) | indicative, not calibrated, indicative | not calibrated |
| 13.B default leak | 0.27 | 806 | 6 | 12 | 8 | 1.4 % | 1.01 / 0.33 | 0.79 / 0.59 | 0.91 / 0.49 | 1 of 6 (UL FSRI) | indicative, not calibrated, indicative | not calibrated |
| 13.B sealed | 0.34 | 804 | 5 | 14 | 9 | 1.6 % | 0.79 / 0.37 | 0.74 / 0.60 | 0.72 / 0.52 | 1 of 6 (NIST/NRC) | not calibrated, all three | not calibrated |
| 13.B default leak | 0.34 | 804 | 5 | 14 | 9 | 1.6 % | 0.80 / 0.38 | 0.74 / 0.61 | 0.72 / 0.52 | 1 of 6 (NIST/NRC) | not calibrated, all three | not calibrated |
| R8b | 0.27 | 808 | 5 | 10 | 9 | 1.1 % | 0.99 / 0.32 | 0.82 / 0.62 | 0.91 / 0.51 | 1 of 6 (UL FSRI) | indicative, not calibrated, indicative | not calibrated |
| R8 | 0.27 | 797 | 1 | 21 | 13 | 2.4 % | 1.08 / 0.33 | 0.87 / 0.61 | 0.85 / 0.49 | 1 of 6 (UL FSRI) | indicative, not calibrated, indicative | not calibrated |
| R8b | 0.34 | 804 | 5 | 14 | 9 | 1.6 % | 0.79 / 0.36 | 0.73 / 0.58 | 0.70 / 0.52 | 1 of 6 (NIST/NRC) | not calibrated, all three | not calibrated |
| R8 | 0.34 | 804 | 5 | 14 | 9 | 1.6 % | 0.85 / 0.37 | 0.82 / 0.61 | 0.77 / 0.51 | 2 of 6 (NIST/NRC, UL FSRI) | not calibrated, all three | not calibrated |

- **On 0.27 m the pooled reading is in band with all three solvers** (0.98 to 1.08 / 0.32 to 0.33, misses 1.1 to 2.4 %), which is "indicative" by 16.54's rule in that reading. It is one series: UL FSRI gives 720 of the 797 to 808 pairs and is the only series in band (0.96 to 1.06 / 0.29 to 0.30). Weighted by series or by bins the scatter is 0.49 to 0.62, and the word is "not calibrated". The strictest word stands.
- **By series on 0.27 m** (13.B sealed, R8b, R8): NIST/NRC 1.35 / 0.15, 1.31 / 0.16, 1.48 / 0.15, a small scatter and a late model; ATF 0.87 / 0.52, 0.87 / 0.53, 1.06 / 0.63; NBS 0.77 / 0.57, 1.09 / 0.78, 0.88 / 0.55; UL/NIST 0.38, 0.38, 0.43 over 3 pairs; WTC one pair, at 0.33 to 0.35.
- **On 0.34 m the model closes early.** The pooled δ is 0.79 to 0.85, with UL FSRI at 0.76 (13.B, R8b) and 0.82 (R8). Under 16.54 a δ below 0.8 fails the rule by itself.
- **On 0.20 m the misses are 4.2 to 4.3 %,** just inside the limit. 26 of the 36 are in seven of UL FSRI's 100 kW tests with the door open, where the model reads cold on this cell (in one of them, a peak rise of 51 to 56 K against 84 to 94 K measured).
- **Item 8 on 0.27 m doubles the misses** (10 in R8b, 21 in R8) and removes four of five early closures. Eight of the new misses are in NIST/NRC's tests 1 and 7, where the measured dose ends just above 0.3 and the model's just below (for example 0.32 to 0.46 against 0.24 to 0.26).
- **The ninth series** is in band on every cell (0.93 to 0.97 / 0.07, 92 pairs). Words unchanged.

**B. What the dose and the escape deadline inherit.** Reported, not judged. Eight series, the same 882 devices.

| Result | Cell | Peak rise, δ / σ_M | Model's peak over measured in °C, median | Time to 60 °C: δ, median ratio | Time to dose 0.3: δ, median ratio | Summed dose at the run's end, model over measured: quartiles |
| --- | --- | --- | --- | --- | --- | --- |
| 13.B sealed | 0.20 | 1.33 / 0.37 | 1.20 | 1.32, 1.08 | 0.97, 0.85 | 0.77, 1.43, 4.67 |
| 13.B sealed | 0.27 | 1.37 / 0.39 | 1.19 | 1.09, 0.93 | 0.98, 0.90 | 0.93, 1.40, 4.31 |
| 13.B sealed | 0.34 | 1.68 / 0.42 | 1.59 | 0.92, 0.76 | 0.79, 0.71 | 1.12, 3.94, 11.46 |
| R8b | 0.27 | 1.38 / 0.41 | 1.19 | 1.10, 0.92 | 0.99, 0.90 | 0.94, 1.35, 4.20 |
| R8 | 0.27 | 1.08 / 0.37 | 1.06 | 1.14, 0.96 | 1.08, 0.99 | 0.60, 0.89, 1.74 |
| R8b | 0.34 | 1.68 / 0.41 | 1.58 | 0.93, 0.77 | 0.79, 0.71 | 1.20, 3.89, 11.36 |
| R8 | 0.34 | 1.25 / 0.36 | 1.22 | 0.91, 0.77 | 0.85, 0.77 | 0.72, 1.92, 3.19 |

- **The temperature's bias does not carry into the times one for one.** A peak rise 33 to 68 % high goes with a median time to 60 °C between 24 % early and 8 % late, and a median time to the dose 1 to 29 % early. The times that can be compared come from the devices that cross slowly, far from the fire, where the model is late or cold. The devices near the fire, where it is hot, cross too soon to carry a time.
- **It carries into the summed dose more than one for one,** as T^3.4 says it must. A median peak 19 to 20 % high in °C goes with a median dose 35 to 43 % high at the run's end (1.19^3.4 is 1.81). On 0.34 m a peak 58 to 59 % high goes with a dose 3.9 times the measured (1.59^3.4 is 4.8).
- **Item 8 brings the dose's median near 1 on 0.27 m** (0.89, quartiles 0.60 to 1.74) and halves it on 0.34 m (1.92). The dose time's δ moves from 0.99 to 1.08 and from 0.79 to 0.85. It moves no word.
- **The side of the error.** The model closes a place early more often than late where the times are resolved (median ratios 0.71 to 0.99). The misses, 0.8 to 4.3 % of the devices that cross, are the far devices.

**D. Item 8's loose ends (16.52).** From `tools/calibration/item8-loose-ends.mjs`; tables in `2026-11-24-tenability-item8-loose-ends.md`. None enters a statistic.

*The heat ledgers.* Each run's last row, median share of the heat released, 13.B sealed, R8b, R8:

| Series | Cell | Walls | Out of the grid's sides | Remainder |
| --- | --- | --- | --- | --- |
| UL FSRI | 0.27 | 0.51, 0.51, 0.57 | 0.41, 0.41, 0.35 | 0.030, 0.030, 0.028 |
| UL FSRI | 0.34 | 0.61, 0.61, 0.66 | 0.28, 0.28, 0.20 | 0.072, 0.072, 0.076 |
| UL/NIST | 0.27 | 0.45, 0.44, 0.51 | 0.46, 0.46, 0.39 | 0.094, 0.072, 0.075 |
| UL/NIST | 0.34 | 0.47, 0.47, 0.57 | 0.38, 0.38, 0.31 | 0.125, 0.109, 0.102 |
| NIST/NRC | 0.27 | 0.74, 0.73, 0.76 | 0.14, 0.16, 0.12 | 0.026, 0.023, 0.023 |
| WTC | 0.27 | 0.69, 0.65, 0.65 | 0.25, 0.29, 0.29 | 0.054, 0.038, 0.042 |
| NBS | 0.27 | 0.81, 0.81, 0.82 | 0.07, 0.08, 0.06 | 0.074, 0.067, 0.067 |
| ATF | 0.27 | 0.82, 0.79, 0.80 | 0.00, 0.00, 0.00 | 0.156, 0.149, 0.145 |
| Steckler | 0.27 | 0.27, 0.25, 0.26 | 0.62, 0.61, 0.60 | 0.051, 0.042, 0.042 |

- **Item 8 moves heat from the door to the walls in the two UL rooms:** the walls' share rises by 0.05 to 0.11 and the share leaving by the grid's sides falls by about as much. These are the rooms 13.27 found with walls taking too little. In the other series the walls' share moves by 0.03 at most. The base arm (items 10, 11 and 14) beside 13.B moves the walls' share by 0.04 at most, downward in WTC and Steckler. FSE is left out of both statements: a run of it stops in some arms.
- **The remainder** (released less gas, walls, sides, leak and `radiatedOut`) is 2 to 16 % in the median by series in every arm, and item 8 moves it by 0.7 % of the heat released at most. `radiatedOut` is 0 to 4 % of the heat released.

*Item 8's tallies,* summed over each series' runs:
- `radNegative` and `radShareNegative` are 0 in every run of both cells.
- `radCapped` is 0 in all but WTC on 0.34 m (0.41 MJ of 8778 MJ released; at most 2 × 10⁻⁴ of a run's heat) and the stopped ISOHept26.
- `radFloor`, what the gas between ambient and 10 K above would have emitted, is at most 2.6 % of a run's heat released (ATF) and under 1.5 % elsewhere.
- `surfaceCapped` is 0. `surfaceFloor` is 1 to 27 MJ a series, the same in the two arms within 4 % in every series but FSE.
- `flameSilenced` with item 8 on: the largest run of a series has 0.04 (Steckler) to 1.5 (WTC on 0.34 m) times its heat released. `flameCleared` is 0 in every R8 run.
- `gasRays` is 5.8 × 10¹⁰ and `gasCrossings` 7.6 × 10¹¹ over the eight series on 0.27 m.

*Both `radRoundoff` sums beside the heat released.* Own, deposits and (R8) burner together, as a share of a run's heat released: the median by series is between −3.7 × 10⁻⁵ and +1.6 × 10⁻⁵, and the largest run is 7.1 × 10⁻⁵ (NBS, R8 on 0.27 m). As a share of the stage's gross emission the largest is 9 × 10⁻⁵. **16.47's float32 finding is negligible in the validation rooms,** in both arms, as 13.36 found for R11. With item 8 on the deposits' sum is negative in every series on 0.27 m, the side 16.47 named (heat leaves the gas that the walls do not gain), at 10⁻⁵ of the heat released.

*Wall time, R8 over R8b, run by run:* medians by series of 5.3 (WTC) to 13.2 (UL FSRI) on 0.27 m and 5.2 to 10.9 on 0.34 m; the largest single run 24.0.

*The stop of FSE ISOHept26.* In every case the check that stops it is the temperature limit, the hottest gas cell above 3000 K (`HEALTH.maxT`). The stored record holds the reason, the time and that one reading, and no field.

| Arm | 0.27 m | 0.34 m |
| --- | --- | --- |
| 13.B | ran to 600 s | ran to 600 s |
| R8b | stopped at 423 s, 3008.6 K | ran to 600 s |
| R8 | stopped at 345 s, 3012.3 K | stopped at 515 s, 3026.7 K |

- The fire is 0.56 MW to 180 s and 2.0 MW from 240 s. Every stop is in the 2 MW stage. ISOHept25 (1.0 MW) and ISONG3 (0.75 MW) run to their end in every arm on both cells.
- The speed and energy checks are not near: the fastest gas is 9.7 to 14.7 m/s where the run ends, and the gas holds 6 to 7 MJ of 400 to 900 MJ released.
- **The base's item 14 is the visible difference.** With it the flame's cells emit nothing of their own. What they would have emitted (`flameSilenced`) is, on 0.27 m in R8b, 0.05 of the heat released at 60 s, 0.79 at 300 s and 1.61 at 420 s, three seconds before the stop. In R8 on 0.27 m it is 1.52 at 300 s; in R8 on 0.34 m, 1.74 at 480 s. In R8b on 0.34 m, which does not stop, it is 0.85 at 480 s and 1.00 at 600 s, and that run's ceiling thermocouples read 1413 and 2023 K of rise against 1376 and 1282 K measured (13.B: 619 and 586 K).
- So where the run stops, the flame's cells are hot enough that their silenced emission passes about 1.5 times the heat released. Item 8 brings the stop forward on 0.27 m and adds it on 0.34 m; its own tallies show `flameSilenced` roughly doubled against the base at the same time.
- **This is a correlation in the ledger and no cause is shown.** WTC's runs reach 1.2 to 1.7 by the same measure and do not stop. The test is 2 MW behind a 0.4 m door, the solver has no extinction (item 6 is not made), and the prescribed heat is released in the burner's cells whatever the oxygen. Whether that is what drives the cells past 3000 K once their own emission is silenced was not tested: it needs a run that stores the hottest cell's place and temperature each second. No run was made for it.
- No FSE figure of R8b or R8 is used on the cells where the run stops (16.52).

*The head-height temperature on 0.27 m, series by series* (δ / σ_M, every run, the files' own statistics; **bold** is in band):

| Series | Pairs | 13.B sealed | R8b | R8 |
| --- | --- | --- | --- | --- |
| NIST/NRC | 59 | **0.84 / 0.16** | **0.92 / 0.16** | 0.75 / 0.15 |
| WTC | 24 | **1.09 / 0.15** | 1.32 / 0.16 | 1.35 / 0.13 |
| Steckler | 220 | 1.27 / 0.11 | **1.21 / 0.12** | **1.16 / 0.10** |
| UL FSRI | 728 | 1.38 / 0.25 | 1.37 / 0.26 | **1.06 / 0.14** |
| NBS | 25 | 1.27 / 0.62 | 1.34 / 0.66 | 1.22 / 0.72 |
| UL/NIST | 16 | 2.14 / 0.19 | 2.15 / 0.18 | 1.48 / 0.27 |
| ATF | 30 | 2.31 / 1.38 | 2.34 / 1.48 | 1.82 / 1.54 |
| Pooled | 1102 | 1.35 / 0.35 | 1.34 / 0.37 | 1.10 / 0.34 |
| Pooled, 10-bin weighted | | 1.53 / 0.51 | 1.60 / 0.54 | 1.15 / 0.52 |
| Pooled without ATF | 1072 | 1.33 / 0.27 | 1.32 / 0.27 | 1.08 / 0.21 |
| The same, 10-bin weighted | | 1.48 / 0.32 | 1.53 / 0.31 | 1.10 / 0.25 |

- **In R8 the word on 0.27 m is held by two things.** The unweighted pooled figure is in band (1.10 / 0.34), but two series are in band where 12.6 asks three, so that reading says "indicative". The weighted pooled scatter is 0.52 against the indicative band's 0.50, so that reading says "not calibrated", and it is the strictest.
- **Most of the scatter is inside the series, not between them.** Of the variance of ln(M/E), 13 to 19 % lies between the series' means. With each series' own mean taken out, σ_M is still 0.30 to 0.34.
- **ATF's 30 pairs carry it.** Its own σ_M is 1.4 to 1.5: its tree nearest the fire reads 2.7 to 3.5 times the measured rise and its farthest tree 0.03 to 0.6 of it (13.B). Without ATF the pooled scatter falls to 0.21 to 0.27 and the weighted to 0.25 to 0.32, both in band in R8. NBS, the other long corridor, has the next largest (0.62 to 0.72) and moves the pooled figure by 0.02 at most. No series is dropped by this: it is where the scatter is, and it is the same far-field error the crossing times show.
- **Item 8 moves series both ways.** UL FSRI comes into band (1.37 to 1.06) and UL/NIST halves its excess (2.15 to 1.48). NIST/NRC leaves the band downward (0.92 to 0.75), and WTC stays above it (1.35). The count in band stays two.
- On 0.34 m R8's weighted scatter is 0.39 and its word is indicative (16.52).

**What is not concluded.**
- Nothing is fitted, and no constant, band or rule changed after a figure was seen. No bug was found in the tool after it read results.
- "Not calibrated" for a crossing time is a statement of scatter and of coverage. It rests on 265 to 271 pairs of 882 devices and four series for 60 °C.
- The unresolved devices are not evidence that the model is right near the fire: they carry no time either way.
- 16.54's "each series weighted alike" is not a reading the suite's code has. The figures for it here are this section's, by a statistic written for it and stated above.
- The cause of ISOHept26's stop is not established.

**What follows, for the coordinator and the user.**
1. R8b on 0.20 m is read with the same command when it ends.
2. A second level for the crossing (16.53, proposal 1) would need its own section. A level above 60 °C would resolve more of the near devices only where the measured rise is slower than three intervals, which the hot rooms' are not.
3. The far-field lateness (NBS, ATF) appears in three places now: the crossing times, the misses, and the head-height scatter. It is a transport finding and none of the radiation items addresses it.
