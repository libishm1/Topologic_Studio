# Fire calibration: the claim protocol (2026-11-12)

Written on 2026-11-12, before any held-out comparison on the solver with proposal F, at the user's request "finish tasks that give us the defensive claim the fire is calibrated". It continues [fire-calibration-plan-2026-11-11.md](fire-calibration-plan-2026-11-11.md) (its tiers, statistics and sources stand) and fixes, ahead of the results, what will be compared, how a fit is chosen, and what may be claimed. Nothing in it may be changed after the held-out results are seen; anything added later is marked as added, with its date.

## 1. What changed in the solver before the comparisons, and why

Each change is justified by a check that uses no validation data.

- **F, heat-conserving transport.** The gas's density and species move through the faces in fluxes, by the advective form d rho / dt + u . grad rho = 0 (density, not T, mixes: gases mix by volume at constant pressure), and heat put into the gas expands it, a divergence ln(T1/T0)/dt in the projection, as FDS's low-Mach form does. Check: `tools/calibration/heat-closure.mjs` (no fire data). A sealed box whose lowest metre starts 300 K hot kept 1.49 times its heat and 0.88 of its mass after 60 s on the old transport, and 1.001 and 1.000 on the new; the burner room's heat closes to 0.96 to 0.97 (3.7 to 5.3 before); a sealed room next to a fire stays at ambient on both.
- **Particles no longer tunnel.** A flame-front particle moved more than a cell a step and jumped walls a cell thick; it now moves in half-cell pieces and stops in the first solid. Check: the sealed-room test in `flamescapes.test.mjs` (smoke and flame stay out of a room joined only by ducts).
- **Harness, not solver:** thermocouples read only cells joined to the fire's gas (the solver's regions). On 0.27 m cells Steckler's corner tree had read a sealed pocket under the ceiling.
- **Inert materials per run** (`createSolver(scene, { inert })`): a validation run gives its walls the test's documented materials; the app's walls stay stone. Thin boards are lumped into whole cells, so their density is scaled by (real thickness / model thickness) to keep the areal heat capacity; the rule is the same for every test and is set from the test's documentation before its comparison is seen.

## 2. The one fitted parameter

The vorticity confinement's strength, refitted on the new transport by the rule of `tools/calibration/fit-confinement.mjs`, unchanged since 2026-11-11: candidates 0, 1, 2, 3, 4, 6 and 10; out if a run diverges or the open plume's centreline flows down in more than 5 % of samples; score the mean of (ln delta)^2 + (sigma_M - sigma_E)^2 over McCaffrey's intermittent temperature, his plume temperature and velocity, and Alpert's ceiling-jet temperature; on 0.27 and 0.34 m cells, the best two also on 0.20 m; the lowest score on all three wins. The training data are these two correlations only. No other parameter is fitted. The walls' convection coefficient stays 10 W/m^2 K (FDS-like, set 2026-11-08 on the sprinkler data, which are therefore not held out for it; section 4 says how that is reported).

## 3. Held-out data and quantities

Measured tests, never used to fit anything:

| Dataset | Lab and year | Quantities |
| --- | --- | --- |
| Steckler, Quintiere and Rinkinen (1982) | NBS | hot gas layer temperature rise and depth (Janssens and Tran), door centreline velocity, neutral plane |
| NBS multi-room (Peacock, Davis and Babrauskas 1991) | NBS/NIST | hot gas layer temperature and depth |
| NIST/NRC (Hamins et al., NUREG/CR-6905) | NIST | hot gas layer temperature and depth, ceiling jet, plume |
| WTC spray burner tests (NIST NCSTAR 1-5B) | NIST | hot gas layer temperature and depth |
| Sprinkler activation (Vettori 1998 and 2003, Bittern 2004, UL/NFPRF series I) | NIST, University of Canterbury, UL | time of the first head's activation |

The exact tests, devices and reductions are those of `tools/calibration/validation-suite.mjs` and `tools/sprinkler-validation.mjs` as they stand when this protocol is written; the validation suite's own section (plan section 12) lists them. Experimental sigma_E as the FDS Validation Guide's Table 4.1 (temperature 0.07, velocity 0.08; layer depth as that table gives it).

## 4. What may be claimed

Per quantity and per cell size, from the held-out results:

- **Calibrated** (the app may say so for that quantity, with its delta, sigma_M, the cell sizes and the fire sizes it holds for): within the calibrated band (0.8 <= delta <= 1.25, sigma_M <= 0.35) in at least three of the independent datasets of section 3 that measure it, and not outside the indicative band (0.67 to 1.5, sigma_M <= 0.5) in any.
- **Indicative**: within the indicative band in at least two.
- **Not calibrated**: otherwise, or measured by fewer than two datasets.

Sprinkler activation is reported apart: the walls' convection coefficient was set on those data, so they are not fully held out for it, and that is said with the number. The overall statement is never "the fire is calibrated"; it is a list of quantities and their status.

### Added 2026-11-12, 00:26 (commit d9160e1), before any held-out verdict

The validation suite as built (plan section 12, pre-registered there before its runs) differs from section 3 in five ways, recorded here and not changed:

- NIST/NRC has no plume comparison: the FDS Validation Guide's inputs give none for it. "Plume" in section 3's NIST/NRC row is struck.
- The suite also compares WTC's ceiling jet, O2 and CO2, and NIST/NRC's O2 and CO2, as the FDS guide does. They are reported; with two datasets from one laboratory they can be indicative at most.
- A dataset with any run stopped by the health guard on a cell counts, on that cell, as outside the indicative band for every quantity it measures. Section 4 did not say; this is the stricter reading.
- NIST/NRC and WTC (one laboratory, one lead, a year apart, different series) count as two datasets; section 3 already listed them so.
- On 0.20 m cells only Steckler and NBS run (NIST/NRC and WTC take about an hour a test there), so nothing can be calibrated on 0.20 m cells.

Where the suite's rule (plan 12.6) and section 4 differ, the stricter gives the verdict.

### Added 2026-11-12, 00:28 (commit 0afce90), before any final-solver Steckler result

A check on geometry alone (the fire's open cells flooded to see whether they reach the domain's open boundary) found that the grid seals some of Steckler's doors: the 0.24 to 0.25 m doors (tests 10, 114 and 514) on every cell size, and the 0.36 m doors (11, 144 and 544) on 0.34 m cells. In those 12 of the 165 Steckler runs the room has no opening, so they do not represent their tests. The rule: a run whose fire's gas cannot reach the open boundary, though its test had an opening, is left out and reported as "not represented on this grid". Tests sealed by design (NIST/NRC 1, 2, 7, 8 and 13 and NBS 100O, door shut, leakage below a cell) are unaffected. The statistics and verdicts are reported both ways, as pre-registered with every test and with this rule; the rule's verdict is the one the app states, and the pre-registered one is given beside it. For the app, the same fact is a limit: an opening narrower than about a cell closes on the grid.

### Added 2026-11-12, after the final held-out run and before its WTC rerun

A second harness fault, found by a check on geometry alone (each tree's modelled ceiling against the test's nominal one): on 0.27 m cells a small instrument bracket in the WTC compartment seals two cells in the west tree's column, so that tree read only the cells below them and missed the top 0.8 m of the layer. A scan of every tree column on every grid finds no other case. The fix: a tree reads every open cell joined to the fire in its column below the test's nominal ceiling, skipping solids. Section 5 asks for every comparison to be rerun after a fix. The six affected WTC runs on 0.27 m are rerun at once, and one unaffected run is rerun to show the solver repeats it bit for bit, so the other 200 runs' readings are unchanged. All 206 are then rerun in full as well, and both are reported: the run as made, and the run with the fix.

## 5. If the comparisons find a fault

A fault found while comparing (in the solver or the harness) may be fixed only when a check that uses no validation data shows it, as section 1's do. After any such fix, every held-out comparison is rerun, and both the results before and after are reported.

### Added 2026-11-12, 06:54 (commit 7292e28), after the final-solver run and before its rerun's verdict

Five records, made before any result of the rerun below is seen; the text above stays as written.

1. **Dates.** This protocol was written on 2026-11-11 and first committed at 22:31 (+0300) in 7e2e9f5, not "on 2026-11-12" as its second line says; its file name keeps the date it was planned for.
2. **Section 3 and the sprinkler row.** Section 3 calls its tests "never used to fit anything", but section 2 says the walls' convection coefficient was set on the sprinkler data, and section 4 reports them apart for that reason. The sprinkler row is not held out for that coefficient.
3. **The bands are this project's.** The calibrated band (0.8 <= delta <= 1.25, sigma_M <= 0.35) and the indicative band (0.67 to 1.5, sigma_M <= 0.5) were set by this project in the 2026-11-08 plan, section 8. The FDS Validation Guide gives the statistics (delta, sigma_M, sigma_E), not these bands.
4. **Two transport faults (section 5).** An independent multi-agent review on 2026-11-12 reproduced two faults with checks that use no measured data:
   - the transport's sweep order favoured one axis (a centred burner under a flat ceiling heated it 11 % more along x than z);
   - porous cells were slowed after the projection, so their gas made heat (a porous burner room closed to 1.27).

   They were fixed in 2ac6141 (CPU) and 6773eae (GPU). The confinement was refitted by section 2's unchanged rule on that solver and stayed at 3 (scores 0.131 for 3, 0.179 for 2, on all three cells). The 218-run held-out comparison was rerun on 0c62bd9 (2026-11-12 02:46 to 04:17, `benchmark/results/2026-11-12-validation-final.json`), which added ATF's corridors as the plan's 12.9 pre-registered.
5. **A burner fault (section 5).** That run stopped two ATF corridor runs on 0.27 m cells on the temperature guard (500 kW at 56 s, 3,105 K; the pulsed fire at 545 s). Tracing the hottest cell found a fault that needs no measured data to show it: the prescribed burner heated each flame cell whatever its oxygen, and the top flame cell under the corridor's ceiling, with no oxygen left, kept taking its share.
   - The fix (6ee8c93, CPU and GPU): a burner's fuel burns only with oxygen. What a flame cell cannot burn goes to the burner's other cells, then to the open cells round them out to Heskestad's flame height. What still finds no oxygen waits.
   - The check: a 200 kW burner in a sealed room releases no more than its oxygen allows and its gas stays below methane's flame temperature (a unit test that fails on the old burner).
   - The same commit gives pools (classes B and K) design fires in the app; no held-out test uses a pool.

   Under section 5 every held-out comparison is rerun on 6ee8c93 (`benchmark/results/2026-11-12-validation-o2.json`). Both runs are reported: the one before this fix with its stopped runs, and the rerun. The claim the app states is the rerun's, by section 4 and the stricter reading of the 00:26 addendum, with the run before it stated beside it.

This fault was found because a held-out run stopped, not by a check made in advance. A reader should weigh the rerun's verdict with that in mind: the fix was chosen by physics that needs no measured data, but the search for it was prompted by held-out data.
