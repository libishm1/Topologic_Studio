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

### Added 2026-11-12, 00:40, before any held-out verdict

The validation suite as built (plan section 12, pre-registered there before its runs) differs from section 3 in five ways, recorded here and not changed:

- NIST/NRC has no plume comparison: the FDS Validation Guide's inputs give none for it. "Plume" in section 3's NIST/NRC row is struck.
- The suite also compares WTC's ceiling jet, O2 and CO2, and NIST/NRC's O2 and CO2, as the FDS guide does. They are reported; with two datasets from one laboratory they can be indicative at most.
- A dataset with any run stopped by the health guard on a cell counts, on that cell, as outside the indicative band for every quantity it measures. Section 4 did not say; this is the stricter reading.
- NIST/NRC and WTC (one laboratory, one lead, a year apart, different series) count as two datasets; section 3 already listed them so.
- On 0.20 m cells only Steckler and NBS run (NIST/NRC and WTC take about an hour a test there), so nothing can be calibrated on 0.20 m cells.

Where the suite's rule (plan 12.6) and section 4 differ, the stricter gives the verdict.

## 5. If the comparisons find a fault

A fault found while comparing (in the solver or the harness) may be fixed only when a check that uses no validation data shows it, as section 1's do. After any such fix, every held-out comparison is rerun, and both the results before and after are reported.
