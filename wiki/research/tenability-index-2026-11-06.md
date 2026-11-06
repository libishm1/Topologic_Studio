# Smoke, gas and tenability measures for the Flamescapes runs (2026-11-06)

Status: built and tested. The user asked for more of the smoke and gas measures in the graph's colours and the exports, and for a combined tenability index supported by the literature and by what the Flamescapes solver computes. This note gives the method, its sources, what was read directly, the checks and the limits.

## What the solver gives, and what is read

The browser Flamescapes solver (`src/core/flamescapes.js`) carries six species per cell: fuel vapour, O2, CO2, H2O, soot and CO. It also carries gas temperature and density. At each graph node and frame, `readNodes` reads the cell at head height (1.8 m above the node's floor). The CPU version is in `flamescapesRun.js` and the GPU version is the `gather` kernel. It now reads six quantities:

- gas temperature;
- visibility;
- CO in ppm;
- soot mass concentration in mg/m³;
- CO2 and O2 in % by volume.

The last three are new. Outside the grid a node reads clear air: 20.9 % O2, no CO2 and no soot. The volume fractions use the same molar masses as the CO reading. The solver's air (Y_O2 0.232, the rest N2) reads as 20.906 % O2.

Unburned hydrocarbons are not offered. The solver's fuel species is a lumped C1 pyrolysate, not a measurable hydrocarbon. PM2.5 and PM10 are not offered either, because the solver has no particle sizes.

## Smoke measures

All the smoke measures follow from the soot concentration rho_s by Bouguer's law. The extinction coefficient is K = sigma_m rho_s, with sigma_m = 8.7 m²/g (Mulholland and Croarkin 2000). The user's own note gives 4.5 to 8.7 m²/g across fuels. From K:

- visibility of a light-reflecting sign, S = 3 / K (Jin 1978), capped at 100 m;
- light lost over one metre, 1 − exp(−K);
- optical density per metre, K / ln 10.

Each is a monotone function of the soot, so all of them shade the graph alike and only the legend changes.

## The tenability index

ISO 13571 takes the time to compromised tenability as "the shortest of four distinct times estimated from consideration of asphyxiant fire gases, irritant fire gases, heat, and visual obscuration due to smoke" (2012, cl. 4.1). Each hazard is treated as acting independently (cl. 4.6 a). The index at a node is therefore the largest of these ratios, each 1 at its limit:

| Criterion | Ratio | Source of the limit |
|---|---|---|
| Heat | (T − T_amb) / (T_lim − T_amb) | The app's untenable temperature, 60 °C by default. ISO 13571:2007 cl. 8.1 NOTE: respiratory-tract burns from saturated air above 60 °C. |
| Smoke | S_lim / S, and 0 at the 100 m cap | 10 m by default; PD 7974-6 gives 5 m in small rooms and 10 m in large spaces. |
| Asphyxiant dose | X_FED / FED_crit | The CO term is φ_CO / 35 000 per minute (ISO 13571:2007 Eq. 2). Above 2 % CO2 by volume it is multiplied by ν_CO2 = exp(φ_CO2 / 5) (Eq. 3, ±20 %). Purser's low-oxygen dose, 1 / exp(8.13 − 0.54 (20.9 − φ_O2)) per minute, is added as FDS implements it. ISO 13571:2007 notes that O2 need not be considered above 13 %, where this term is negligible. |
| Convected-heat dose | X_FED / FED_crit | T^3.4 / 5×10⁷ per minute, lightly clothed (ISO 13571:2007 Eq. 10 and 11). |

FED_crit is 0.3. ISO 13571:2007 cl. 5.2 suggests this value for general occupancies; at 0.3, 11.4 % of the population is more susceptible. ISO 13571:2012 cl. 5.4 requires reduced thresholds below the median value of 1.0, and requires one value for FED and FEC. The doses are summed over frames, at each frame's readings.

### Agreement with the app's existing criteria

The first two criteria are the ones the app already closed a Flamescapes node on (`forecastFromFrames`). The planner now also closes a node when either dose reaches FED_crit, so the plan and the index agree. The planner's routing rate is the convected-heat FED rate plus the asphyxiant FED rate. Without CO2 and O2 it is exactly the previous rate.

### Left out

The solver gives none of the following, so the index leaves them out, and the app says so:

- radiant heat at head height (the solver casts rays, but has no flux at a node);
- irritant gases and HCN (not modelled);
- particle sizes.

## What was read directly

- **ISO 13571:2012**, the SIS preview, read for pages 1 to 5: cl. 4.1, 4.6, 5.4 and Eq. 1. The preview stops before the CO constant and the heat equations.
- **ISO 13571:2007**, a full text posted with a national adoption, read for Eq. 1, 2, 3, 7 to 11, cl. 5.2 and cl. 9.
  - The 2012 edition's numerical constants were not read directly. That they carried over unchanged is supported only by search results quoting the 2012 text: the 35 000 µl/l·min dose for CO, and the CO2 factor above 2 %. Confirm against a full 2012 copy before a publication relies on it.
  - The standard's text is not reproduced here.
- **The FDS User's Guide** (NIST SP 1019, current edition), section "Fractional Effective Dose (FED) and Fractional Irritant Concentration (FIC)", read in the firemodels/fds repository on 2026-11-06. It gives Purser's low-O2 term above, Purser's CO2 hyperventilation factor exp(0.1903 φ_CO2 + 2.0004) / 7.1, and Purser's CO term 2.764e-5 C^1.036.
  - The app uses ISO's simpler CO and CO2 terms. Purser's factor agrees with ISO's to within 3 % at 2 % and 5 % CO2.
  - The DOI 10.6028/NIST.SP.1019 resolves to the version 4 guide, so no DOI is cited for the current one.
- **Purser and McAllister (2016)**, SFPE Handbook chapter, doi:10.1007/978-1-4939-2565-0_63, already in the Methods page. Not read in full this round.

## Checks

- **Unit tests** in `src/core/tenability.test.mjs`, `runTable.test.mjs` and the existing forecast and export tests:
  - the formulas against the cited equations;
  - the visibility against the solver's own function;
  - the doses never falling;
  - the index reaching 1 exactly at each limit;
  - clear air reading 0;
  - the table's doses recomputed by hand.

  255 tests pass in all.
- **CPU and GPU agreement** (`tools/flamescapes-gpu-check.mjs --phase full --quick`, the synthetic room, 120 s, both paths), as correlation over all node frames:
  - soot 0.90, with means 359 and 426 mg/m³;
  - CO2 0.93;
  - O2 0.92;
  - CO 0.93, as before.

  The soot means differ about as much as the heat each path released (0.20 and 0.22 MJ). The two paths are compared statistically, as the fire is chaotic.
- **Speed.** The readings are taken once per frame, not per step.
  - GPU, per step: 3.06 to 3.04 ms at 27 300 cells, and 16.08 to 16.02 ms at 186 760 cells.
  - GPU, a full run through the app's worker: 1.9 s before and after.
  - CPU (`tools/fire-bench.mjs --horizon 30`, the five examples): 33 to 44 ms per step after, against 36 to 45 ms before, within noise.
  - Untenable node counts: identical in all ten runs before and after the dose criteria were added. Within 30 s, heat and smoke decide first.
- **Memory.** Recording soot, CO2 and O2 doubles a stored frame, from 12 to 24 bytes a node. A run whose extra readings would pass 64 MB keeps the basic three, and its run card says so; a switch turns them off.
- **End to end in the app** (Duplex, the guided run "Smoke behind closed doors"):
  - every Show option and measure gave its legend;
  - the CSV had 95 820 rows (1 597 nodes by 60 frames) with every column;
  - the TopologicPy JSON carried the series, the doses, each node's time to an index of 1, and the method's constants;
  - no console errors.

## Limits

- **An uncalibrated simulation.** The inputs come from the Flamescapes solver, which its authors describe as having "a bias towards graphics rather than engineering-grade physics". Its soot and CO yields are those of its fuels, not measured ones. The index therefore ranks places and times under the model; it is not a measured exposure.
- **ISO 13571's own uncertainty.** It gives ±35 % for the CO model, ±20 % for the CO2 factor and ±25 % for the heat equations.
- **The 60 °C limit** is the app's existing tenability temperature, not ISO's dose. The convected-heat dose is the ISO criterion; both count.

## References

- ISO 13571:2007, Life-threatening components of fire: guidelines for the estimation of time available for escape using fire data.
- ISO 13571:2012, Life-threatening components of fire: guidelines for the estimation of time to compromised tenability in fires. https://www.iso.org/standard/56172.html
- McGrattan, K., et al. Fire Dynamics Simulator User's Guide, NIST Special Publication 1019, sixth edition (current).
- Jin, T. (1978). Visibility through fire smoke. Journal of Fire and Flammability 9, 135-155.
- Mulholland, G. W., Croarkin, C. (2000). Specific extinction coefficient of flame generated smoke. Fire and Materials 24, 227-230. https://doi.org/10.1002/1099-1018(200009/10)24:5<227::AID-FAM742>3.0.CO;2-9
- Purser, D. A., McAllister, J. L. (2016). Assessment of hazards to occupants from smoke, toxic gases, and heat. SFPE Handbook of Fire Protection Engineering, 5th edition. https://doi.org/10.1007/978-1-4939-2565-0_63
- BSI PD 7974-6:2019, Application of fire safety engineering principles to the design of buildings, part 6: human factors.
- Wrede, H., Wagner, et al. (2026). Flamescapes. ACM Transactions on Graphics 45(6), 246.
